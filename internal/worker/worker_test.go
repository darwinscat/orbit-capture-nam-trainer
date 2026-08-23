// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"orbit-capture-nam-trainer/internal/applog"
	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/storetest"
	"orbit-capture-nam-trainer/internal/testsupport"
)

// stubDriverArg is a stable "driver" argv token: the worker's recovery guard and
// the manually-spawned orphan both argv-match on its basename, exactly as the real
// deployment matches on trainer_driver.py.
const stubDriverArg = "trainer_driver.py"

// The worker identity every harness pool runs under.
const (
	testWorker   = "test-worker.local"
	testInstance = "inst-test"
	testDriver   = "2222222222222222222222222222222222222222222222222222222222222222"
)

var (
	stubBin  string
	validWav []byte // a real 48 kHz / 30 s capture — the worker validates the header
)

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "stubdriver-*")
	if err != nil {
		panic(err)
	}
	stubBin = filepath.Join(dir, "stubdriver")
	if out, err := exec.Command("go", "build", "-o", stubBin,
		"orbit-capture-nam-trainer/cmd/stubdriver").CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "build stubdriver: %v\n%s", err, out)
		os.Exit(1)
	}
	validWav = testsupport.ValidCapture()
	code := m.Run()
	_ = os.RemoveAll(dir)
	os.Exit(code)
}

func testProfile() (Provenance, bool) {
	return Provenance{NamVersion: storetest.NamVersion, DriverSHA256: testDriver, SignalSHA256: storetest.SignalSHA}, true
}

type harness struct {
	pool  *Pool
	store *store.Store
	base  string

	cmu      sync.Mutex
	lastR    int
	lastQ    int
	cntCalls int
}

func (h *harness) recordCounts(r, q int) {
	h.cmu.Lock()
	h.lastR, h.lastQ, h.cntCalls = r, q, h.cntCalls+1
	h.cmu.Unlock()
}

func (h *harness) counts() (running, queued, calls int) {
	h.cmu.Lock()
	defer h.cmu.Unlock()
	return h.lastR, h.lastQ, h.cntCalls
}

// newHarness opens a private schema (skips without a DSN), registers the worker
// row the jobs.worker FK needs, and builds a pool over the stub driver in mode.
func newHarness(t *testing.T, mode string, stall time.Duration) *harness {
	t.Helper()
	st := storetest.Open(t)
	if _, _, err := st.Heartbeat(context.Background(), store.WorkerInfo{
		Name: testWorker, Instance: testInstance, NamVersion: storetest.NamVersion,
		SchemaVersion: store.SupportedQueueContract, TrainCap: 1, ProbeCap: 1, Ready: true}); err != nil {
		t.Fatalf("register worker: %v", err)
	}
	base := t.TempDir()
	lg, err := applog.Open(filepath.Join(base, "logs", "trainer.log"))
	if err != nil {
		t.Fatalf("applog.Open: %v", err)
	}
	t.Cleanup(func() { lg.Close() })

	signal := filepath.Join(base, "signal.wav")
	if err := os.WriteFile(signal, []byte("SIGNAL"), 0o644); err != nil {
		t.Fatal(err)
	}

	h := &harness{store: st, base: base}
	h.pool = New(Options{
		Store:        st,
		Log:          lg,
		Runner:       ProcessRunner{Python: stubBin, Driver: stubDriverArg, Env: []string{"ONCT_STUB_MODE=" + mode}},
		WorkerName:   testWorker,
		Instance:     testInstance,
		SignalPath:   signal,
		ScratchRoot:  filepath.Join(base, "scratch"),
		Cap:          1,
		StallTimeout: stall,

		OnCounts: h.recordCounts,
		Profile:  testProfile,
		// The harness remembers its pause like a real daemon does, so the tests exercise the path
		// that actually runs rather than the one where the memory is switched off.
		PauseStatePath: filepath.Join(base, "paused"),
	})
	return h
}

// seed queues a job of kind on a fresh take carrying a VALID capture.
func (h *harness) seed(t *testing.T, kind string, epochs int) int64 {
	t.Helper()
	return h.seedWav(t, kind, epochs, validWav)
}

func (h *harness) seedWav(t *testing.T, kind string, epochs int, wav []byte) int64 {
	t.Helper()
	tk := storetest.SeedTake(t, h.store, wav)
	return storetest.InsertJob(t, h.store, storetest.JobSpec{Take: tk, Kind: kind, Epochs: epochs})
}

// seedSucceededParent builds a legitimate succeeded train parent DIRECTLY via the
// store (not the pool, so it is mode-independent) with a stored checkpoint whose
// CONTENT is the epoch count as decimal text — exactly what the real/stub driver
// leaves, so resume_ok reads it back to know where to continue numbering.
func (h *harness) seedSucceededParent(t *testing.T, epochs int) (id int64, tk storetest.Take) {
	t.Helper()
	ctx := context.Background()
	tk = storetest.SeedTake(t, h.store, validWav)
	id = storetest.InsertJob(t, h.store, storetest.JobSpec{Take: tk, Kind: jobs.KindTrain, Epochs: epochs})
	j, ok, err := h.store.ClaimNext(ctx, jobs.LaneTrain, testWorker, testInstance, storetest.NamVersion)
	if err != nil || !ok || j.ID != id {
		t.Fatalf("seed parent claim: ok=%v err=%v id=%d", ok, err, j.ID)
	}
	ok, err = h.store.FinishSucceeded(ctx, id, j.ClaimToken,
		store.Result{Reached: int64(epochs), Nam: []byte("nam-parent")}, Provenance{})
	if err != nil || !ok {
		t.Fatalf("seed parent finish: ok=%v err=%v", ok, err)
	}
	// …and the weights it left on the TAKE, which is what a continuation picks up. Success is a
	// pause: the row stays after the run is over, which is the whole of migration 0007.
	storetest.SeedTakeCheckpoint(t, h.store, tk.ID, epochs, []byte("nam-parent"), []byte(strconv.Itoa(epochs)))
	return id, tk
}

// seedTrainMore queues a train_more child off base on the same take, exactly as the app does:
// start_epoch = the base's reached. There is no snapshot to make any more — the weights are already
// on the take, put there by the run that trained them.
func (h *harness) seedTrainMore(t *testing.T, baseID int64, tk storetest.Take, epochs int) int64 {
	t.Helper()
	base := h.get(t, baseID)
	if base.Reached == nil {
		t.Fatalf("base %d has no reached", baseID)
	}
	id := storetest.InsertJob(t, h.store, storetest.JobSpec{Take: tk, Kind: jobs.KindTrainMore, Epochs: epochs,
		BaseJobID: &baseID, StartEpoch: base.Reached})
	return id
}

// takeCkpt returns the weights the library holds for THIS JOB'S TAKE (nil/ok=false when none). The
// checkpoint moved off the job and onto the take in 0007, so a run's work is asked for by the take.
func (h *harness) takeCkpt(t *testing.T, id int64) ([]byte, bool) {
	t.Helper()
	_, _, ckpt, ok := storetest.TakeCkpt(t, h.store, h.get(t, id).TakeID)
	if !ok || ckpt == nil {
		return nil, false
	}
	return ckpt, true
}

func (h *harness) hasResult(t *testing.T, id int64) bool {
	t.Helper()
	_, ok := storetest.Result(t, h.store, id)
	return ok
}

func (h *harness) modelNam(t *testing.T, id int64) []byte {
	t.Helper()
	nam, ok := storetest.Result(t, h.store, id)
	if !ok {
		return nil
	}
	return nam
}

// The app's commands, written on the row exactly as the app writes them.
func (h *harness) cancel(t *testing.T, id int64) {
	t.Helper()
	storetest.Exec(t, h.store, `UPDATE jobs SET cancel_requested_at = now() WHERE id = $1`, id)
}

func (h *harness) requestStop(t *testing.T, id int64) {
	t.Helper()
	storetest.Exec(t, h.store, `UPDATE jobs SET stop_requested_at = now() WHERE id = $1`, id)
}

func (h *harness) requestLive(t *testing.T, id int64) {
	t.Helper()
	storetest.Exec(t, h.store, `UPDATE jobs SET live_requested_at = now() WHERE id = $1`, id)
}

func (h *harness) start(t *testing.T) {
	t.Helper()
	if err := h.pool.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(h.pool.Stop)
}

// capturingRunner records the Spec each Spawn saw (and whether a ResumeCkpt file
// was on disk at spawn time — materialize writes it before Spawn).
type capturingRunner struct {
	inner Runner

	mu           sync.Mutex
	sawResume    bool
	lastResume   string
	resumeExists bool
	lastSpec     Spec
	spawns       int
}

func (c *capturingRunner) Spawn(spec Spec) (*Proc, error) {
	c.mu.Lock()
	if spec.ResumeCkpt != "" {
		c.sawResume = true
		c.lastResume = spec.ResumeCkpt
		_, err := os.Stat(spec.ResumeCkpt)
		c.resumeExists = err == nil
	}
	c.lastSpec = spec
	c.spawns++
	c.mu.Unlock()
	return c.inner.Spawn(spec)
}

func (c *capturingRunner) spec() (Spec, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lastSpec, c.spawns
}

func (c *capturingRunner) DriverBase() string { return c.inner.DriverBase() }

func (c *capturingRunner) resume() (path string, exists, saw bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lastResume, c.resumeExists, c.sawResume
}

func firstEpochLine(lines []string) int {
	for _, l := range lines {
		if ep := parseEpoch(l); ep >= 0 {
			return ep
		}
	}
	return -1
}

func logContains(lines []string, sub string) bool {
	for _, l := range lines {
		if strings.Contains(l, sub) {
			return true
		}
	}
	return false
}

func (h *harness) get(t *testing.T, id int64) jobs.Job {
	t.Helper()
	j, ok, err := h.store.GetJob(context.Background(), id)
	if err != nil {
		t.Fatalf("GetJob: %v", err)
	}
	if !ok {
		t.Fatalf("job %d missing", id)
	}
	return j
}

func (h *harness) waitState(t *testing.T, id int64, want string, timeout time.Duration) jobs.Job {
	t.Helper()
	deadline := time.Now().Add(timeout)
	var last string
	for time.Now().Before(deadline) {
		j, ok, err := h.store.GetJob(context.Background(), id)
		if err != nil {
			t.Fatalf("GetJob: %v", err)
		}
		if ok {
			if j.State == want {
				return j
			}
			last = j.State
		}
		time.Sleep(15 * time.Millisecond)
	}
	t.Fatalf("job %d never reached %q (last %q)", id, want, last)
	return jobs.Job{}
}

// waitRunningWithPGID waits until the job runs with a recorded pgid and returns it.
func (h *harness) waitRunningWithPGID(t *testing.T, id int64) int {
	t.Helper()
	waitFor(t, 5*time.Second, func() bool {
		j := h.get(t, id)
		return j.State == jobs.StateRunning && j.PGID != nil
	}, "job never reached running with a pgid")
	return *h.get(t, id).PGID
}

func waitFor(t *testing.T, timeout time.Duration, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(15 * time.Millisecond)
	}
	t.Fatal(msg)
}

func processAlive(pid int) bool { return syscall.Kill(pid, 0) == nil }

// unregister must be a compare-and-delete so a lagging teardown of an earlier
// attempt never drops a newer attempt's entry (which would orphan a trainer the
// control poller could no longer reach).
func TestUnregisterIsCompareAndDelete(t *testing.T) {
	h := newHarness(t, "", 0)
	p := h.pool
	a := &procEntry{pgid: 111}
	b := &procEntry{pgid: 222}

	p.register(7, a)
	p.register(7, b) // newer attempt overwrites a

	p.unregister(7, a) // the OLD worker's deferred unregister — must be a no-op
	p.mu.Lock()
	got := p.procs[7]
	p.mu.Unlock()
	if got != b {
		t.Fatalf("unregister(a) dropped the newer entry; procs[7]=%v, want b", got)
	}

	p.unregister(7, b)
	p.mu.Lock()
	_, present := p.procs[7]
	p.mu.Unlock()
	if present {
		t.Error("unregister(b) did not remove b")
	}
}

// Notify must publish queue counts even with no worker running, so the keep-awake
// assertion tracks a backlog queued while the runtime is not yet provisioned.
func TestNotifyPublishesQueueCounts(t *testing.T) {
	h := newHarness(t, "", 0)
	h.seed(t, jobs.KindTrain, 5)

	h.pool.Notify()

	r, q, calls := h.counts()
	if calls == 0 {
		t.Fatal("Notify published no counts")
	}
	if r != 0 || q != 1 {
		t.Errorf("counts = running %d / queued %d, want 0/1", r, q)
	}
}

func TestTrainSuccess(t *testing.T) {
	h := newHarness(t, "train-ok", time.Minute)
	id := h.seed(t, jobs.KindTrain, 5)
	h.start(t)

	j := h.waitState(t, id, jobs.StateSucceeded, 15*time.Second)
	if !h.hasResult(t, id) {
		t.Error("a successful train must store a job_result row")
	}
	if j.Epoch == nil || *j.Epoch != 4 {
		t.Errorf("epoch = %v, want 4 (last of 5)", j.Epoch)
	}
	if j.Reached == nil || *j.Reached != 5 {
		t.Errorf("reached = %v, want 5 (the epochs the trainer was spawned with)", j.Reached)
	}
	if j.SPerEpoch == nil || *j.SPerEpoch <= 0 {
		t.Errorf("s_per_epoch = %v, want > 0", j.SPerEpoch)
	}
	if j.ESR == nil || *j.ESR <= 0 {
		t.Errorf("train esr = %v, want the final DRIVER: esr value", j.ESR)
	}
	if j.Worker == nil || *j.Worker != testWorker || j.ClaimToken == "" || j.PGID != nil || j.FinishedAt == nil {
		t.Errorf("terminal row: worker=%v token=%q pgid=%v finished=%v", j.Worker, j.ClaimToken, j.PGID, j.FinishedAt)
	}
	if j.NamVersion == nil || *j.NamVersion != storetest.NamVersion || j.DriverSHA256 == nil || *j.DriverSHA256 != testDriver || j.SignalSHA256 == nil || *j.SignalSHA256 != storetest.SignalSHA {
		t.Errorf("provenance not stamped: nam=%v driver=%v signal=%v", j.NamVersion, j.DriverSHA256, j.SignalSHA256)
	}
	nam := h.modelNam(t, id)
	var parsed map[string]any
	if err := json.Unmarshal(nam, &parsed); err != nil || len(parsed) == 0 {
		t.Errorf("model is not valid non-trivial JSON: %v", err)
	}
	if ckpt, ok := h.takeCkpt(t, id); !ok || string(ckpt) != "5" {
		t.Errorf("stored ckpt = %q (ok=%v), want \"5\"", ckpt, ok)
	}
	lines, _ := h.store.JobLog(context.Background(), id)
	if len(lines) == 0 {
		t.Error("job_log should have captured stdout lines")
	}
	// The take's audio is the library's — the daemon never drops it.
	if _, _, ok, _ := h.store.TakeAudio(context.Background(), j.TakeID); !ok {
		t.Error("take_audio must survive the run (it is the library's, not the daemon's)")
	}
	// Per-attempt scratch dir removed (deferred RemoveAll lands after the terminal write).
	waitFor(t, 3*time.Second, func() bool {
		entries, _ := os.ReadDir(filepath.Join(h.base, "scratch"))
		return len(entries) == 0
	}, "scratch root not emptied after job (deferred RemoveAll)")
}

func TestTrainFailNonzeroExit(t *testing.T) {
	h := newHarness(t, "train-fail", time.Minute)
	id := h.seed(t, jobs.KindTrain, 5)
	h.start(t)

	j := h.waitState(t, id, jobs.StateFailed, 15*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrTrainFailed {
		t.Errorf("error_code = %v, want train_failed", j.ErrorCode)
	}
	if j.ErrorMessage == nil || *j.ErrorMessage == "" {
		t.Error("error_message should carry the exit detail")
	}
	if h.hasResult(t, id) {
		t.Error("no job_result on failure")
	}
}

func TestStallWatchdog(t *testing.T) {
	h := newHarness(t, "silent-hang", 300*time.Millisecond)
	id := h.seed(t, jobs.KindTrain, 100)
	h.start(t)

	j := h.waitState(t, id, jobs.StateFailed, 10*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrStalled {
		t.Errorf("error_code = %v, want stalled", j.ErrorCode)
	}
}

// cancel_requested_at on a running row: the control poller kills the process
// group and the row ends 'cancelled' with error_code cancelled — nothing kept.
func TestCancelKillsProcessGroup(t *testing.T) {
	// A cancel is heard at the next epoch, so the run has to keep having them.
	h := newHarness(t, "train-keeps-going", time.Minute)
	id := h.seed(t, jobs.KindTrain, 100)
	h.start(t)

	pgid := h.waitRunningWithPGID(t, id)
	if !processAlive(pgid) {
		t.Fatalf("trainer pgid %d should be alive", pgid)
	}

	h.cancel(t, id)

	j := h.waitState(t, id, jobs.StateCancelled, 10*time.Second)
	waitFor(t, 3*time.Second, func() bool { return !processAlive(pgid) },
		fmt.Sprintf("process group %d survived the cancel", pgid))
	if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrCancelled || j.FinishedAt == nil || j.PGID != nil {
		t.Errorf("cancelled row = code %v finished %v pgid %v", j.ErrorCode, j.FinishedAt, j.PGID)
	}
	if h.hasResult(t, id) {
		t.Error("a cancelled run keeps nothing")
	}
}

// The app may take a running row away (it cancels a running job whose worker has
// not been seen for 5 minutes): the poller finds the row no longer ours, kills the
// child, and writes nothing — the app's verdict stands.
func TestLostRowKillsChildAndWritesNothing(t *testing.T) {
	// Losing the row is heard the same way: at the next epoch's write.
	h := newHarness(t, "train-keeps-going", time.Minute)
	id := h.seed(t, jobs.KindTrain, 100)
	h.start(t)
	pgid := h.waitRunningWithPGID(t, id)

	storetest.Exec(t, h.store,
		`UPDATE jobs SET state = 'cancelled', finished_at = now(), error_code = 'cancelled', claim_token = NULL, pgid = NULL WHERE id = $1`, id)

	waitFor(t, 3*time.Second, func() bool { return !processAlive(pgid) },
		fmt.Sprintf("process group %d survived losing its row", pgid))
	// The row stays exactly as the app left it; the pool's fenced writes all miss.
	time.Sleep(300 * time.Millisecond)
	j := h.get(t, id)
	if j.State != jobs.StateCancelled || j.ClaimToken != "" || j.Worker == nil {
		t.Errorf("row after the lost attempt = state %s token %q worker %v", j.State, j.ClaimToken, j.Worker)
	}
	if h.hasResult(t, id) || storetest.Count(t, h.store, `SELECT count(*) FROM job_log WHERE job_id = $1 AND claim_token IS NULL`, id) != 0 {
		t.Error("nothing may be written onto a row that is no longer ours")
	}
}

func TestProbeSelfPassKillsOnVerdict(t *testing.T) {
	h := newHarness(t, "probe-self-pass", time.Minute) // prints verdict then hangs forever
	id := h.seed(t, jobs.KindProbeSelf, jobs.ProbeSelfEpochs)
	h.start(t)

	start := time.Now()
	j := h.waitState(t, id, jobs.StateSucceeded, 10*time.Second)
	if elapsed := time.Since(start); elapsed > 8*time.Second {
		t.Errorf("verdict took %s — kill-on-verdict did not fire (child hangs forever)", elapsed)
	}
	if j.Verdict == nil || *j.Verdict != jobs.VerdictPass {
		t.Errorf("verdict = %v, want pass", j.Verdict)
	}
	if j.ESR == nil || *j.ESR <= 0 {
		t.Errorf("esr = %v, want the replicate ESR", j.ESR)
	}
	if j.Reached != nil || h.hasResult(t, id) {
		t.Error("a probe must store no model and no reached")
	}
}

func TestProbeSelfFail(t *testing.T) {
	h := newHarness(t, "probe-self-fail", time.Minute)
	id := h.seed(t, jobs.KindProbeSelf, jobs.ProbeSelfEpochs)
	h.start(t)

	j := h.waitState(t, id, jobs.StateSucceeded, 10*time.Second)
	if j.Verdict == nil || *j.Verdict != jobs.VerdictFail {
		t.Errorf("verdict = %v, want fail", j.Verdict)
	}
}

func TestProbeSelfCrashIsNoVerdict(t *testing.T) {
	h := newHarness(t, "probe-self-crash", time.Minute) // exits with no verdict
	id := h.seed(t, jobs.KindProbeSelf, jobs.ProbeSelfEpochs)
	h.start(t)

	j := h.waitState(t, id, jobs.StateFailed, 10*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrNoVerdict {
		t.Errorf("error_code = %v, want no_verdict (a crash is not a fail verdict)", j.ErrorCode)
	}
	if j.Verdict != nil {
		t.Errorf("verdict = %v, want nil for a crashed probe", j.Verdict)
	}
}

func TestShutdownRequeuesNotFails(t *testing.T) {
	h := newHarness(t, "train-hang", time.Minute)
	id := h.seed(t, jobs.KindTrain, 100)
	h.start(t)
	pgid := h.waitRunningWithPGID(t, id)

	h.pool.Stop() // graceful shutdown must requeue, never fail

	j := h.get(t, id)
	if j.State != jobs.StateQueued {
		t.Errorf("state = %q after shutdown, want queued (never failed)", j.State)
	}
	if j.PGID != nil || j.Worker != nil || j.WorkerInstance != nil || j.ClaimToken != "" || j.StartedAt != nil {
		t.Errorf("requeued row not released: pgid=%v worker=%v instance=%v token=%q started=%v", j.PGID, j.Worker, j.WorkerInstance, j.ClaimToken, j.StartedAt)
	}
	waitFor(t, 3*time.Second, func() bool { return !processAlive(pgid) },
		"child survived shutdown")
}

// The scheduler guarantee the app needs: a self-ESR verdict returns in seconds
// even while a long train occupies the single training cap, because the probe
// lane drains independently.
func TestProbeRunsConcurrentlyWithLongTrain(t *testing.T) {
	h := newHarness(t, "auto", time.Minute) // stub picks behaviour by epoch count
	long := h.seed(t, jobs.KindTrain, 400)  // hangs, occupying the train lane (cap 1)
	h.start(t)
	waitFor(t, 5*time.Second, func() bool {
		return h.get(t, long).State == jobs.StateRunning
	}, "long train never started")

	probe := h.seed(t, jobs.KindProbeSelf, jobs.ProbeSelfEpochs)
	h.pool.Notify()

	start := time.Now()
	j := h.waitState(t, probe, jobs.StateSucceeded, 10*time.Second)
	if elapsed := time.Since(start); elapsed > 8*time.Second {
		t.Errorf("probe verdict took %s while a train ran — lanes are not independent", elapsed)
	}
	if j.Verdict == nil || *j.Verdict != jobs.VerdictPass {
		t.Errorf("probe verdict = %v, want pass", j.Verdict)
	}
	if st := h.get(t, long).State; st != jobs.StateRunning {
		t.Errorf("long train state = %q, want still running alongside the probe", st)
	}
}

// spawnOrphan starts a "previous-run" child in its own group that never exits on
// its own (recovery must kill it), with the scratch path in its argv.
func (h *harness) spawnOrphan(t *testing.T, scratchDir string, epochs int) int {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(scratchDir, "out"), 0o755); err != nil {
		t.Fatal(err)
	}
	orphan := exec.Command(stubBin, "-u", stubDriverArg,
		"--input", "sig",
		"--output", filepath.Join(scratchDir, "capture.wav"),
		"--outdir", filepath.Join(scratchDir, "out"),
		"--name", "model", "--epochs", strconv.Itoa(epochs), "--arch", "standard")
	orphan.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	orphan.Env = append(os.Environ(), "ONCT_STUB_MODE=silent-hang")
	if err := orphan.Start(); err != nil {
		t.Fatalf("spawn orphan: %v", err)
	}
	pgid := orphan.Process.Pid
	go orphan.Wait() // reap it once recovery kills it, so processAlive flips to false
	waitFor(t, 3*time.Second, func() bool { return processAlive(pgid) }, "orphan should be alive before recovery")
	return pgid
}

// markRunningByPreviousInstance makes a row look like a previous process of THIS
// worker left it: running, claimed, with the orphan's pgid.
func (h *harness) markRunningByPreviousInstance(t *testing.T, id int64, pgid int) {
	t.Helper()
	storetest.Exec(t, h.store,
		`UPDATE jobs SET state = 'running', worker = $2, worker_instance = 'previous-instance',
		        claim_token = gen_random_uuid(), claimed_at = now(), started_at = now(), pgid = $3, epoch = 3
		 WHERE id = $1`, id, testWorker, pgid)
}

func TestRestartRecoveryKillsOrphanAndRequeues(t *testing.T) {
	h := newHarness(t, "train-ok", time.Minute) // the re-run mode after recovery
	id := h.seed(t, jobs.KindTrain, 5)
	pgid := h.spawnOrphan(t, filepath.Join(h.base, "scratch", "job-prev"), 100)
	h.markRunningByPreviousInstance(t, id, pgid)

	// Another worker's running job must NOT be touched by our recovery.
	other := h.seed(t, jobs.KindTrain, 100)
	storetest.Exec(t, h.store, `INSERT INTO workers (name, instance) VALUES ('other.local', 'x')`)
	storetest.Exec(t, h.store,
		`UPDATE jobs SET state = 'running', worker = 'other.local', worker_instance = 'x', claim_token = gen_random_uuid(), pgid = 99999 WHERE id = $1`, other)

	h.start(t) // Start → recovery kills the orphan, requeues, workers re-run to success

	waitFor(t, 3*time.Second, func() bool { return !processAlive(pgid) },
		fmt.Sprintf("recovery did not kill the orphan pgid %d", pgid))

	j := h.waitState(t, id, jobs.StateSucceeded, 15*time.Second)
	if !h.hasResult(t, id) {
		t.Error("requeued job should train to success with a model")
	}
	if j.WorkerInstance == nil || *j.WorkerInstance != testInstance {
		t.Errorf("re-run instance = %v, want the new instance %s", j.WorkerInstance, testInstance)
	}
	if o := h.get(t, other); o.State != jobs.StateRunning || o.PGID == nil || *o.PGID != 99999 {
		t.Errorf("another worker's row was recovered by us: %+v", o)
	}
}

// Pause(killRunning=true) must kill the child, REQUEUE the job (the shutdown
// rule — never a failure), and hold it queued until Resume claims it again.
func TestPauseNowKillsRequeuesAndResumes(t *testing.T) {
	h := newHarness(t, "train-hang", time.Minute)
	id := h.seed(t, jobs.KindTrain, 100)
	h.start(t)
	pgid := h.waitRunningWithPGID(t, id)

	h.pool.Pause(true)
	if !h.pool.Paused() {
		t.Fatal("Paused() = false right after Pause")
	}
	h.waitState(t, id, jobs.StateQueued, 10*time.Second)
	waitFor(t, 3*time.Second, func() bool { return !processAlive(pgid) }, "child survived pause")

	time.Sleep(600 * time.Millisecond)
	if st := h.get(t, id).State; st != jobs.StateQueued {
		t.Fatalf("state = %q while paused, want queued", st)
	}

	h.pool.Resume()
	if h.pool.Paused() {
		t.Fatal("Paused() = true after Resume")
	}
	waitFor(t, 5*time.Second, func() bool {
		return h.get(t, id).State == jobs.StateRunning
	}, "job never reclaimed after Resume")
}

// Pause(killRunning=false) lets the running job finish normally and only stops
// NEW claims: the second queued job must sit still until Resume.
func TestPauseAfterCurrentFinishesRunning(t *testing.T) {
	h := newHarness(t, "train-ok", time.Minute)
	first := h.seed(t, jobs.KindTrain, 5)
	h.start(t)
	waitFor(t, 5*time.Second, func() bool {
		return h.get(t, first).State == jobs.StateRunning
	}, "first job never started")

	h.pool.Pause(false)
	second := h.seed(t, jobs.KindTrain, 5)
	h.pool.Notify()

	h.waitState(t, first, jobs.StateSucceeded, 15*time.Second)

	time.Sleep(600 * time.Millisecond)
	if st := h.get(t, second).State; st != jobs.StateQueued {
		t.Fatalf("second state = %q while paused, want queued", st)
	}

	h.pool.Resume()
	h.waitState(t, second, jobs.StateSucceeded, 15*time.Second)
}

// gatingRunner blocks inside Spawn until released, holding a worker in the
// claim→register window (job running in the DB, nothing in procs yet).
type gatingRunner struct {
	inner   Runner
	entered chan struct{} // closed when Spawn is reached
	release chan struct{} // Spawn proceeds when closed
}

func (g *gatingRunner) Spawn(spec Spec) (*Proc, error) {
	close(g.entered)
	<-g.release
	return g.inner.Spawn(spec)
}

func (g *gatingRunner) DriverBase() string { return g.inner.DriverBase() }

// A kill-Pause that fires while a worker sits between claim and register must
// still catch that job: the procs snapshot cannot see it, so the worker's
// post-register pauseKill check has to kill the child it just spawned.
func TestPauseNowCatchesClaimRegisterWindow(t *testing.T) {
	h := newHarness(t, "train-hang", time.Minute)
	g := &gatingRunner{
		inner:   h.pool.runner,
		entered: make(chan struct{}),
		release: make(chan struct{}),
	}
	h.pool.runner = g
	id := h.seed(t, jobs.KindTrain, 100)
	h.start(t)

	<-g.entered // claimed (running in DB), spawn blocked, NOT registered
	h.pool.Pause(true)
	close(g.release)

	h.waitState(t, id, jobs.StateQueued, 10*time.Second)
	waitFor(t, 3*time.Second, func() bool {
		j := h.get(t, id)
		return j.State == jobs.StateQueued && j.PGID == nil
	}, "pgid not cleared after window-escape requeue")
}

// SetCap must resize the training lane LIVE. Raising it wakes an idle spawned
// worker (a second hang-train claims immediately); Cap() reports the change.
func TestSetCapGrowsLive(t *testing.T) {
	h := newHarness(t, "train-hang", time.Minute)
	h.pool.lanes[0].cap = 2 // spawn width (CapLimit); live cap stays 1
	a := h.seed(t, jobs.KindTrain, 100)
	b := h.seed(t, jobs.KindTrain, 100)
	h.start(t)

	h.waitState(t, a, jobs.StateRunning, 5*time.Second)
	time.Sleep(600 * time.Millisecond)
	if st := h.get(t, b).State; st != jobs.StateQueued {
		t.Fatalf("b state = %q at cap 1, want queued", st)
	}
	if h.pool.Cap() != 1 {
		t.Fatalf("Cap() = %d, want 1", h.pool.Cap())
	}

	h.pool.SetCap(2)
	if h.pool.Cap() != 2 {
		t.Fatalf("Cap() = %d after SetCap(2), want 2", h.pool.Cap())
	}
	h.waitState(t, b, jobs.StateRunning, 5*time.Second)
	if st := h.get(t, a).State; st != jobs.StateRunning {
		t.Errorf("a state = %q, want still running", st)
	}
}

// Lowering the cap kills nothing: running jobs finish their course, and only
// then does the lane narrow — the freed slot above the new cap claims no more.
func TestSetCapShrinksAsJobsFinish(t *testing.T) {
	h := newHarness(t, "auto", time.Minute) // epochs: 5 → train-ok, else train-hang
	h.pool.lanes[0].cap = 2
	h.pool.trainCap.Store(2)
	a := h.seed(t, jobs.KindTrain, 5)
	b := h.seed(t, jobs.KindTrain, 5)
	h.start(t)

	waitFor(t, 5*time.Second, func() bool {
		return h.get(t, a).State == jobs.StateRunning && h.get(t, b).State == jobs.StateRunning
	}, "both jobs never ran at cap 2")

	h.pool.SetCap(1)
	h.waitState(t, a, jobs.StateSucceeded, 15*time.Second)
	h.waitState(t, b, jobs.StateSucceeded, 15*time.Second)

	c := h.seed(t, jobs.KindTrain, 400) // hangs, occupying the single slot
	d := h.seed(t, jobs.KindTrain, 400)
	h.pool.Notify()
	h.waitState(t, c, jobs.StateRunning, 5*time.Second)
	time.Sleep(600 * time.Millisecond)
	if st := h.get(t, d).State; st != jobs.StateQueued {
		t.Fatalf("d state = %q at cap 1, want queued", st)
	}
}

func TestSetCapClamps(t *testing.T) {
	h := newHarness(t, "train-hang", time.Minute)
	h.pool.lanes[0].cap = 2
	h.pool.SetCap(0)
	if h.pool.Cap() != 1 {
		t.Errorf("Cap() = %d after SetCap(0), want 1", h.pool.Cap())
	}
	h.pool.SetCap(99)
	if h.pool.Cap() != 2 {
		t.Errorf("Cap() = %d after SetCap(99), want clamped to 2", h.pool.Cap())
	}
}

// A train_more resumes from the app's job_resume snapshot: the worker materializes
// it to <scratch>/resume.ckpt, passes --resume-from, and the child numbers epochs
// absolutely from start_epoch, exporting a NEW nam + a NEW ckpt (chain-ready).
func TestTrainMoreResumesFromParentCkpt(t *testing.T) {
	h := newHarness(t, "resume_ok", time.Minute)
	cr := &capturingRunner{inner: h.pool.runner}
	h.pool.runner = cr

	parent, tk := h.seedSucceededParent(t, 5)
	child := h.seedTrainMore(t, parent, tk, 12)
	h.start(t)

	j := h.waitState(t, child, jobs.StateSucceeded, 20*time.Second)

	resume, exists, saw := cr.resume()
	if !saw {
		t.Fatal("Spawn never received a --resume-from ckpt for the train_more")
	}
	if filepath.Base(resume) != "resume.ckpt" || !exists {
		t.Errorf("resume ckpt = %q (exists=%v), want a present <scratch>/resume.ckpt", resume, exists)
	}
	if j.StartEpoch == nil || *j.StartEpoch != 5 {
		t.Errorf("start_epoch = %v, want 5", j.StartEpoch)
	}
	if j.Epoch == nil || *j.Epoch != 11 {
		t.Errorf("epoch = %v, want 11 (last of 12, absolute)", j.Epoch)
	}
	if j.Reached == nil || *j.Reached != 12 {
		t.Errorf("reached = %v, want 12", j.Reached)
	}
	lines, _ := h.store.JobLog(context.Background(), child)
	if first := firstEpochLine(lines); first != 5 {
		t.Errorf("first Epoch line = %d, want 5 (resumed numbering)", first)
	}
	if !logContains(lines, "DRIVER: resuming from epoch 5") {
		t.Errorf("job_log missing the resuming banner; got %v", lines)
	}
	if ckpt, ok := h.takeCkpt(t, child); !ok || string(ckpt) != "12" {
		t.Errorf("stored ckpt = %q (ok=%v), want \"12\" (the new total)", ckpt, ok)
	}
	// …and the take keeps the weights the child produced: success is a pause, so a second continuation
	// can pick them up without anybody copying anything.
	if c, ok, _ := h.store.Checkpoint(context.Background(), h.get(t, child).TakeID); !ok || c.Reached != 12 {
		t.Errorf("the take must hold the continuation's weights (ok=%v reached=%d)", ok, c.Reached)
	}
}

// A train_more whose ckpt restore blows up BEFORE any Epoch line failed to prove
// the resume → resume_failed; the log (the traceback) is kept as history.
func TestTrainMoreBadCkptIsResumeFailed(t *testing.T) {
	h := newHarness(t, "resume_badckpt", time.Minute)
	parent, tk := h.seedSucceededParent(t, 5)
	child := h.seedTrainMore(t, parent, tk, 12)
	h.start(t)

	j := h.waitState(t, child, jobs.StateFailed, 15*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrResumeFailed {
		t.Errorf("error_code = %v, want resume_failed (died before any Epoch line)", j.ErrorCode)
	}
	if lines, _ := h.store.JobLog(context.Background(), child); len(lines) == 0 {
		t.Error("job_log should be kept on failure")
	}
}

// A train_more that crashes AFTER the resume demonstrably took (Epoch lines were
// seen) is a plain train_failed — NOT resume_failed.
func TestTrainMoreLateCrashIsTrainFailed(t *testing.T) {
	h := newHarness(t, "train-fail", time.Minute) // prints Epoch lines, then exits nonzero
	parent, tk := h.seedSucceededParent(t, 5)
	child := h.seedTrainMore(t, parent, tk, 12)
	h.start(t)

	j := h.waitState(t, child, jobs.StateFailed, 15*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrTrainFailed {
		t.Errorf("error_code = %v, want train_failed (crashed after resuming)", j.ErrorCode)
	}
}

// A continuation of a take that has no weights must NOT run from scratch: it fails base_unavailable
// before anything is spawned. "Add epochs to this" quietly becoming "train it again from zero" is the
// one thing this path forbids — and since 0007 the take's own row is the only source, so a take whose
// weights were cancelled away, or whose history was picked apart, is exactly this case.
func TestContinuingATakeWithNoWeightsIsBaseUnavailable(t *testing.T) {
	h := newHarness(t, "resume_ok", time.Minute)
	cr := &capturingRunner{inner: h.pool.runner}
	h.pool.runner = cr
	parent, tk := h.seedSucceededParent(t, 5)
	storetest.Exec(t, h.store, `SELECT discard_take_checkpoint($1, 'cancelled')`, tk.ID)
	start := int64(5)
	child := storetest.InsertJob(t, h.store, storetest.JobSpec{Take: tk, Kind: jobs.KindTrainMore, Epochs: 12, BaseJobID: &parent, StartEpoch: &start})
	h.start(t)

	j := h.waitState(t, child, jobs.StateFailed, 15*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrBaseUnavailable {
		t.Errorf("error_code = %v, want base_unavailable", j.ErrorCode)
	}
	if _, spawns := cr.spec(); spawns != 0 {
		t.Errorf("spawns = %d, want 0 (never run a continuation from scratch)", spawns)
	}
}

// The full chain THROUGH THE POOL: a pool-run plain train stores its ckpt, and a
// train_more chained off that pool-made ckpt (snapshotted into job_resume as the
// app does) resumes and stores its own.
func TestChainTrainThenTrainMoreThroughPool(t *testing.T) {
	h := newHarness(t, "auto", time.Minute)
	tk := storetest.SeedTake(t, h.store, validWav)
	parent := storetest.InsertJob(t, h.store, storetest.JobSpec{Take: tk, Kind: jobs.KindTrain, Epochs: 5})
	h.start(t)

	h.waitState(t, parent, jobs.StateSucceeded, 20*time.Second)
	if ckpt, ok := h.takeCkpt(t, parent); !ok || string(ckpt) != "5" {
		t.Fatalf("pool-run train stored ckpt %q (ok=%v), want \"5\"", ckpt, ok)
	}

	child := h.seedTrainMore(t, parent, tk, 12)
	h.pool.Notify()
	j := h.waitState(t, child, jobs.StateSucceeded, 20*time.Second)
	if j.Epoch == nil || *j.Epoch != 11 {
		t.Errorf("child epoch = %v, want 11 (absolute numbering)", j.Epoch)
	}
	if ckpt, ok := h.takeCkpt(t, child); !ok || string(ckpt) != "12" {
		t.Errorf("child ckpt = %q (ok=%v), want \"12\" (chain-ready)", ckpt, ok)
	}
}

// A train_more killed by the stall watchdog BEFORE any Epoch line is `stalled`,
// never resume_failed — the reason-first rule outranks the failure-code selection.
func TestTrainMoreStallBeatsResumeFailed(t *testing.T) {
	h := newHarness(t, "silent-hang", 300*time.Millisecond)
	parent, tk := h.seedSucceededParent(t, 5)
	child := h.seedTrainMore(t, parent, tk, 12)
	h.start(t)

	j := h.waitState(t, child, jobs.StateFailed, 15*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrStalled {
		t.Errorf("error_code = %v, want stalled (stall reason outranks resume_failed)", j.ErrorCode)
	}
}

// A kill -9 of a RUNNING train_more (recovery on the next start) requeues it; the
// app's job_resume snapshot is untouched, so the re-claim resumes and completes.
func TestTrainMoreRecoveryResumesAgain(t *testing.T) {
	h := newHarness(t, "resume_ok", time.Minute)
	parent, tk := h.seedSucceededParent(t, 5)
	child := h.seedTrainMore(t, parent, tk, 12)
	pgid := h.spawnOrphan(t, filepath.Join(h.base, "scratch", "job-prev"), 12)
	h.markRunningByPreviousInstance(t, child, pgid)

	h.start(t)

	waitFor(t, 3*time.Second, func() bool { return !processAlive(pgid) },
		fmt.Sprintf("recovery did not kill the orphan pgid %d", pgid))
	j := h.waitState(t, child, jobs.StateSucceeded, 20*time.Second)
	if !h.hasResult(t, child) || j.Reached == nil || *j.Reached != 12 {
		t.Errorf("recovered train_more should resume and complete: result=%v reached=%v", h.hasResult(t, child), j.Reached)
	}
}

// --- materialize gates: the library facts a claim must satisfy before a spawn ---

func TestMaterializeRefusals(t *testing.T) {
	t.Run("signal mismatch", func(t *testing.T) {
		h := newHarness(t, "train-ok", time.Minute)
		h.pool.profile = func() (Provenance, bool) {
			return Provenance{NamVersion: storetest.NamVersion, DriverSHA256: testDriver,
				SignalSHA256: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}, true
		}
		id := h.seed(t, jobs.KindTrain, 5)
		h.start(t)
		j := h.waitState(t, id, jobs.StateFailed, 15*time.Second)
		if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrSignalMismatch {
			t.Errorf("error_code = %v, want signal_mismatch", j.ErrorCode)
		}
	})
	t.Run("wav invalid", func(t *testing.T) {
		h := newHarness(t, "train-ok", time.Minute)
		id := h.seedWav(t, jobs.KindTrain, 5, testsupport.WAV(44100, 30)) // wrong rate
		h.start(t)
		j := h.waitState(t, id, jobs.StateFailed, 15*time.Second)
		if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrWavInvalid {
			t.Errorf("error_code = %v, want wav_invalid", j.ErrorCode)
		}
		if j.ErrorMessage == nil || !strings.Contains(*j.ErrorMessage, "44100") {
			t.Errorf("error_message = %v, want the validator's reason", j.ErrorMessage)
		}
	})
	t.Run("audio bytes do not match the recorded sha", func(t *testing.T) {
		h := newHarness(t, "train-ok", time.Minute)
		id := h.seed(t, jobs.KindTrain, 5)
		// Same length, different content: the sha the library recorded (and the job
		// pinned) no longer describes the bytes.
		tampered := testsupport.Distinct(validWav, 0x7f)
		storetest.Exec(t, h.store, `UPDATE take_audio SET bytes = $2 WHERE take_id = (SELECT take_id FROM jobs WHERE id = $1)`, id, tampered)
		h.start(t)
		j := h.waitState(t, id, jobs.StateFailed, 15*time.Second)
		if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrMaterialize {
			t.Errorf("error_code = %v, want materialize", j.ErrorCode)
		}
	})
}

// (TestLiveRequestAnsweredOnTheRow is gone with the request it covered. "Let me hear this run as it
// stands" was a conversation — the app set a column, the daemon exported a .nam and answered with a
// row of its own — and it existed because the weights lived on one machine's disk. They are in the
// library now, refreshed every epoch, so a listener reads the take's row and there is nobody to ask.)
