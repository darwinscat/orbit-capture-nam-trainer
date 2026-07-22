// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
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
)

// stubDriverArg is a stable "driver" argv token: the worker's recovery guard and
// the manually-spawned orphan both argv-match on its basename, exactly as the real
// deployment matches on trainer_driver.py.
const stubDriverArg = "trainer_driver.py"

var stubBin string

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
	code := m.Run()
	_ = os.RemoveAll(dir)
	os.Exit(code)
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

func newHarness(t *testing.T, mode string, stall time.Duration) *harness {
	t.Helper()
	base := t.TempDir()
	st, err := store.Open(context.Background(), filepath.Join(base, "trainer.db"))
	if err != nil {
		t.Fatalf("store.Open: %v", err)
	}
	t.Cleanup(func() { st.Close() })
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
		SignalPath:   signal,
		ScratchRoot:  filepath.Join(base, "scratch"),
		Cap:          1,
		StallTimeout: stall,
		OnCounts:     h.recordCounts,
	})
	return h
}

func (h *harness) seed(t *testing.T, key, kind string, epochs int) {
	t.Helper()
	err := h.store.InsertJob(context.Background(), jobs.Job{
		Key: key, Kind: kind, State: jobs.StateQueued,
		Priority: 1, Epochs: epochs, Arch: "standard", CreatedAt: 1,
	}, []byte("capture-bytes"))
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
}

// seedSucceededParent builds a legitimate succeeded train parent DIRECTLY via the
// store (not the pool, so it is mode-independent) with a stored checkpoint whose
// CONTENT is the epoch count as decimal text — exactly what the real/stub driver
// leaves, so resume_ok reads it back to know where to continue numbering. The wav
// must match the child's byte-for-byte (snapshotParent compares wav_sha).
func (h *harness) seedSucceededParent(t *testing.T, key string, epochs int, wav []byte) {
	t.Helper()
	ctx := context.Background()
	if err := h.store.InsertJob(ctx, jobs.Job{
		Key: key, Kind: jobs.KindTrain, State: jobs.StateQueued,
		Priority: 1, Epochs: epochs, Arch: "standard", CreatedAt: 1,
	}, wav); err != nil {
		t.Fatalf("seed parent insert: %v", err)
	}
	j, ok, err := h.store.ClaimNextQueued(ctx, 1, jobs.KindTrain)
	if err != nil || !ok || j.Key != key {
		t.Fatalf("seed parent claim: ok=%v err=%v key=%q", ok, err, j.Key)
	}
	ok, err = h.store.FinishTrainSuccess(ctx, key, 2, []byte("nam-"+key), "{}", nil, []byte(strconv.Itoa(epochs)))
	if err != nil || !ok {
		t.Fatalf("seed parent finish: ok=%v err=%v", ok, err)
	}
}

// seedTrainMore inserts a queued train_more child off baseKey. InsertJob validates
// the parent and snapshots its ckpt into resume_ckpts(child) in the same tx.
func (h *harness) seedTrainMore(t *testing.T, key, baseKey string, epochs int, wav []byte) {
	t.Helper()
	base := baseKey
	if err := h.store.InsertJob(context.Background(), jobs.Job{
		Key: key, Kind: jobs.KindTrainMore, State: jobs.StateQueued,
		Priority: 1, Epochs: epochs, Arch: "standard", CreatedAt: 2,
		BaseKey: &base,
	}, wav); err != nil {
		t.Fatalf("seed train_more: %v", err)
	}
}

// resultCkpt returns the stored results.ckpt for a job (nil/ok=false when none).
func (h *harness) resultCkpt(t *testing.T, key string) ([]byte, bool) {
	t.Helper()
	var ckpt []byte
	err := h.store.DB().QueryRowContext(context.Background(),
		`SELECT ckpt FROM results WHERE job_key = ? AND ckpt IS NOT NULL`, key).Scan(&ckpt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, false
	}
	if err != nil {
		t.Fatalf("resultCkpt: %v", err)
	}
	return ckpt, true
}

// hasResumeCkpt reports whether the child's parent-ckpt snapshot is still present.
func (h *harness) hasResumeCkpt(t *testing.T, key string) bool {
	t.Helper()
	_, ok, err := h.store.ResumeCkpt(context.Background(), key)
	if err != nil {
		t.Fatalf("ResumeCkpt: %v", err)
	}
	return ok
}

func (h *harness) start(t *testing.T) {
	t.Helper()
	if err := h.pool.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(h.pool.Stop)
}

// capturingRunner records the Spec each Spawn saw (and whether a ResumeCkpt file
// was on disk at spawn time — materialize writes it before Spawn), so a test can
// assert the --resume-from arg reached the child with a real file.
type capturingRunner struct {
	inner Runner

	mu           sync.Mutex
	sawResume    bool
	lastResume   string
	resumeExists bool
}

func (c *capturingRunner) Spawn(spec Spec) (*Proc, error) {
	c.mu.Lock()
	if spec.ResumeCkpt != "" {
		c.sawResume = true
		c.lastResume = spec.ResumeCkpt
		_, err := os.Stat(spec.ResumeCkpt)
		c.resumeExists = err == nil
	}
	c.mu.Unlock()
	return c.inner.Spawn(spec)
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

func (h *harness) get(t *testing.T, key string) jobs.Job {
	t.Helper()
	j, ok, err := h.store.GetJob(context.Background(), key)
	if err != nil {
		t.Fatalf("GetJob: %v", err)
	}
	if !ok {
		t.Fatalf("job %s missing", key)
	}
	return j
}

func (h *harness) waitState(t *testing.T, key, want string, timeout time.Duration) jobs.Job {
	t.Helper()
	deadline := time.Now().Add(timeout)
	var last string
	for time.Now().Before(deadline) {
		j, ok, err := h.store.GetJob(context.Background(), key)
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
	t.Fatalf("job %s never reached %q (last %q)", key, want, last)
	return jobs.Job{}
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

// A cap>=2 delete+resubmit of the same content key can have a NEW worker overwrite
// procs[key] before the OLD worker's deferred unregister runs. unregister must be a
// compare-and-delete so the old teardown never drops the new attempt's entry (which
// would orphan a trainer DELETE can no longer reach).
func TestUnregisterIsCompareAndDelete(t *testing.T) {
	h := newHarness(t, "", 0)
	p := h.pool
	a := &procEntry{pgid: 111}
	b := &procEntry{pgid: 222}

	p.register("k", a)
	p.register("k", b) // newer attempt overwrites a

	p.unregister("k", a) // the OLD worker's deferred unregister — must be a no-op
	p.mu.Lock()
	got := p.procs["k"]
	p.mu.Unlock()
	if got != b {
		t.Fatalf("unregister(a) dropped the newer entry; procs[k]=%v, want b", got)
	}

	p.unregister("k", b) // b's own worker removes b
	p.mu.Lock()
	_, present := p.procs["k"]
	p.mu.Unlock()
	if present {
		t.Error("unregister(b) did not remove b")
	}
}

// Notify must publish queue counts even with no worker running, so the keep-awake
// assertion tracks a backlog enqueued (or a job deleted) while the runtime is not
// yet provisioned and nothing claims. The pool is deliberately NOT started here.
func TestNotifyPublishesQueueCounts(t *testing.T) {
	h := newHarness(t, "", 0)
	h.seed(t, "k", jobs.KindTrain, 5)

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
	h.seed(t, "k", jobs.KindTrain, 5)
	h.start(t)

	j := h.waitState(t, "k", jobs.StateSucceeded, 10*time.Second)
	if !j.HasModel {
		t.Error("has_model should be true after a successful train")
	}
	if j.Epoch == nil || *j.Epoch != 4 {
		t.Errorf("epoch = %v, want 4 (last of 5)", j.Epoch)
	}
	if j.SPerEpoch == nil || *j.SPerEpoch <= 0 {
		t.Errorf("s_per_epoch = %v, want > 0", j.SPerEpoch)
	}
	// The final validation ESR is reported for a train job (not just probes).
	if j.ESR == nil || *j.ESR <= 0 {
		t.Errorf("train esr = %v, want the final DRIVER: esr value", j.ESR)
	}
	// Model is stored and non-trivial.
	nam, ok, err := h.store.ModelBytes(context.Background(), "k")
	if err != nil || !ok {
		t.Fatalf("ModelBytes: ok=%v err=%v", ok, err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(nam, &parsed); err != nil || len(parsed) == 0 {
		t.Errorf("model is not valid non-trivial JSON: %v", err)
	}
	// Capture blob dropped at the terminal state.
	if _, ok, _ := h.store.AudioBlob(context.Background(), "k"); ok {
		t.Error("capture blob should be gone after success")
	}
	// The log captured the epoch lines.
	lines, _ := h.store.JobLog(context.Background(), "k")
	if len(lines) == 0 {
		t.Error("job_log should have captured stdout lines")
	}
	// Per-attempt scratch dir removed (scratch root left empty). runJob's teardown
	// is a DEFERRED os.RemoveAll that lands microseconds AFTER the terminal state
	// write waitState just observed — poll for it rather than racing an immediate
	// ReadDir (the measured ~1-2/40 flake this replaces).
	waitFor(t, 3*time.Second, func() bool {
		entries, _ := os.ReadDir(filepath.Join(h.base, "scratch"))
		return len(entries) == 0
	}, "scratch root not emptied after job (deferred RemoveAll)")
}

func TestTrainFailNonzeroExit(t *testing.T) {
	h := newHarness(t, "train-fail", time.Minute)
	h.seed(t, "k", jobs.KindTrain, 5)
	h.start(t)

	j := h.waitState(t, "k", jobs.StateFailed, 10*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != "train_failed" {
		t.Errorf("error_code = %v, want train_failed", j.ErrorCode)
	}
	if j.HasModel {
		t.Error("has_model should be false on failure")
	}
	if _, ok, _ := h.store.AudioBlob(context.Background(), "k"); ok {
		t.Error("blob should be dropped even on failure")
	}
}

func TestStallWatchdog(t *testing.T) {
	h := newHarness(t, "silent-hang", 300*time.Millisecond)
	h.seed(t, "k", jobs.KindTrain, 100)
	h.start(t)

	j := h.waitState(t, "k", jobs.StateFailed, 5*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != "stalled" {
		t.Errorf("error_code = %v, want stalled", j.ErrorCode)
	}
}

func TestDeleteKillsProcessGroup(t *testing.T) {
	h := newHarness(t, "train-hang", time.Minute) // prints Epoch 0 then hangs forever
	h.seed(t, "k", jobs.KindTrain, 100)
	h.start(t)

	// Wait until it is running with a recorded pid.
	waitFor(t, 5*time.Second, func() bool {
		j := h.get(t, "k")
		return j.State == jobs.StateRunning && j.PID != nil
	}, "job never reached running with a pid")
	pgid := int(*h.get(t, "k").PID)
	if !processAlive(pgid) {
		t.Fatalf("trainer pgid %d should be alive", pgid)
	}

	// Mirror the DELETE handler: kill the group, then free the key.
	h.pool.Kill("k")
	if _, err := h.store.DeleteJob(context.Background(), "k"); err != nil {
		t.Fatalf("DeleteJob: %v", err)
	}

	// The process group must die (pgrep -f trainer would show nothing).
	waitFor(t, 3*time.Second, func() bool { return !processAlive(pgid) },
		fmt.Sprintf("process group %d survived the delete", pgid))
	if _, ok, _ := h.store.GetJob(context.Background(), "k"); ok {
		t.Error("job row should be gone after delete")
	}
}

func TestProbeSelfPassKillsOnVerdict(t *testing.T) {
	h := newHarness(t, "probe-self-pass", time.Minute) // prints verdict then hangs forever
	h.seed(t, "k", jobs.KindProbeSelf, jobs.ProbeSelfEpochs)
	h.start(t)

	start := time.Now()
	j := h.waitState(t, "k", jobs.StateSucceeded, 5*time.Second)
	if elapsed := time.Since(start); elapsed > 4*time.Second {
		t.Errorf("verdict took %s — kill-on-verdict did not fire (child hangs forever)", elapsed)
	}
	if j.Verdict == nil || *j.Verdict != jobs.VerdictPass {
		t.Errorf("verdict = %v, want pass", j.Verdict)
	}
	if j.ESR == nil || *j.ESR <= 0 {
		t.Errorf("esr = %v, want the replicate ESR", j.ESR)
	}
	if j.HasModel {
		t.Error("a probe must not store a model")
	}
}

func TestProbeSelfFail(t *testing.T) {
	h := newHarness(t, "probe-self-fail", time.Minute)
	h.seed(t, "k", jobs.KindProbeSelf, jobs.ProbeSelfEpochs)
	h.start(t)

	j := h.waitState(t, "k", jobs.StateSucceeded, 5*time.Second)
	if j.Verdict == nil || *j.Verdict != jobs.VerdictFail {
		t.Errorf("verdict = %v, want fail", j.Verdict)
	}
}

func TestProbeSelfCrashIsNoVerdict(t *testing.T) {
	h := newHarness(t, "probe-self-crash", time.Minute) // exits with no verdict
	h.seed(t, "k", jobs.KindProbeSelf, jobs.ProbeSelfEpochs)
	h.start(t)

	j := h.waitState(t, "k", jobs.StateFailed, 5*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != "no_verdict" {
		t.Errorf("error_code = %v, want no_verdict (a crash is not a fail verdict)", j.ErrorCode)
	}
	if j.Verdict != nil {
		t.Errorf("verdict = %v, want nil for a crashed probe", j.Verdict)
	}
}

func TestProbeE10OK(t *testing.T) {
	h := newHarness(t, "probe-e10-ok", time.Minute)
	h.seed(t, "k", jobs.KindProbeE10, jobs.ProbeE10Epochs)
	h.start(t)

	j := h.waitState(t, "k", jobs.StateSucceeded, 10*time.Second)
	if j.ESR == nil || *j.ESR <= 0 {
		t.Errorf("esr = %v, want the E@10 value", j.ESR)
	}
	if j.HasModel {
		t.Error("probe_e10 must not store a model even though the driver exports one")
	}
}

func TestProbeE10NA(t *testing.T) {
	h := newHarness(t, "probe-e10-na", time.Minute)
	h.seed(t, "k", jobs.KindProbeE10, jobs.ProbeE10Epochs)
	h.start(t)

	j := h.waitState(t, "k", jobs.StateFailed, 10*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != "no_esr" {
		t.Errorf("error_code = %v, want no_esr", j.ErrorCode)
	}
}

func TestShutdownRequeuesNotFails(t *testing.T) {
	h := newHarness(t, "train-hang", time.Minute)
	h.seed(t, "k", jobs.KindTrain, 100)
	h.start(t)

	waitFor(t, 5*time.Second, func() bool {
		j := h.get(t, "k")
		return j.State == jobs.StateRunning && j.PID != nil
	}, "job never reached running")
	pgid := int(*h.get(t, "k").PID)

	h.pool.Stop() // graceful shutdown must requeue, never fail

	j := h.get(t, "k")
	if j.State != jobs.StateQueued {
		t.Errorf("state = %q after shutdown, want queued (never failed)", j.State)
	}
	if j.PID != nil {
		t.Errorf("pid should be cleared on requeue, got %v", j.PID)
	}
	waitFor(t, 3*time.Second, func() bool { return !processAlive(pgid) },
		"child survived shutdown")
}

// TestProbeRunsConcurrentlyWithLongTrain is the scheduler guarantee the app needs:
// a self-ESR verdict must return in seconds even while a long train occupies the
// single training cap, because the probe lane drains independently.
func TestProbeRunsConcurrentlyWithLongTrain(t *testing.T) {
	h := newHarness(t, "auto", time.Minute) // stub picks behaviour by epoch count
	// A long "train" that hangs, occupying the train lane (cap 1).
	h.seed(t, "longtrain", jobs.KindTrain, 400)
	h.start(t)
	waitFor(t, 5*time.Second, func() bool {
		return h.get(t, "longtrain").State == jobs.StateRunning
	}, "long train never started")

	// Now a self-ESR probe arrives; it must reach a verdict fast, NOT wait for the
	// train (which is still running).
	h.seed(t, "probe", jobs.KindProbeSelf, jobs.ProbeSelfEpochs)
	h.pool.Notify()

	start := time.Now()
	j := h.waitState(t, "probe", jobs.StateSucceeded, 5*time.Second)
	if elapsed := time.Since(start); elapsed > 4*time.Second {
		t.Errorf("probe verdict took %s while a train ran — lanes are not independent", elapsed)
	}
	if j.Verdict == nil || *j.Verdict != jobs.VerdictPass {
		t.Errorf("probe verdict = %v, want pass", j.Verdict)
	}
	// The train is still going (was not displaced by the probe).
	if st := h.get(t, "longtrain").State; st != jobs.StateRunning {
		t.Errorf("long train state = %q, want still running alongside the probe", st)
	}
}

func TestRestartRecoveryKillsOrphanAndRequeues(t *testing.T) {
	h := newHarness(t, "train-ok", time.Minute) // the re-run mode after recovery
	scratchKey := filepath.Join(h.base, "scratch", "k")
	if err := os.MkdirAll(filepath.Join(scratchKey, "out"), 0o755); err != nil {
		t.Fatal(err)
	}

	// Spawn an orphan "previous-run" child in its own group, silent-hang so it
	// never exits on its own — recovery must kill it.
	orphan := exec.Command(stubBin, "-u", stubDriverArg,
		"--input", "sig",
		"--output", filepath.Join(scratchKey, "capture.wav"),
		"--outdir", filepath.Join(scratchKey, "out"),
		"--name", "model", "--epochs", "100", "--arch", "standard")
	orphan.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	orphan.Env = append(os.Environ(), "ONCT_STUB_MODE=silent-hang")
	if err := orphan.Start(); err != nil {
		t.Fatalf("spawn orphan: %v", err)
	}
	pgid := orphan.Process.Pid
	go orphan.Wait() // reap it once recovery kills it, so processAlive flips to false

	// A running row from the "previous process", pointing at the orphan pgid.
	err := h.store.InsertJob(context.Background(), jobs.Job{
		Key: "k", Kind: jobs.KindTrain, State: jobs.StateQueued,
		Priority: 1, Epochs: 5, Arch: "standard", CreatedAt: 1,
	}, []byte("capture-bytes"))
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	if _, err := h.store.DB().ExecContext(context.Background(),
		"UPDATE jobs SET state='running', pid=? WHERE key='k'", pgid); err != nil {
		t.Fatalf("mark running: %v", err)
	}

	waitFor(t, 3*time.Second, func() bool { return processAlive(pgid) }, "orphan should be alive before recovery")

	h.start(t) // Start → recovery kills the orphan, requeues, workers re-run to success

	waitFor(t, 3*time.Second, func() bool { return !processAlive(pgid) },
		fmt.Sprintf("recovery did not kill the orphan pgid %d", pgid))

	j := h.waitState(t, "k", jobs.StateSucceeded, 10*time.Second)
	if !j.HasModel {
		t.Error("requeued job should train to success with a model")
	}
}

// Pause(killRunning=true) must kill the child, REQUEUE the job (the shutdown
// rule — never a failure), and hold it queued until Resume claims it again.
func TestPauseNowKillsRequeuesAndResumes(t *testing.T) {
	h := newHarness(t, "train-hang", time.Minute)
	h.seed(t, "k", jobs.KindTrain, 100)
	h.start(t)

	waitFor(t, 5*time.Second, func() bool {
		j := h.get(t, "k")
		return j.State == jobs.StateRunning && j.PID != nil
	}, "job never reached running")
	pgid := int(*h.get(t, "k").PID)

	h.pool.Pause(true)
	if !h.pool.Paused() {
		t.Fatal("Paused() = false right after Pause")
	}
	h.waitState(t, "k", jobs.StateQueued, 5*time.Second)
	waitFor(t, 3*time.Second, func() bool { return !processAlive(pgid) },
		"child survived pause")

	// The gate holds: nothing reclaims the job while paused.
	time.Sleep(600 * time.Millisecond)
	if st := h.get(t, "k").State; st != jobs.StateQueued {
		t.Fatalf("state = %q while paused, want queued", st)
	}

	h.pool.Resume()
	if h.pool.Paused() {
		t.Fatal("Paused() = true after Resume")
	}
	waitFor(t, 5*time.Second, func() bool {
		return h.get(t, "k").State == jobs.StateRunning
	}, "job never reclaimed after Resume")
}

// Pause(killRunning=false) lets the running job finish normally and only stops
// NEW claims: the second queued job must sit still until Resume.
func TestPauseAfterCurrentFinishesRunning(t *testing.T) {
	h := newHarness(t, "train-ok", time.Minute)
	h.seed(t, "first", jobs.KindTrain, 5)
	h.start(t)
	waitFor(t, 5*time.Second, func() bool {
		return h.get(t, "first").State == jobs.StateRunning
	}, "first job never started")

	h.pool.Pause(false)
	h.seed(t, "second", jobs.KindTrain, 5)
	h.pool.Notify()

	// The running job completes with its real result — not killed, not requeued.
	h.waitState(t, "first", jobs.StateSucceeded, 10*time.Second)

	time.Sleep(600 * time.Millisecond)
	if st := h.get(t, "second").State; st != jobs.StateQueued {
		t.Fatalf("second state = %q while paused, want queued", st)
	}

	h.pool.Resume()
	h.waitState(t, "second", jobs.StateSucceeded, 10*time.Second)
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
// post-register pauseKill check has to kill the child it just spawned. Without
// that check the job runs to completion under a "paused" icon.
func TestPauseNowCatchesClaimRegisterWindow(t *testing.T) {
	h := newHarness(t, "train-hang", time.Minute)
	g := &gatingRunner{
		inner:   h.pool.runner,
		entered: make(chan struct{}),
		release: make(chan struct{}),
	}
	h.pool.runner = g
	h.seed(t, "k", jobs.KindTrain, 100)
	h.start(t)

	<-g.entered // claimed (running in DB), spawn blocked, NOT registered
	h.pool.Pause(true)
	close(g.release)

	// The self-kill lands after register; the job must requeue, not keep running.
	h.waitState(t, "k", jobs.StateQueued, 5*time.Second)
	waitFor(t, 3*time.Second, func() bool {
		j := h.get(t, "k")
		return j.State == jobs.StateQueued && j.PID == nil
	}, "pid not cleared after window-escape requeue")
}

// SetCap must resize the training lane LIVE. Raising it wakes an idle spawned
// worker (a second hang-train claims immediately); Cap() reports the change.
func TestSetCapGrowsLive(t *testing.T) {
	h := newHarness(t, "train-hang", time.Minute)
	h.pool.lanes[0].cap = 2 // spawn width (CapLimit); live cap stays 1
	h.seed(t, "a", jobs.KindTrain, 100)
	h.seed(t, "b", jobs.KindTrain, 100)
	h.start(t)

	h.waitState(t, "a", jobs.StateRunning, 5*time.Second)
	time.Sleep(600 * time.Millisecond)
	if st := h.get(t, "b").State; st != jobs.StateQueued {
		t.Fatalf("b state = %q at cap 1, want queued", st)
	}
	if h.pool.Cap() != 1 {
		t.Fatalf("Cap() = %d, want 1", h.pool.Cap())
	}

	h.pool.SetCap(2)
	if h.pool.Cap() != 2 {
		t.Fatalf("Cap() = %d after SetCap(2), want 2", h.pool.Cap())
	}
	h.waitState(t, "b", jobs.StateRunning, 5*time.Second)
	if st := h.get(t, "a").State; st != jobs.StateRunning {
		t.Errorf("a state = %q, want still running", st)
	}
}

// Lowering the cap kills nothing: running jobs finish their course, and only
// then does the lane narrow — the freed slot above the new cap claims no more.
func TestSetCapShrinksAsJobsFinish(t *testing.T) {
	h := newHarness(t, "auto", time.Minute) // epochs: 5 → train-ok, else train-hang
	h.pool.lanes[0].cap = 2
	h.pool.trainCap.Store(2)
	h.seed(t, "a", jobs.KindTrain, 5) // completes on its own
	h.seed(t, "b", jobs.KindTrain, 5) // completes on its own
	h.start(t)

	waitFor(t, 5*time.Second, func() bool {
		return h.get(t, "a").State == jobs.StateRunning && h.get(t, "b").State == jobs.StateRunning
	}, "both jobs never ran at cap 2")

	h.pool.SetCap(1)
	// Nothing is killed: both finish with their real result.
	h.waitState(t, "a", jobs.StateSucceeded, 10*time.Second)
	h.waitState(t, "b", jobs.StateSucceeded, 10*time.Second)

	// At cap 1 the narrowed lane runs strictly one at a time: c claims, d waits.
	h.seed(t, "c", jobs.KindTrain, 400) // hangs, occupying the single slot
	h.seed(t, "d", jobs.KindTrain, 400)
	h.pool.Notify()
	h.waitState(t, "c", jobs.StateRunning, 5*time.Second)
	time.Sleep(600 * time.Millisecond)
	if st := h.get(t, "d").State; st != jobs.StateQueued {
		t.Fatalf("d state = %q at cap 1, want queued", st)
	}
}

// SetCap clamps to 1..the spawned width.
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

// A train_more resumes from its parent's checkpoint: the worker materializes the
// snapshot to <scratch>/resume.ckpt, passes --resume-from, and the child numbers
// epochs absolutely from start_epoch, exporting a NEW nam + a NEW ckpt (chain-ready).
func TestTrainMoreResumesFromParentCkpt(t *testing.T) {
	h := newHarness(t, "resume_ok", time.Minute)
	cr := &capturingRunner{inner: h.pool.runner}
	h.pool.runner = cr

	wav := []byte("capture-bytes")
	h.seedSucceededParent(t, "parent", 5, wav)
	h.seedTrainMore(t, "child", "parent", 12, wav)

	if !h.hasResumeCkpt(t, "child") {
		t.Fatal("resume snapshot should exist for a queued train_more")
	}
	h.start(t)

	j := h.waitState(t, "child", jobs.StateSucceeded, 15*time.Second)

	// --resume-from reached the stub, and the scratch resume.ckpt was on disk when
	// the child spawned.
	resume, exists, saw := cr.resume()
	if !saw {
		t.Fatal("Spawn never received a --resume-from ckpt for the train_more")
	}
	if filepath.Base(resume) != "resume.ckpt" || !exists {
		t.Errorf("resume ckpt = %q (exists=%v), want a present <scratch>/resume.ckpt", resume, exists)
	}
	// Numbering resumes at start_epoch (5) and runs in ABSOLUTE epochs to 11.
	if j.StartEpoch == nil || *j.StartEpoch != 5 {
		t.Errorf("start_epoch = %v, want 5", j.StartEpoch)
	}
	if j.Epoch == nil || *j.Epoch != 11 {
		t.Errorf("epoch = %v, want 11 (last of 12, absolute)", j.Epoch)
	}
	lines, _ := h.store.JobLog(context.Background(), "child")
	if first := firstEpochLine(lines); first != 5 {
		t.Errorf("first Epoch line = %d, want 5 (resumed numbering)", first)
	}
	if !logContains(lines, "DRIVER: resuming from epoch 5") {
		t.Errorf("job_log missing the resuming banner; got %v", lines)
	}
	// A NEW nam AND a NEW ckpt (content = the new total) are stored; the run-input
	// snapshot is dropped at the terminal state.
	if !j.HasModel {
		t.Error("train_more should store a model")
	}
	if ckpt, ok := h.resultCkpt(t, "child"); !ok || string(ckpt) != "12" {
		t.Errorf("stored ckpt = %q (ok=%v), want \"12\" (the new total)", ckpt, ok)
	}
	if h.hasResumeCkpt(t, "child") {
		t.Error("resume snapshot should be dropped at the terminal state")
	}
}

// A train_more whose ckpt restore blows up BEFORE any Epoch line failed to prove
// the resume → resume_failed; its run-input (blob + snapshot) is dropped, the log
// (the traceback) is kept as history.
func TestTrainMoreBadCkptIsResumeFailed(t *testing.T) {
	h := newHarness(t, "resume_badckpt", time.Minute)
	wav := []byte("capture-bytes")
	h.seedSucceededParent(t, "parent", 5, wav)
	h.seedTrainMore(t, "child", "parent", 12, wav)
	h.start(t)

	j := h.waitState(t, "child", jobs.StateFailed, 10*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != "resume_failed" {
		t.Errorf("error_code = %v, want resume_failed (died before any Epoch line)", j.ErrorCode)
	}
	if _, ok, _ := h.store.AudioBlob(context.Background(), "child"); ok {
		t.Error("capture blob should be dropped on a failed train_more")
	}
	if h.hasResumeCkpt(t, "child") {
		t.Error("resume snapshot should be dropped on a failed train_more")
	}
	if lines, _ := h.store.JobLog(context.Background(), "child"); len(lines) == 0 {
		t.Error("job_log should be kept on failure")
	}
}

// A train_more that crashes AFTER the resume demonstrably took (Epoch lines were
// seen) is a plain train_failed — NOT resume_failed (crew F6).
func TestTrainMoreLateCrashIsTrainFailed(t *testing.T) {
	h := newHarness(t, "train-fail", time.Minute) // prints Epoch lines, then exits nonzero
	wav := []byte("capture-bytes")
	h.seedSucceededParent(t, "parent", 5, wav)
	h.seedTrainMore(t, "child", "parent", 12, wav)
	h.start(t)

	j := h.waitState(t, "child", jobs.StateFailed, 10*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != "train_failed" {
		t.Errorf("error_code = %v, want train_failed (crashed after resuming)", j.ErrorCode)
	}
}

// The full chain THROUGH THE POOL: a pool-run plain train stores its ckpt (not
// only store-seeded parents — a kind-conditional regression in the ckpt store
// would slip past every other test), and a train_more chained off that pool-made
// ckpt resumes and stores its own. Mode "auto" runs the parent as train-ok
// (epochs=5) and the child as resume_ok (--resume-from present).
func TestChainTrainThenTrainMoreThroughPool(t *testing.T) {
	h := newHarness(t, "auto", time.Minute)
	h.seed(t, "parent", jobs.KindTrain, 5)
	h.start(t)

	h.waitState(t, "parent", jobs.StateSucceeded, 15*time.Second)
	if ckpt, ok := h.resultCkpt(t, "parent"); !ok || string(ckpt) != "5" {
		t.Fatalf("pool-run train stored ckpt %q (ok=%v), want \"5\"", ckpt, ok)
	}

	h.seedTrainMore(t, "child", "parent", 12, []byte("capture-bytes")) // seed()'s wav
	h.pool.Notify()
	j := h.waitState(t, "child", jobs.StateSucceeded, 15*time.Second)
	if j.Epoch == nil || *j.Epoch != 11 {
		t.Errorf("child epoch = %v, want 11 (absolute numbering)", j.Epoch)
	}
	if ckpt, ok := h.resultCkpt(t, "child"); !ok || string(ckpt) != "12" {
		t.Errorf("child ckpt = %q (ok=%v), want \"12\" (chain-ready)", ckpt, ok)
	}
}

// A train_more killed by the stall watchdog BEFORE any Epoch line is `stalled`,
// never resume_failed — the reason-first rule outranks the failure-code
// selection. Pins the ordering in classify: hoisting the resume_failed choice
// above the stall check would break exactly this.
func TestTrainMoreStallBeatsResumeFailed(t *testing.T) {
	h := newHarness(t, "silent-hang", 300*time.Millisecond)
	wav := []byte("capture-bytes")
	h.seedSucceededParent(t, "parent", 5, wav)
	h.seedTrainMore(t, "child", "parent", 12, wav)
	h.start(t)

	j := h.waitState(t, "child", jobs.StateFailed, 10*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != "stalled" {
		t.Errorf("error_code = %v, want stalled (stall reason outranks resume_failed)", j.ErrorCode)
	}
}

// A probe_e10 killed AFTER banking its ESR but BEFORE exporting model.ckpt must
// succeed with the ESR and store NO ckpt — a torn/absent ckpt must never seed a
// train_more (crew F1 regression).
func TestProbeE10KillAfterESRStoresNoCkpt(t *testing.T) {
	// 750ms (not 300): the stub prints 10 epoch lines BEFORE the ESR — on a starved
	// runner a 300ms watchdog could fire mid-run and flip the outcome to no_esr
	// instead of merely delaying the kill.
	h := newHarness(t, "probe_kill_after_esr", 750*time.Millisecond)
	h.seed(t, "k", jobs.KindProbeE10, jobs.ProbeE10Epochs)
	h.start(t)

	j := h.waitState(t, "k", jobs.StateSucceeded, 10*time.Second)
	if j.ESR == nil || *j.ESR <= 0 {
		t.Errorf("esr = %v, want the E@10 value banked before the kill", j.ESR)
	}
	if j.HasModel {
		t.Error("probe_e10 must not store a model")
	}
	if ckpt, ok := h.resultCkpt(t, "k"); ok {
		t.Errorf("stored ckpt = %q, want none (killed before export)", ckpt)
	}
}

// A probe_e10 that runs to natural exit now stores its ckpt (nam=NULL, ckpt="10")
// so its 10 epochs can seed a train_more — the app's standard probe→train flow.
func TestProbeE10StoresCkpt(t *testing.T) {
	h := newHarness(t, "probe-e10-ok", time.Minute)
	h.seed(t, "k", jobs.KindProbeE10, jobs.ProbeE10Epochs)
	h.start(t)

	j := h.waitState(t, "k", jobs.StateSucceeded, 10*time.Second)
	if j.HasModel {
		t.Error("probe_e10 must not store a model (nam stays NULL)")
	}
	if ckpt, ok := h.resultCkpt(t, "k"); !ok || string(ckpt) != "10" {
		t.Errorf("stored ckpt = %q (ok=%v), want \"10\"", ckpt, ok)
	}
}

// A kill -9 of a RUNNING train_more (recovery on the next start) requeues it with
// its snapshot intact — finish never ran — so the re-claim resumes and completes.
// A resume_ok success is airtight proof the snapshot survived: had it been dropped,
// materialize would have failed the child "resume checkpoint missing".
func TestTrainMoreRecoveryKeepsSnapshot(t *testing.T) {
	h := newHarness(t, "resume_ok", time.Minute) // the re-run mode after recovery
	wav := []byte("capture-bytes")
	h.seedSucceededParent(t, "parent", 5, wav)
	h.seedTrainMore(t, "child", "parent", 12, wav)

	scratchKey := filepath.Join(h.base, "scratch", "child")
	if err := os.MkdirAll(filepath.Join(scratchKey, "out"), 0o755); err != nil {
		t.Fatal(err)
	}
	// A "previous-run" child that never exits — recovery must kill it.
	orphan := exec.Command(stubBin, "-u", stubDriverArg,
		"--input", "sig",
		"--output", filepath.Join(scratchKey, "capture.wav"),
		"--outdir", filepath.Join(scratchKey, "out"),
		"--name", "model", "--epochs", "12", "--arch", "standard")
	orphan.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	orphan.Env = append(os.Environ(), "ONCT_STUB_MODE=silent-hang")
	if err := orphan.Start(); err != nil {
		t.Fatalf("spawn orphan: %v", err)
	}
	pgid := orphan.Process.Pid
	go orphan.Wait()

	if _, err := h.store.DB().ExecContext(context.Background(),
		"UPDATE jobs SET state='running', pid=? WHERE key='child'", pgid); err != nil {
		t.Fatalf("mark running: %v", err)
	}
	waitFor(t, 3*time.Second, func() bool { return processAlive(pgid) }, "orphan should be alive before recovery")
	if !h.hasResumeCkpt(t, "child") {
		t.Fatal("resume snapshot should exist before recovery")
	}

	h.start(t) // recovery kills the orphan and requeues the child (finish never runs)

	waitFor(t, 3*time.Second, func() bool { return !processAlive(pgid) },
		fmt.Sprintf("recovery did not kill the orphan pgid %d", pgid))

	j := h.waitState(t, "child", jobs.StateSucceeded, 15*time.Second)
	if !j.HasModel {
		t.Error("recovered train_more should resume and complete with a model")
	}
	if ckpt, ok := h.resultCkpt(t, "child"); !ok || string(ckpt) != "12" {
		t.Errorf("stored ckpt = %q (ok=%v), want \"12\"", ckpt, ok)
	}
	if h.hasResumeCkpt(t, "child") {
		t.Error("snapshot should be dropped once the resumed run finished")
	}
}

// A mid-run DELETE of a train_more kills the process group, removes the row, and
// CASCADE-drops the resume_ckpts snapshot with it.
func TestTrainMoreDeleteCascadesSnapshot(t *testing.T) {
	h := newHarness(t, "train-hang", time.Minute) // prints an Epoch line, then hangs
	wav := []byte("capture-bytes")
	h.seedSucceededParent(t, "parent", 5, wav)
	h.seedTrainMore(t, "child", "parent", 12, wav)
	h.start(t)

	waitFor(t, 5*time.Second, func() bool {
		j := h.get(t, "child")
		return j.State == jobs.StateRunning && j.PID != nil
	}, "child never reached running with a pid")
	pgid := int(*h.get(t, "child").PID)
	if !h.hasResumeCkpt(t, "child") {
		t.Fatal("resume snapshot should exist while the train_more runs")
	}

	// Mirror the DELETE handler: kill the group, then free the key.
	h.pool.Kill("child")
	if _, err := h.store.DeleteJob(context.Background(), "child"); err != nil {
		t.Fatalf("DeleteJob: %v", err)
	}

	waitFor(t, 3*time.Second, func() bool { return !processAlive(pgid) },
		fmt.Sprintf("process group %d survived the delete", pgid))
	if _, ok, _ := h.store.GetJob(context.Background(), "child"); ok {
		t.Error("child row should be gone after delete")
	}
	if h.hasResumeCkpt(t, "child") {
		t.Error("resume snapshot should be CASCADE-gone after the child is deleted")
	}
}
