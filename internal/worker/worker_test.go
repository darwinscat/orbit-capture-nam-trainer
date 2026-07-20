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

func (h *harness) start(t *testing.T) {
	t.Helper()
	if err := h.pool.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(h.pool.Stop)
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
	// Per-attempt scratch dir removed (scratch root left empty).
	if entries, _ := os.ReadDir(filepath.Join(h.base, "scratch")); len(entries) != 0 {
		t.Errorf("scratch root not empty after job: %d leftover entries", len(entries))
	}
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
