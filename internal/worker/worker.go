// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package worker owns the training subprocess lifecycle: a pool of `cap` workers
// drains the queued jobs, each spawning ONE trainer child in its own process
// group, streaming its output into progress + job_log, and enforcing kill,
// stall-watchdog, and restart-recovery semantics (the design notes).
//
// The kill/Wait discipline is the load-bearing part:
// every SIGKILL happens-before cmd.Wait(), so a pgid can never be recycled and
// re-killed (an unreaped zombie leader reserves its pgid). External kills (DELETE,
// stall, shutdown) go through procEntry.kill and are gated by a `reaping` flag the
// worker sets just before it reaps; the worker also SIGKILLs the whole group
// unconditionally after EOF, so a child that closes stdout while still alive is
// never left behind. The terminal outcome is chosen by the kill REASON first
// (delete → nothing, shutdown → requeue, stall → failed) and only then by exit
// code — so a clean daemon restart never writes an in-flight job `failed`.
package worker

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"orbit-capture-nam-trainer/internal/applog"
	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/store"
)

// DefaultStallTimeout is the no-output kill threshold. Torch import is silent for
// minutes, so this slack is deliberate — do not lower it (the design notes).
const DefaultStallTimeout = 15 * time.Minute

// kill reasons
const (
	reasonDelete   = "delete"
	reasonStall    = "stall"
	reasonVerdict  = "verdict"
	reasonShutdown = "shutdown"
	reasonPause    = "pause"
)

// Options configures a Pool.
type Options struct {
	Store          *store.Store
	Log            *applog.Logger
	Runner         Runner
	SignalPath     string                    // the --input capture signal (materialized embedded wav)
	ScratchRoot    string                    // parent of per-job scratch dirs
	Cap            int                       // initial train-lane width (the GPU-bound lane); live-adjustable via SetCap
	CapLimit       int                       // train workers actually SPAWNED (SetCap's ceiling); 0 → Cap
	ProbeSelfCap   int                       // probe_self-lane workers (0 → 1)
	ProbeE10Cap    int                       // probe_e10-lane workers (0 → 1)
	StallTimeout   time.Duration             // 0 → DefaultStallTimeout
	OnCounts       func(running, queued int) // publish live counts to /v1/health; may be nil
	OnAvgSPerEpoch func(*float64)            // publish the moving-average s/epoch; may be nil
	Now            func() time.Time          // DB timestamps; 0 → time.Now
	Ready          func() bool               // workers idle until this is true (runtime provisioned); nil → always ready
}

// laneSpec is one scheduling lane: a set of job kinds drained by its own workers.
// Probe lanes run alongside the train lane so a self-ESR verdict is seconds away
// even while a long train occupies the training cap (they time-slice the one GPU).
type laneSpec struct {
	name  string
	kinds []string
	cap   int
}

// Pool is the worker pool. It implements httpapi.Killer.
type Pool struct {
	store       *store.Store
	log         *applog.Logger
	runner      Runner
	signalPath  string
	scratchRoot string
	lanes       []laneSpec
	stall       time.Duration
	onCounts    func(running, queued int)
	onAvg       func(*float64)
	now         func() time.Time
	ready       func() bool

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// paused gates claiming only (the menu-bar control): queued jobs stay
	// queued, running ones are untouched unless Pause killed them explicitly.
	// In-memory by design — a daemon restart resumes. pauseKill covers the
	// claim→register window a kill-Pause's procs snapshot cannot see: a worker
	// that claimed before the gate closed re-checks it right after registering
	// and self-kills (the same closure SetJobPID provides for DELETE).
	paused    atomic.Bool
	pauseKill atomic.Bool

	// trainCap is the LIVE train-lane width (1..the spawned worker count):
	// CapLimit workers are always spawned, and train worker i claims only while
	// i < trainCap. Raising it wakes idle workers instantly; lowering it stops
	// further claims — running jobs finish, nothing is killed.
	trainCap atomic.Int32

	mu    sync.Mutex
	procs map[string]*procEntry
	wake  chan struct{}

	// countsMu serializes the read-then-publish in publishCounts so a stale
	// snapshot from one lane worker can't overwrite a fresher one from another
	// (with multiple lanes the publishes are no longer totally ordered).
	countsMu sync.Mutex
}

// procEntry tracks one running child for external kills. See the kill/Wait
// discipline in the package doc.
type procEntry struct {
	mu      sync.Mutex
	pgid    int
	reason  string
	reaping bool
}

// kill SIGKILLs the group unless the worker has already begun reaping. The first
// caller's reason sticks. Safe pre-Wait: the pgid is reserved by the live/zombie
// leader, so the signal never reaches a recycled process.
func (e *procEntry) kill(why string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.reaping {
		return
	}
	if e.reason == "" {
		e.reason = why
	}
	killGroup(e.pgid)
}

// beginReap closes the door on external kills, sweeps the group one last time
// (covering a child that closed stdout but is still alive), and returns the
// decided reason. It MUST be called immediately before Proc.Wait.
func (e *procEntry) beginReap() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.reaping = true
	killGroup(e.pgid)
	return e.reason
}

// New builds a Pool.
func New(o Options) *Pool {
	stall := o.StallTimeout
	if stall <= 0 {
		stall = DefaultStallTimeout
	}
	now := o.Now
	if now == nil {
		now = time.Now
	}
	atLeast1 := func(n int) int {
		if n < 1 {
			return 1
		}
		return n
	}
	ready := o.Ready
	if ready == nil {
		ready = func() bool { return true }
	}
	capLimit := o.CapLimit
	if capLimit < atLeast1(o.Cap) {
		capLimit = atLeast1(o.Cap)
	}
	p := &Pool{
		store:       o.Store,
		log:         o.Log,
		runner:      o.Runner,
		signalPath:  o.SignalPath,
		scratchRoot: o.ScratchRoot,
		lanes: []laneSpec{
			// The train lane is the GPU-bound one: CapLimit workers are spawned and
			// the LIVE width (trainCap) gates which of them claim. The probe lanes
			// are separate so a self-ESR verdict (seconds, kill-on-verdict) and an
			// E@10 probe (~10 epochs) run immediately alongside a long train instead
			// of queueing behind it — the whole point of a rig-side self-check.
			{name: "train", kinds: jobs.LaneKinds(jobs.KindTrain), cap: capLimit},
			{name: "probe_self", kinds: []string{jobs.KindProbeSelf}, cap: atLeast1(o.ProbeSelfCap)},
			{name: "probe_e10", kinds: []string{jobs.KindProbeE10}, cap: atLeast1(o.ProbeE10Cap)},
		},
		stall:    stall,
		onCounts: o.OnCounts,
		onAvg:    o.OnAvgSPerEpoch,
		now:      now,
		ready:    ready,
		procs:    make(map[string]*procEntry),
		wake:     make(chan struct{}, 1),
	}
	p.trainCap.Store(int32(atLeast1(o.Cap)))
	return p
}

// SetCap resizes the training lane LIVE (the PATCH /v1/cap and menu control):
// raising it wakes idle workers now; lowering it stops further claims and
// takes effect as running jobs finish — nothing is killed. A worker already
// past its gate check may still claim one job right after a lower (a µs-wide
// TOCTOU) — identical outcome to lowering a moment later, accepted over
// claim-then-requeue churn. Clamped to 1..the spawned worker count (CapLimit).
func (p *Pool) SetCap(n int) {
	if n < 1 {
		n = 1
	}
	if limit := p.lanes[0].cap; n > limit {
		n = limit
	}
	if old := p.trainCap.Swap(int32(n)); old != int32(n) {
		p.log.Printf("cap %d → %d (live)", old, n)
		p.Notify()
	}
}

// Cap reports the live training-lane width (what /v1/health and the menu show).
func (p *Pool) Cap() int { return int(p.trainCap.Load()) }

// Kill aborts a running job's process group (the httpapi.Killer for DELETE).
func (p *Pool) Kill(key string) {
	p.mu.Lock()
	e := p.procs[key]
	p.mu.Unlock()
	if e != nil {
		e.kill(reasonDelete)
	}
}

// Pause stops the pool claiming new jobs (the menu-bar control). With
// killRunning, every running child is also killed and its job REQUEUED (the
// shutdown rule, never a failure) — Resume claims it again from scratch;
// mid-run progress is lost by design (resume-from-checkpoint is out of scope).
// Without killRunning, running jobs finish normally ("pause after current").
func (p *Pool) Pause(killRunning bool) {
	p.paused.Store(true)
	// Republish so OnCounts consumers (keep-awake) see the new gate now — a
	// pure gate change has no claim/finish edge to piggyback on.
	defer p.publishStats()
	if !killRunning {
		p.log.Printf("pause: claiming stopped, running jobs will finish")
		return
	}
	p.log.Printf("pause: claiming stopped, killing running jobs (they requeue)")
	// The flag must be up BEFORE the snapshot: a worker inside the
	// claim→register window is invisible to the snapshot but sees the flag at
	// its post-register check — one of the two always catches the child.
	p.pauseKill.Store(true)
	p.mu.Lock()
	entries := make([]*procEntry, 0, len(p.procs))
	for _, e := range p.procs {
		entries = append(entries, e)
	}
	p.mu.Unlock()
	for _, e := range entries {
		e.kill(reasonPause)
	}
}

// Resume lifts a Pause and wakes the workers.
func (p *Pool) Resume() {
	p.pauseKill.Store(false)
	if p.paused.CompareAndSwap(true, false) {
		p.log.Printf("resume: claiming jobs again")
	}
	p.Notify()
}

// Paused reports the pause gate (the menu reflects it).
func (p *Pool) Paused() bool { return p.paused.Load() }

// Notify nudges an idle worker to check the queue now and republishes the queue
// counts (called after a PUT or a DELETE). Republishing here — not only on the
// claim/finish edges a worker drives — is what keeps the /v1/health counts and
// the keep-awake assertion correct for work that is enqueued or removed while no
// worker claims it: e.g. a backlog PUT during first-run provisioning (ready is
// still false), or a DELETE of a still-queued job.
func (p *Pool) Notify() {
	p.publishStats()
	select {
	case p.wake <- struct{}{}:
	default:
	}
}

// Start runs restart-recovery, then launches the worker goroutines. Recovery
// fully completes (killing old children, sweeping scratch) BEFORE any worker can
// claim a job, so nothing races the sweep.
func (p *Pool) Start(ctx context.Context) error {
	p.ctx, p.cancel = context.WithCancel(ctx)
	if err := p.recover(p.ctx); err != nil {
		return err
	}
	p.publishStats()
	total := 0
	for _, ln := range p.lanes {
		kinds := ln.kinds
		trainLane := ln.name == "train"
		for i := 0; i < ln.cap; i++ {
			idx := i
			gate := func() bool { return true }
			if trainLane {
				// Train worker i claims only while i < the live cap — the whole
				// live-resize mechanism; the extra workers just idle.
				gate = func() bool { return idx < int(p.trainCap.Load()) }
			}
			p.wg.Add(1)
			go p.workerLoop(kinds, gate)
		}
		total += ln.cap
	}
	p.log.Printf("worker pool started (%d workers: train=%d of max %d, probe_self=%d probe_e10=%d, stall %s)",
		total, p.trainCap.Load(), p.lanes[0].cap, p.lanes[1].cap, p.lanes[2].cap, p.stall)
	return nil
}

// Stop cancels the pool: each worker SIGKILLs its child (reason shutdown → the
// job is requeued, never failed) and joins.
func (p *Pool) Stop() {
	if p.cancel != nil {
		p.cancel()
	}
	p.wg.Wait()
}

func (p *Pool) workerLoop(kinds []string, gate func() bool) {
	defer p.wg.Done()
	for {
		if p.ctx.Err() != nil {
			return
		}
		// Idle until the runtime is provisioned (jobs are accepted and queue
		// while ready is false — nothing spawns until python exists), while the
		// pool is paused from the menu bar, and while this worker sits above
		// the live train cap.
		if !p.ready() || p.paused.Load() || !gate() {
			select {
			case <-p.ctx.Done():
				return
			case <-p.wake:
			case <-time.After(200 * time.Millisecond):
			}
			continue
		}
		job, ok := p.claim(kinds)
		if !ok {
			select {
			case <-p.ctx.Done():
				return
			case <-p.wake:
			case <-time.After(200 * time.Millisecond):
			}
			continue
		}
		p.runJob(job)
	}
}

// claim pops the next queued job of one of kinds (flips it to running) or reports
// nothing available.
func (p *Pool) claim(kinds []string) (jobs.Job, bool) {
	job, ok, err := p.store.ClaimNextQueued(p.ctx, p.now().Unix(), kinds...)
	if err != nil {
		if p.ctx.Err() == nil {
			p.log.Printf("ERROR: claim next queued: %v", err)
		}
		return jobs.Job{}, false
	}
	if !ok {
		return jobs.Job{}, false
	}
	p.log.Printf("job %s started: kind=%s epochs=%d", job.Key, job.Kind, job.Epochs)
	p.publishStats()
	return job, true
}

// runJob materializes scratch, spawns the trainer, supervises it, reaps it, and
// records the terminal outcome. The scratch dir is unique PER ATTEMPT (not keyed
// by job.Key), so a lagging worker from a since-deleted run can never delete a
// re-submitted run's live scratch (the delete-and-resubmit retry path reuses the
// content key). It is torn down on every exit path.
func (p *Pool) runJob(job jobs.Job) {
	defer p.publishStats()

	scratch, err := os.MkdirTemp(p.scratchRoot, job.Key+"-")
	if err != nil {
		p.log.Printf("job %s: create scratch: %v", job.Key, err)
		p.finishFailed(job, "scratch", err.Error())
		return
	}
	defer os.RemoveAll(scratch)

	capturePath := filepath.Join(scratch, "capture.wav")
	outdir := filepath.Join(scratch, "out")
	resumeCkpt, err := p.materialize(job, scratch, capturePath, outdir)
	if err != nil {
		p.log.Printf("job %s: materialize failed: %v", job.Key, err)
		p.finishFailed(job, "materialize", err.Error())
		return
	}

	proc, err := p.runner.Spawn(Spec{
		Signal:     p.signalPath,
		Capture:    capturePath,
		Outdir:     outdir,
		Name:       "model",
		Epochs:     job.Epochs,
		Arch:       job.Arch,
		ResumeCkpt: resumeCkpt,
	})
	if err != nil {
		p.log.Printf("job %s: spawn failed: %v", job.Key, err)
		p.finishFailed(job, "spawn", err.Error())
		return
	}
	defer proc.Close() // release the output pipe read end on every exit path

	entry := &procEntry{pgid: proc.Pgid}
	p.register(job.Key, entry)
	defer p.unregister(job.Key, entry)

	// Record the pgid. If the row already left running (deleted in the tiny
	// claim→register window), kill the child we just spawned — no DELETE could.
	if ok, err := p.store.SetJobPID(p.ctx, job.Key, proc.Pgid); err != nil {
		p.log.Printf("job %s: record pid: %v", job.Key, err)
	} else if !ok {
		entry.kill(reasonDelete)
	}
	// Same closure for a kill-Pause that fired inside that window: its procs
	// snapshot could not see this entry, so re-check the flag now that it can.
	if p.pauseKill.Load() {
		entry.kill(reasonPause)
	}

	oc := p.supervise(job, proc, entry)

	reason := entry.beginReap()
	waitErr := proc.Wait()

	p.classify(job, outdir, reason, oc, waitErr)
}

// outcome carries what stdout parsing decided, consumed by classify.
type outcome struct {
	selfVerdict string   // probe_self: "", pass, fail
	selfESR     *float64 // probe_self: replicate ESR if seen
	driverESR   *float64 // train + probe_e10: the final "DRIVER: esr=" value
	driverNA    bool     // "DRIVER: esr=na"
	driverSeen  bool     // a "DRIVER: esr=" line was seen
	sawEpoch    bool     // at least one "Epoch " line was seen (tracker.have) — for a
	// failed train_more this distinguishes a run that got past ckpt restore
	// (train_failed) from one that died at/before it (resume_failed).
}

// supervise streams the child's merged output until EOF, recording progress and
// job_log lines and, for probes, the verdict — killing the group the instant a
// self-ESR verdict is known. A stall watchdog and a shutdown watcher can also end
// the run by killing the group.
func (p *Pool) supervise(job jobs.Job, proc *Proc, entry *procEntry) outcome {
	var oc outcome
	var tracker epochTracker
	var lastWrite time.Time

	// This run's started_at fences every progress/log write so a lagging worker
	// from a since-deleted run can't write onto a key-reusing new run.
	var startedAt int64
	if job.StartedAt != nil {
		startedAt = *job.StartedAt
	}

	watchdog := time.AfterFunc(p.stall, func() { entry.kill(reasonStall) })
	defer watchdog.Stop()

	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-p.ctx.Done():
			entry.kill(reasonShutdown)
		case <-done:
		}
	}()

	lr := newLineReader(proc.Out)
	for {
		line, err := lr.next()
		if err != nil {
			break // EOF: the whole group closed the pipe
		}
		watchdog.Reset(p.stall)
		if line == "" {
			continue
		}
		if e := p.store.AppendLog(p.ctx, job.Key, line, startedAt); e != nil && p.ctx.Err() == nil {
			p.log.Printf("job %s: append log: %v", job.Key, e)
		}

		if ep := parseEpoch(line); ep >= 0 {
			if tracker.observe(ep, time.Now()) {
				if now := time.Now(); now.Sub(lastWrite) >= time.Second {
					lastWrite = now
					_ = p.store.UpdateProgress(p.ctx, job.Key, tracker.lastEpoch, tracker.sPerEpoch, startedAt)
				}
			}
		}

		switch job.Kind {
		case jobs.KindProbeSelf:
			if oc.selfVerdict == "" {
				if v, ok := parseReplicateESR(line); ok {
					oc.selfESR = &v
				}
				switch {
				case isProbeFail(line):
					oc.selfVerdict = jobs.VerdictFail
					entry.kill(reasonVerdict)
				case parseEpoch(line) >= 0: // training started ⇒ checks passed
					oc.selfVerdict = jobs.VerdictPass
					entry.kill(reasonVerdict)
				}
			}
		default: // train and probe_e10 both print a final "DRIVER: esr=" line
			if v, isNA, ok := parseDriverESR(line); ok {
				oc.driverSeen, oc.driverNA = true, isNA
				if !isNA {
					oc.driverESR = &v
				}
			}
		}
	}

	// Final progress flush so the last epoch isn't lost to the 1/s throttle.
	if tracker.have {
		_ = p.store.UpdateProgress(p.ctx, job.Key, tracker.lastEpoch, tracker.sPerEpoch, startedAt)
	}
	// tracker.have is the "training demonstrably resumed" signal classify keys a
	// failed train_more on: an Epoch line only prints once the child is past ckpt
	// restore (Lightning resumes numbering at start_epoch).
	oc.sawEpoch = tracker.have
	return oc
}

// classify writes the terminal state. Reason wins over exit code: a killed child
// returns a nonzero exit, but a delete/shutdown/verdict is not a failure. Terminal
// writes use a fresh context so a concurrent shutdown can't cancel them.
func (p *Pool) classify(job jobs.Job, outdir, reason string, oc outcome, waitErr error) {
	ctx := context.Background()
	now := p.now().Unix()

	if reason == reasonDelete {
		p.log.Printf("job %s: killed by delete", job.Key)
		return // row is gone; nothing to write
	}
	// A shutdown or pause that actually killed the child mid-run requeues it
	// (never a failure). But a child that had already exited on its own
	// (waitErr==nil) finished BEFORE the kill landed — honor its real result via
	// the normal path below, so a completed run is not discarded and re-run.
	if (reason == reasonShutdown || reason == reasonPause) && waitErr != nil {
		if err := p.store.RequeueJob(ctx, job.Key); err != nil {
			p.log.Printf("job %s: requeue on %s: %v", job.Key, reason, err)
		} else {
			p.log.Printf("job %s: requeued (%s)", job.Key, reason)
		}
		return
	}

	switch job.Kind {
	case jobs.KindProbeSelf:
		if oc.selfVerdict != "" {
			ok, err := p.store.FinishProbeSelf(ctx, job.Key, now, oc.selfVerdict, oc.selfESR)
			p.done(ok, err, job.Key, "probe_self "+oc.selfVerdict)
			return
		}
		if reason == reasonStall {
			p.finishFailed(job, "stalled", "no output within the stall window")
			return
		}
		p.finishFailed(job, "no_verdict", "probe ended without a verdict")

	case jobs.KindProbeE10:
		// Honor a produced ESR before the stall reason: like the shutdown branch, a
		// run that yielded its result before the watchdog kill landed is not discarded.
		if oc.driverSeen && !oc.driverNA {
			// A probe_e10 that ran to natural exit exported model.ckpt (the probe's 10
			// epochs can seed a train_more); a stall/verdict kill before that export
			// leaves none → nil, no results row, has_model stays false.
			ckpt := readOptionalCkpt(filepath.Join(outdir, "model.ckpt"))
			ok, err := p.store.FinishProbeE10(ctx, job.Key, now, *oc.driverESR, ckpt)
			p.done(ok, err, job.Key, "probe_e10 esr")
			return
		}
		if reason == reasonStall {
			p.finishFailed(job, "stalled", "no output within the stall window")
			return
		}
		p.finishFailed(job, "no_esr", "probe produced no ESR")

	default: // train and train_more
		// A model on disk from a clean exit wins over a stall reason (same rule as
		// the shutdown branch: a completed run is not thrown away and re-run).
		modelPath := filepath.Join(outdir, "model.nam")
		if waitErr == nil && fileExists(modelPath) {
			// The final validation ESR (DRIVER: esr=) is stored alongside the model
			// so the client can show convergence, not just "trained · N ep".
			p.finishTrainSuccess(ctx, job, now, modelPath, outdir, oc.driverESR)
			return
		}
		if reason == reasonStall {
			p.finishFailed(job, "stalled", "no output within the stall window")
			return
		}
		// A failed train_more that never printed an Epoch line died at/before ckpt
		// restore → resume_failed; once training demonstrably resumed, a later crash
		// is a plain train_failed (an OOM at epoch 350 is not a checkpoint problem).
		code := "train_failed"
		if job.Kind == jobs.KindTrainMore && !oc.sawEpoch {
			code = "resume_failed"
		}
		p.finishFailed(job, code, exitMessage(waitErr))
	}
}

func (p *Pool) finishTrainSuccess(ctx context.Context, job jobs.Job, now int64, modelPath, outdir string, esr *float64) {
	nam, err := os.ReadFile(modelPath)
	if err != nil {
		p.finishFailed(job, "train_failed", "model unreadable: "+err.Error())
		return
	}
	trainJSON, _ := os.ReadFile(filepath.Join(outdir, "model.train.json")) // optional
	// The checkpoint is optional: a run that exported none is still a success, just
	// not continuable (missing/empty → nil, and the store writes no ckpt).
	ckpt := readOptionalCkpt(filepath.Join(outdir, "model.ckpt"))
	ok, err := p.store.FinishTrainSuccess(ctx, job.Key, now, nam, string(trainJSON), esr, ckpt)
	p.done(ok, err, job.Key, "succeeded")
}

// readOptionalCkpt reads <outdir>/model.ckpt, returning nil when it is absent or
// empty. The driver exports it atomically (tmp + rename), so a read never sees a
// torn file; a stall-kill before the export simply leaves none.
func readOptionalCkpt(path string) []byte {
	b, err := os.ReadFile(path)
	if err != nil || len(b) == 0 {
		return nil
	}
	return b
}

func (p *Pool) finishFailed(job jobs.Job, code, msg string) {
	ok, err := p.store.FinishFailed(context.Background(), job.Key, p.now().Unix(), code, msg)
	p.done(ok, err, job.Key, "failed/"+code)
}

// done logs the terminal transition, tolerating ok=false (the job was deleted
// mid-run — the row is gone and there is nothing to record).
func (p *Pool) done(ok bool, err error, key, what string) {
	if err != nil {
		p.log.Printf("job %s: finish (%s): %v", key, what, err)
		return
	}
	if !ok {
		p.log.Printf("job %s: %s but row already gone (deleted mid-run)", key, what)
		return
	}
	p.log.Printf("job %s: %s", key, what)
}

// materialize fills the (already-created, unique) scratch dir: the capture wav
// from the blob and an empty outdir. The provisioned signal (--input, a sha-pinned
// download) is a shared file, not copied. For a train_more job (StartEpoch set) it
// also materializes the parent-checkpoint snapshot to <scratch>/resume.ckpt and
// returns its path (the Spec.ResumeCkpt the driver resumes from); a missing
// snapshot is a clean failure — never a from-scratch run of a continuation.
func (p *Pool) materialize(job jobs.Job, scratch, capturePath, outdir string) (resumeCkpt string, err error) {
	if err := os.MkdirAll(outdir, 0o755); err != nil {
		return "", err
	}
	blob, ok, err := p.store.AudioBlob(context.Background(), job.Key)
	if err != nil {
		return "", err
	}
	if !ok {
		return "", errors.New("capture blob missing")
	}
	if err := os.WriteFile(capturePath, blob, 0o644); err != nil {
		return "", err
	}
	if job.StartEpoch == nil {
		// Belt (crew F4): the HTTP+store path always stamps start_epoch on a
		// train_more; a row without one can only come from a buggy direct caller,
		// and running it from scratch is the one behavior this path forbids.
		if job.Kind == jobs.KindTrainMore {
			return "", errors.New("train_more without start_epoch — refusing to run from scratch")
		}
		return "", nil // a from-scratch train/probe — no checkpoint to resume from
	}
	ckpt, ok, err := p.store.ResumeCkpt(context.Background(), job.Key)
	if err != nil {
		return "", err
	}
	if !ok {
		// A train_more with no snapshot must NOT silently retrain from scratch —
		// fail it cleanly so the client re-seeds with a fresh train.
		return "", errors.New("resume checkpoint missing")
	}
	resumePath := filepath.Join(scratch, "resume.ckpt")
	if err := os.WriteFile(resumePath, ckpt, 0o644); err != nil {
		return "", err
	}
	return resumePath, nil
}

func (p *Pool) register(key string, e *procEntry) {
	p.mu.Lock()
	p.procs[key] = e
	p.mu.Unlock()
}

// unregister removes the entry ONLY if it is still e — a compare-and-delete. At
// cap>=2 a delete+resubmit of the same content key can have a NEW worker overwrite
// procs[key] before this (old) worker's deferred unregister runs; an unconditional
// delete would then drop the new attempt's entry, orphaning a trainer that DELETE
// could no longer reach. Comparing keeps each worker's teardown to its own child.
func (p *Pool) unregister(key string, e *procEntry) {
	p.mu.Lock()
	if p.procs[key] == e {
		delete(p.procs, key)
	}
	p.mu.Unlock()
}

// publishStats recomputes the /v1/health counters and the moving-average
// seconds/epoch and publishes them. Serialized so a stale snapshot from one lane
// worker can't overwrite a fresher one from another.
func (p *Pool) publishStats() {
	p.countsMu.Lock()
	defer p.countsMu.Unlock()
	ctx := context.Background()
	if p.onCounts != nil {
		// On a read error keep the last published counts rather than reporting a
		// fabricated 0 (which would wrongly drop the keep-awake assertion mid-train),
		// but log it — this is the only reconcile path, so a silent miss strands the
		// assertion until the next queue transition.
		if r, q, err := p.store.CountByState(ctx); err == nil {
			p.onCounts(r, q)
		} else {
			p.log.Printf("publish queue counts: %v", err)
		}
	}
	if p.onAvg != nil {
		if avg, err := p.store.AvgSPerEpoch(ctx); err == nil {
			p.onAvg(avg)
		} else {
			p.log.Printf("publish avg s/epoch: %v", err)
		}
	}
}

func fileExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir() && st.Size() > 0
}

func exitMessage(waitErr error) string {
	if waitErr == nil {
		return "trainer exited 0 but produced no model"
	}
	return "trainer exited: " + waitErr.Error()
}
