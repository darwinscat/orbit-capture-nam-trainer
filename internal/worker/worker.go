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
)

// Options configures a Pool.
type Options struct {
	Store          *store.Store
	Log            *applog.Logger
	Runner         Runner
	SignalPath     string                    // the --input capture signal (materialized embedded wav)
	ScratchRoot    string                    // parent of per-job scratch dirs
	Cap            int                       // train-lane workers (the GPU-bound lane)
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
	return &Pool{
		store:       o.Store,
		log:         o.Log,
		runner:      o.Runner,
		signalPath:  o.SignalPath,
		scratchRoot: o.ScratchRoot,
		lanes: []laneSpec{
			// The train lane is the GPU-bound one, sized by cap. The probe lanes are
			// separate so a self-ESR verdict (seconds, kill-on-verdict) and an E@10
			// probe (~10 epochs) run immediately alongside a long train instead of
			// queueing behind it — the whole point of a rig-side self-check.
			{name: "train", kinds: []string{jobs.KindTrain}, cap: atLeast1(o.Cap)},
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
}

// Kill aborts a running job's process group (the httpapi.Killer for DELETE).
func (p *Pool) Kill(key string) {
	p.mu.Lock()
	e := p.procs[key]
	p.mu.Unlock()
	if e != nil {
		e.kill(reasonDelete)
	}
}

// Notify nudges an idle worker to check the queue now (called after a PUT).
func (p *Pool) Notify() {
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
		for i := 0; i < ln.cap; i++ {
			p.wg.Add(1)
			go p.workerLoop(kinds)
		}
		total += ln.cap
	}
	p.log.Printf("worker pool started (%d workers: train=%d probe_self=%d probe_e10=%d, stall %s)",
		total, p.lanes[0].cap, p.lanes[1].cap, p.lanes[2].cap, p.stall)
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

func (p *Pool) workerLoop(kinds []string) {
	defer p.wg.Done()
	for {
		if p.ctx.Err() != nil {
			return
		}
		// Idle until the runtime is provisioned: jobs are accepted and queue while
		// ready is false, but nothing is spawned until python exists.
		if !p.ready() {
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
	if err := p.materialize(job.Key, capturePath, outdir); err != nil {
		p.log.Printf("job %s: materialize failed: %v", job.Key, err)
		p.finishFailed(job, "materialize", err.Error())
		return
	}

	proc, err := p.runner.Spawn(Spec{
		Signal:  p.signalPath,
		Capture: capturePath,
		Outdir:  outdir,
		Name:    "model",
		Epochs:  job.Epochs,
		Arch:    job.Arch,
	})
	if err != nil {
		p.log.Printf("job %s: spawn failed: %v", job.Key, err)
		p.finishFailed(job, "spawn", err.Error())
		return
	}
	defer proc.Close() // release the output pipe read end on every exit path

	entry := &procEntry{pgid: proc.Pgid}
	p.register(job.Key, entry)
	defer p.unregister(job.Key)

	// Record the pgid. If the row already left running (deleted in the tiny
	// claim→register window), kill the child we just spawned — no DELETE could.
	if ok, err := p.store.SetJobPID(p.ctx, job.Key, proc.Pgid); err != nil {
		p.log.Printf("job %s: record pid: %v", job.Key, err)
	} else if !ok {
		entry.kill(reasonDelete)
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
	// A shutdown that actually killed the child mid-run requeues it (never a
	// failure). But a child that had already exited on its own (waitErr==nil)
	// finished BEFORE the shutdown kill landed — honor its real result via the
	// normal path below, so a completed run is not discarded and re-run.
	if reason == reasonShutdown && waitErr != nil {
		if err := p.store.RequeueJob(ctx, job.Key); err != nil {
			p.log.Printf("job %s: requeue on shutdown: %v", job.Key, err)
		} else {
			p.log.Printf("job %s: requeued (shutdown)", job.Key)
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
		if reason == reasonStall {
			p.finishFailed(job, "stalled", "no output within the stall window")
			return
		}
		if oc.driverSeen && !oc.driverNA {
			ok, err := p.store.FinishProbeE10(ctx, job.Key, now, *oc.driverESR)
			p.done(ok, err, job.Key, "probe_e10 esr")
			return
		}
		p.finishFailed(job, "no_esr", "probe produced no ESR")

	default: // train
		if reason == reasonStall {
			p.finishFailed(job, "stalled", "no output within the stall window")
			return
		}
		modelPath := filepath.Join(outdir, "model.nam")
		if waitErr == nil && fileExists(modelPath) {
			// The final validation ESR (DRIVER: esr=) is stored alongside the model
			// so the client can show convergence, not just "trained · N ep".
			p.finishTrainSuccess(ctx, job, now, modelPath, outdir, oc.driverESR)
			return
		}
		p.finishFailed(job, "train_failed", exitMessage(waitErr))
	}
}

func (p *Pool) finishTrainSuccess(ctx context.Context, job jobs.Job, now int64, modelPath, outdir string, esr *float64) {
	nam, err := os.ReadFile(modelPath)
	if err != nil {
		p.finishFailed(job, "train_failed", "model unreadable: "+err.Error())
		return
	}
	trainJSON, _ := os.ReadFile(filepath.Join(outdir, "model.train.json")) // optional
	ok, err := p.store.FinishTrainSuccess(ctx, job.Key, now, nam, string(trainJSON), esr)
	p.done(ok, err, job.Key, "succeeded")
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
// from the blob and an empty outdir. The embedded signal (--input) is a shared
// file, not copied.
func (p *Pool) materialize(key, capturePath, outdir string) error {
	if err := os.MkdirAll(outdir, 0o755); err != nil {
		return err
	}
	blob, ok, err := p.store.AudioBlob(context.Background(), key)
	if err != nil {
		return err
	}
	if !ok {
		return errors.New("capture blob missing")
	}
	return os.WriteFile(capturePath, blob, 0o644)
}

func (p *Pool) register(key string, e *procEntry) {
	p.mu.Lock()
	p.procs[key] = e
	p.mu.Unlock()
}

func (p *Pool) unregister(key string) {
	p.mu.Lock()
	delete(p.procs, key)
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
		if r, q, err := p.store.CountByState(ctx); err == nil {
			p.onCounts(r, q)
		}
	}
	if p.onAvg != nil {
		if avg, err := p.store.AvgSPerEpoch(ctx); err == nil {
			p.onAvg(avg)
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
