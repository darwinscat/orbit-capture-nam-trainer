// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package worker owns the training subprocess lifecycle: a pool of `cap` workers
// drains the SHARED queue (jobs rows, claimed per lane with FOR UPDATE SKIP
// LOCKED), each spawning ONE trainer child in its own process group, streaming its
// output into progress + job_log, answering the app's commands on the row
// (cancel_requested_at / stop_requested_at / live_requested_at), and enforcing
// kill, stall-watchdog, and restart-recovery semantics (the design notes).
//
// The kill/Wait discipline is the load-bearing part:
// every SIGKILL happens-before cmd.Wait(), so a pgid can never be recycled and
// re-killed (an unreaped zombie leader reserves its pgid). External kills (cancel,
// stop, pause, stall, shutdown) go through procEntry.kill and are gated by a `reaping`
// flag the worker sets just before it reaps; the worker also SIGKILLs the whole
// group unconditionally after EOF, so a child that closes stdout while still
// alive is never left behind. The terminal outcome is chosen by the kill REASON
// first (cancel → cancelled, shutdown → requeue, stall → failed, stop/pause → harvest)
// and only then by exit code — so a clean daemon restart never writes an
// in-flight job `failed`.
//
// Every write after the claim is fenced on (id, claim_token, state='running'): a
// straggler of an earlier attempt cannot write onto a newer one, and a row the
// app took away (cancelled after our heartbeat went stale) is simply no longer
// ours — the control poller notices and kills the child.
package worker

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"orbit-capture-nam-trainer/internal/applog"
	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/wav"
)

// DefaultStallTimeout is the no-output kill threshold. Torch import is silent for
// minutes, so this slack is deliberate — do not lower it (the design notes).
const DefaultStallTimeout = 15 * time.Minute

// DefaultControlPoll is how often a running job's command flags are read.
const DefaultControlPoll = 2 * time.Second

// kill reasons
const (
	// reasonCancel is cancel_requested_at: kill the group, keep nothing, the row
	// ends 'cancelled'. Cancel beats stop when both are seen on one poll.
	reasonCancel   = "cancel"
	reasonStall    = "stall"
	reasonVerdict  = "verdict"
	reasonShutdown = "shutdown"
	reasonPause    = "pause"
	// reasonStop is stop_requested_at: the run becomes a NORMAL succeeded job whose
	// model + retained checkpoint are the last completed epoch's pair. It is a PURE
	// first-reason-sticks participant — a stall/shutdown/cancel/pause that already
	// decided the entry wins and this kill no-ops (procEntry.kill honors the first
	// reason), so a stall-first race ends 'stalled' and a cancel race ends cancelled.
	reasonStop = "stop"
	// reasonLost: the row is no longer this attempt's (the app cancelled a stale
	// running row, or the claim was released) — kill the child, write nothing.
	reasonLost = "lost"
)

// Provenance is what this daemon trains with: the resolved nam version (the claim
// filter: required_nam_version), the driver sha, and the signal sha (the
// signal_mismatch check). Known once the runtime is provisioned.
type Provenance = store.Provenance

// Options configures a Pool.
type Options struct {
	Store        *store.Store
	Log          *applog.Logger
	Runner       Runner
	WorkerName   string // workers.name — the hostname
	Instance     string // workers.instance — a fresh uuid per process start
	SignalPath   string // the --input capture signal (materialized embedded wav)
	ScratchRoot  string // parent of per-job scratch dirs
	Cap          int    // initial train-lane width (the GPU-bound lane); live-adjustable via SetCap
	CapLimit     int    // train workers actually SPAWNED (SetCap's ceiling); 0 → Cap
	ProbeCap     int    // probe-lane workers (0 → 1)
	StallTimeout time.Duration
	ControlPoll  time.Duration             // 0 → DefaultControlPoll
	OnCounts     func(running, queued int) // publish live counts (keep-awake); may be nil
	OnAvg        func(*float64)            // publish the moving-average s/epoch; may be nil
	Now          func() time.Time
	Ready        func() bool // workers idle until this is true (runtime provisioned AND schema ok); nil → always ready
	// Profile returns the provenance once the runtime is provisioned; ok=false
	// before that (nothing is claimed — the claim filter needs the nam version).
	Profile func() (Provenance, bool)
}

// laneSpec is one scheduling lane: drained by its own workers. The probe lane runs
// alongside the train lane so a self-ESR verdict is seconds away even while a long
// train occupies the training cap (they time-slice the one GPU).
type laneSpec struct {
	name string
	cap  int
}

// Pool is the worker pool.
type Pool struct {
	store       *store.Store
	log         *applog.Logger
	runner      Runner
	workerName  string
	instance    string
	signalPath  string
	scratchRoot string
	lanes       []laneSpec
	stall       time.Duration
	controlPoll time.Duration
	onCounts    func(running, queued int)
	onAvg       func(*float64)
	now         func() time.Time
	ready       func() bool
	profile     func() (Provenance, bool)

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// paused gates claiming only (the menu-bar control): queued jobs stay
	// queued, running ones are untouched unless Pause killed them explicitly.
	// In-memory by design — a daemon restart resumes. pauseKill covers the
	// claim→register window a kill-Pause's procs snapshot cannot see: a worker
	// that claimed before the gate closed re-checks it right after registering
	// and self-kills.
	paused    atomic.Bool
	pauseKill atomic.Bool

	// trainCap is the LIVE train-lane width (1..the spawned worker count):
	// CapLimit workers are always spawned, and train worker i claims only while
	// i < trainCap. Raising it wakes idle workers instantly; lowering it stops
	// further claims — running jobs finish, nothing is killed.
	trainCap atomic.Int32

	mu    sync.Mutex
	procs map[int64]*procEntry
	wake  chan struct{}

	// countsMu serializes the read-then-publish in publishStats so a stale
	// snapshot from one lane worker can't overwrite a fresher one from another.
	countsMu sync.Mutex
}

// procEntry tracks one running child for external kills. See the kill/Wait
// discipline in the package doc.
//
// It also OWNS this attempt's live-export state: the per-attempt scratch dir and a
// one-snapshot cache of the best-so-far checkpoint. ExportLive captures the
// *procEntry under Pool.mu and reads/fills the cache through that captured pointer
// only, so the compare-and-delete unregister kills the cache together with the
// attempt. Because the cache lives INSIDE the entry — not in a pool-level map
// keyed by job id — a requeued job's later attempt gets a fresh entry with an empty
// cache and can never be served the old attempt's bytes.
type procEntry struct {
	mu      sync.Mutex
	pgid    int
	reason  string
	reaping bool

	scratch string        // per-attempt scratch dir; "" for entries that never live-export (e.g. tests)
	snapMu  sync.Mutex    // guards snap only — independent of mu's kill/reap discipline
	snap    *liveSnapshot // best-so-far checkpoint served last; nil until the first successful export
}

// liveSnapshot is the one-snapshot cache of the best-so-far checkpoint served for a
// running job. It is IMMUTABLE once stored: ExportLive only ever replaces
// procEntry.snap with a freshly built value (never mutates one in place), so a
// *liveSnapshot captured under snapMu is safe to read after the lock is dropped.
type liveSnapshot struct {
	identity string  // the best-checkpoint filename these bytes came from
	nam      []byte  // that checkpoint's same-stem .nam sibling (weights-only, wire-safe)
	epoch    int64   // ABSOLUTE epoch of the snapshot (train_more numbering starts at start_epoch)
	esr      float64 // validation ESR reported for that epoch
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
	poll := o.ControlPoll
	if poll <= 0 {
		poll = DefaultControlPoll
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
	profile := o.Profile
	if profile == nil {
		profile = func() (Provenance, bool) { return Provenance{}, true }
	}
	capLimit := o.CapLimit
	if capLimit < atLeast1(o.Cap) {
		capLimit = atLeast1(o.Cap)
	}
	p := &Pool{
		store:       o.Store,
		log:         o.Log,
		runner:      o.Runner,
		workerName:  o.WorkerName,
		instance:    o.Instance,
		signalPath:  o.SignalPath,
		scratchRoot: o.ScratchRoot,
		lanes: []laneSpec{
			// The train lane is the GPU-bound one: CapLimit workers are spawned and
			// the LIVE width (trainCap) gates which of them claim. The probe lane is
			// separate so a self-ESR verdict (seconds, kill-on-verdict) runs
			// immediately alongside a long train instead of queueing behind it.
			{name: jobs.LaneTrain, cap: capLimit},
			{name: jobs.LaneProbe, cap: atLeast1(o.ProbeCap)},
		},
		stall:       stall,
		controlPoll: poll,
		onCounts:    o.OnCounts,
		onAvg:       o.OnAvg,
		now:         now,
		ready:       ready,
		profile:     profile,
		procs:       make(map[int64]*procEntry),
		wake:        make(chan struct{}, 1),
	}
	p.trainCap.Store(int32(atLeast1(o.Cap)))
	return p
}

// SetCap resizes the training lane LIVE (the app's train_cap_wanted and the menu
// control): raising it wakes idle workers now; lowering it stops further claims
// and takes effect as running jobs finish — nothing is killed. Clamped to 1..the
// spawned worker count (CapLimit).
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

// Cap reports the live training-lane width (what the heartbeat and the menu show).
func (p *Pool) Cap() int { return int(p.trainCap.Load()) }

// Pause stops the pool claiming new jobs (the menu-bar control). With killRunning, every running
// child is stopped this second and its job is HARVESTED, not discarded: the last completed epoch's
// checkpoint pair is collected before the scratch goes, the job succeeds with `reached` set to what
// the weights actually have, and a Continue picks the take up from there. Only a run stopped before
// its very first epoch has nothing to keep, and that one requeues. Without killRunning, running jobs
// finish their full epoch count ("pause after current").
func (p *Pool) Pause(killRunning bool) {
	p.paused.Store(true)
	defer p.publishStats()
	if !killRunning {
		p.log.Printf("pause: claiming stopped, running jobs will finish")
		return
	}
	// KILLING IS NOT DISCARDING any more: classify harvests the last completed epoch's pair before the
	// scratch is wiped, so this costs the current partial epoch and nothing else. A run stopped before
	// its first epoch has nothing to keep and requeues, which is the same as it always was.
	p.log.Printf("pause: claiming stopped, stopping running jobs (keeping what is trained)")
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

// Paused reports the pause gate (the menu and the heartbeat reflect it).
func (p *Pool) Paused() bool { return p.paused.Load() }

// Running reports how many children this pool has registered right now.
func (p *Pool) Running() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.procs)
}

// Notify nudges an idle worker to check the queue now and republishes the queue
// counts.
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
		lane := ln.name
		trainLane := lane == jobs.LaneTrain
		for i := 0; i < ln.cap; i++ {
			idx := i
			gate := func() bool { return true }
			if trainLane {
				gate = func() bool { return idx < int(p.trainCap.Load()) }
			}
			p.wg.Add(1)
			go p.workerLoop(lane, gate)
		}
		total += ln.cap
	}
	p.log.Printf("worker pool started (%d workers: train=%d of max %d, probe=%d, stall %s, worker %s/%s)",
		total, p.trainCap.Load(), p.lanes[0].cap, p.lanes[1].cap, p.stall, p.workerName, p.instance)
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

func (p *Pool) workerLoop(lane string, gate func() bool) {
	defer p.wg.Done()
	for {
		if p.ctx.Err() != nil {
			return
		}
		// Idle until the runtime is provisioned and the schema is ours (nothing
		// spawns until python exists), while the pool is paused from the menu bar,
		// and while this worker sits above the live train cap.
		if !p.ready() || p.paused.Load() || !gate() {
			select {
			case <-p.ctx.Done():
				return
			case <-p.wake:
			case <-time.After(200 * time.Millisecond):
			}
			continue
		}
		job, ok := p.claim(lane)
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

// claim pops the next queued job of one lane (flips it to running under a fresh
// claim_token) or reports nothing available.
func (p *Pool) claim(lane string) (jobs.Job, bool) {
	prof, ok := p.profile()
	if !ok {
		return jobs.Job{}, false
	}
	job, ok, err := p.store.ClaimNext(p.ctx, lane, p.workerName, p.instance, prof.NamVersion)
	if err != nil {
		if p.ctx.Err() == nil {
			p.log.Printf("ERROR: claim next (%s): %v", lane, err)
		}
		return jobs.Job{}, false
	}
	if !ok {
		return jobs.Job{}, false
	}
	p.log.Printf("job %d (%s) started: kind=%s epochs=%d", job.ID, job.TakeLabel, job.Kind, job.Epochs)
	p.publishStats()
	return job, true
}

// runJob materializes scratch, spawns the trainer, supervises it, reaps it, and
// records the terminal outcome. The scratch dir is unique PER ATTEMPT, so a
// lagging worker from an earlier attempt can never delete a re-claimed run's live
// scratch. It is torn down on every exit path.
func (p *Pool) runJob(job jobs.Job) {
	defer p.publishStats()

	scratch, err := os.MkdirTemp(p.scratchRoot, fmt.Sprintf("job-%d-", job.ID))
	if err != nil {
		p.log.Printf("job %d: create scratch: %v", job.ID, err)
		p.finishFailed(job, jobs.ErrScratch, err.Error())
		return
	}
	defer os.RemoveAll(scratch)

	capturePath := filepath.Join(scratch, "capture.wav")
	outdir := filepath.Join(scratch, "out")
	resumeCkpt, code, err := p.materialize(job, scratch, capturePath, outdir)
	if err != nil {
		p.log.Printf("job %d: materialize failed (%s): %v", job.ID, code, err)
		p.finishFailed(job, code, err.Error())
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
		Latency:    job.Latency, // nil ⇒ no --latency ⇒ the trainer auto-detects
	})
	if err != nil {
		p.log.Printf("job %d: spawn failed: %v", job.ID, err)
		p.finishFailed(job, jobs.ErrSpawn, err.Error())
		return
	}
	defer proc.Close() // release the output pipe read end on every exit path

	// scratch is unique per attempt and carried on the entry so ExportLive can glob
	// this run's checkpoints; it dies with the entry at unregister.
	entry := &procEntry{pgid: proc.Pgid, scratch: scratch}
	p.register(job.ID, entry)
	defer p.unregister(job.ID, entry)

	// Record the pgid. If the row already left us (cancelled/requeued in the tiny
	// claim→register window), kill the child we just spawned — no poll could.
	if ok, err := p.store.SetJobPGID(p.ctx, job.ID, job.ClaimToken, proc.Pgid); err != nil {
		p.log.Printf("job %d: record pgid: %v", job.ID, err)
	} else if !ok {
		entry.kill(reasonLost)
	}
	// Same closure for a kill-Pause that fired inside that window: its procs
	// snapshot could not see this entry, so re-check the flag now that it can.
	if p.pauseKill.Load() {
		entry.kill(reasonPause)
	}

	// The control poller answers cancel/stop/live from the row while the child runs.
	ctlDone := make(chan struct{})
	go p.controlLoop(job, entry, ctlDone)

	oc := p.supervise(job, proc, entry)
	close(ctlDone)

	reason := entry.beginReap()
	waitErr := proc.Wait()

	p.classify(job, outdir, reason, oc, waitErr)
}

// outcome carries what stdout parsing decided, consumed by classify.
type outcome struct {
	selfVerdict string   // probe_self: "", pass, fail
	selfESR     *float64 // probe_self: replicate ESR if seen
	driverESR   *float64 // train: the final "DRIVER: esr=" value
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
	var liveESR *float64 // the driver's newest per-epoch ESR, ridden along on the next progress write
	var lastEpochAt time.Time // when the previous epoch finished — this epoch's cost is the gap

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
		if e := p.store.AppendLog(p.ctx, job.ID, job.ClaimToken, line); e != nil && p.ctx.Err() == nil {
			p.log.Printf("job %d: append log: %v", job.ID, e)
		}

		if ep, v, ok := parseEpochESR(line); ok {
			liveESR = &v // the driver's figure for the epoch just finished; carried on the next write
			// …and kept as a row, so the curve is data and not something to re-parse a log for. The
			// cost recorded is THIS epoch's own: the gap since the previous one finished, which is the
			// only figure that stays true when several runs share the card.
			now := time.Now()
			var secs float64
			if !lastEpochAt.IsZero() {
				secs = now.Sub(lastEpochAt).Seconds()
			}
			lastEpochAt = now
			_ = p.store.RecordEpoch(p.ctx, job.ID, job.ClaimToken, int(ep), &v, secs)
		}
		if ep := parseEpoch(line); ep >= 0 {
			if tracker.observe(ep, time.Now()) {
				if now := time.Now(); now.Sub(lastWrite) >= time.Second {
					lastWrite = now
					_ = p.store.UpdateProgress(p.ctx, job.ID, job.ClaimToken, tracker.lastEpoch, tracker.sPerEpoch, liveESR)
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
		default: // train and train_more print a final "DRIVER: esr=" line
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
		_ = p.store.UpdateProgress(p.ctx, job.ID, job.ClaimToken, tracker.lastEpoch, tracker.sPerEpoch, liveESR)
	}
	// tracker.have is the "training demonstrably resumed" signal classify keys a
	// failed train_more on: an Epoch line only prints once the child is past ckpt
	// restore (Lightning resumes numbering at start_epoch).
	oc.sawEpoch = tracker.have
	return oc
}

// classify writes the terminal state. Reason wins over exit code: a killed child
// returns a nonzero exit, but a cancel/shutdown/verdict is not a failure. Terminal
// writes use a fresh context so a concurrent shutdown can't cancel them.
func (p *Pool) classify(job jobs.Job, outdir, reason string, oc outcome, waitErr error) {
	ctx := context.Background()

	switch reason {
	case reasonLost:
		p.log.Printf("job %d: row no longer ours (cancelled or reclaimed elsewhere) — child killed, nothing written", job.ID)
		return
	case reasonCancel:
		ok, err := p.store.FinishCancelled(ctx, job.ID, job.ClaimToken, p.prov())
		p.done(ok, err, job.ID, "cancelled")
		return
	}
	// A SHUTDOWN requeues; A PAUSE DOES NOT, any more — see the harvest below.
	//
	// The two used to share this branch, and that is why "Pause now" threw away an hour and a half of
	// GPU: the pair of files that holds the last COMPLETED epoch is still on disk at this moment (the
	// scratch wipe is a defer, and this runs first), and nothing looked at it. A shutdown is different
	// in kind — nobody asked for the training to stop, the daemon is merely going away — so its run
	// goes back to the queue to be finished by whoever picks it up.
	//
	// A probe is three seconds and has no checkpoint to keep, so a paused probe requeues like before.
	//
	// A child that had already exited on its own (waitErr==nil) finished BEFORE the kill landed —
	// honour its real result via the normal path below, so a completed run is not discarded and re-run.
	if (reason == reasonShutdown || (reason == reasonPause && job.Kind == jobs.KindProbeSelf)) && waitErr != nil {
		if err := p.store.RequeueJob(ctx, job.ID, job.ClaimToken); err != nil {
			p.log.Printf("job %d: requeue on %s: %v", job.ID, reason, err)
		} else {
			p.log.Printf("job %d: requeued (%s)", job.ID, reason)
		}
		return
	}

	switch job.Kind {
	case jobs.KindProbeSelf:
		if oc.selfVerdict != "" {
			ok, err := p.store.FinishProbeSelf(ctx, job.ID, job.ClaimToken, oc.selfVerdict, oc.selfESR, p.prov())
			p.done(ok, err, job.ID, "probe_self "+oc.selfVerdict)
			return
		}
		if reason == reasonStall {
			p.finishFailed(job, jobs.ErrStalled, "no output within the stall window")
			return
		}
		p.finishFailed(job, jobs.ErrNoVerdict, "probe ended without a verdict")

	default: // train and train_more
		// (1) NATURAL SUCCESS FIRST: a model on disk from a clean exit wins over any
		// kill reason (same rule as the shutdown branch — a completed run is not
		// thrown away and re-run). This is also the "a stop landed as the run
		// finished on its own" case: the FULL natural result is kept, never stop_failed.
		modelPath := filepath.Join(outdir, "model.nam")
		if waitErr == nil && fileExists(modelPath) {
			p.finishTrainSuccess(ctx, job, modelPath, outdir, oc.driverESR)
			return
		}
		// (2) EARLY STOP: harvest the last completed epoch's pair and finish as a
		// NORMAL succeeded run. Placed AFTER natural success so a stop racing a
		// finish keeps the full result. It never collides with the stall branch
		// below: reason is single-valued and first-sticks.
		// …AND A PAUSE IS AN EARLY STOP TOO. Somebody asked for their GPU back; they did not ask to
		// lose the run. The trainer writes a checkpoint/model pair after every epoch, and it is still
		// on disk here — harvesting it turns "kill it" into "stop now, keep everything up to the last
		// finished epoch", which is what a person who needs their own machine actually wants.
		if reason == reasonStop || reason == reasonPause {
			scratch := filepath.Dir(outdir) // harvest globs <scratch>/out/**
			if nam, ckpt, esr, reached, ok := p.harvestStop(ctx, job.ID, scratch); ok {
				okRow, err := p.store.FinishSucceeded(ctx, job.ID, job.ClaimToken,
					store.Result{Reached: reached, ESR: esr, Nam: nam, Ckpt: ckpt}, p.prov())
				p.done(okRow, err, job.ID, fmt.Sprintf("stopped at reached=%d", reached))
				return
			}
			// No harvestable pair, but the driver had already exported model.nam and
			// rmtree'd its work dir when the kill landed → keep the FULL natural result.
			if fileExists(modelPath) {
				p.finishTrainSuccess(ctx, job, modelPath, outdir, oc.driverESR)
				return
			}
			// NOTHING TO KEEP — which for a pause is the ordinary case in the first minutes of a run
			// (torch import, dataset build, epoch 0). That is not a failure: put it back and let it be
			// trained when the machine is free again.
			if reason == reasonPause {
				if err := p.store.RequeueJob(ctx, job.ID, job.ClaimToken); err != nil {
					p.log.Printf("job %d: requeue on pause: %v", job.ID, err)
				} else {
					p.log.Printf("job %d: requeued (paused before the first epoch)", job.ID)
				}
				return
			}
			p.finishFailed(job, jobs.ErrStopFailed, "early stop: no usable checkpoint pair and no exported model")
			return
		}
		// (3) stall / failure branches.
		if reason == reasonStall {
			p.finishFailed(job, jobs.ErrStalled, "no output within the stall window")
			return
		}
		// A failed train_more that never printed an Epoch line died at/before ckpt
		// restore → resume_failed; once training demonstrably resumed, a later crash
		// is a plain train_failed (an OOM at epoch 350 is not a checkpoint problem).
		code := jobs.ErrTrainFailed
		if job.Kind == jobs.KindTrainMore && !oc.sawEpoch {
			code = jobs.ErrResumeFailed
		}
		p.finishFailed(job, code, exitMessage(waitErr))
	}
}

// finishTrainSuccess stores a natural finish: reached = the epochs the trainer was
// spawned with (job.Epochs — passed explicitly, never read back from the row), the
// final validation ESR, the .nam and the optional checkpoint.
func (p *Pool) finishTrainSuccess(ctx context.Context, job jobs.Job, modelPath, outdir string, esr *float64) {
	nam, err := os.ReadFile(modelPath)
	if err != nil {
		p.finishFailed(job, jobs.ErrTrainFailed, "model unreadable: "+err.Error())
		return
	}
	// The checkpoint is optional: a run that exported none is still a success, just
	// not continuable (missing/empty → nil, and the store writes no ckpt).
	ckpt := readOptionalCkpt(filepath.Join(outdir, "model.ckpt"))
	ok, err := p.store.FinishSucceeded(ctx, job.ID, job.ClaimToken,
		store.Result{Reached: int64(job.Epochs), ESR: esr, Nam: nam, Ckpt: ckpt}, p.prov())
	p.done(ok, err, job.ID, "succeeded")
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
	ok, err := p.store.FinishFailed(context.Background(), job.ID, job.ClaimToken, code, msg, p.prov())
	p.done(ok, err, job.ID, "failed/"+code)
}

// prov is the provenance stamped on every terminal row.
func (p *Pool) prov() store.Provenance {
	prof, _ := p.profile()
	return prof
}

// done logs the terminal transition, tolerating ok=false (the row is no longer
// this attempt's — nothing to record).
func (p *Pool) done(ok bool, err error, id int64, what string) {
	if err != nil {
		p.log.Printf("job %d: finish (%s): %v", id, what, err)
		return
	}
	if !ok {
		p.log.Printf("job %d: %s but the row is no longer ours (cancelled or reclaimed)", id, what)
		return
	}
	p.log.Printf("job %d: %s", id, what)
}

// materialize fills the (already-created, unique) scratch dir: the take's wav from
// take_audio (sha-verified against the bytes the job pins and header-validated),
// and an empty outdir. The provisioned signal (--input) is a shared file, not
// copied; its sha must match the stimulus the take was recorded against
// (signal_mismatch otherwise). For a train_more job the app's job_resume snapshot
// is materialized to <scratch>/resume.ckpt and its path returned (Spec.ResumeCkpt);
// a missing snapshot is a clean base_unavailable — never a from-scratch run of a
// continuation. code is the job_error to fail with on error.
func (p *Pool) materialize(job jobs.Job, scratch, capturePath, outdir string) (resumeCkpt, code string, err error) {
	ctx := context.Background()
	if err := os.MkdirAll(outdir, 0o755); err != nil {
		return "", jobs.ErrScratch, err
	}
	prof, _ := p.profile()
	sig, ok, err := p.store.TakeSignalSHA(ctx, job.TakeID)
	if err != nil {
		return "", jobs.ErrMaterialize, err
	}
	if !ok {
		return "", jobs.ErrTakeGone, fmt.Errorf("take %d not found", job.TakeID)
	}
	if sig != prof.SignalSHA256 {
		return "", jobs.ErrSignalMismatch,
			fmt.Errorf("take %d was recorded against signal %.12s…, this daemon trains with %.12s…", job.TakeID, sig, prof.SignalSHA256)
	}
	wavBytes, sha, ok, err := p.store.TakeAudio(ctx, job.TakeID)
	if err != nil {
		return "", jobs.ErrMaterialize, err
	}
	if !ok {
		return "", jobs.ErrTakeGone, fmt.Errorf("take %d has no audio", job.TakeID)
	}
	if sum := sha256Hex(wavBytes); sum != sha || sum != job.WavSHA {
		return "", jobs.ErrMaterialize,
			fmt.Errorf("take %d audio sha %.12s… does not match take_audio %.12s… / job %.12s…", job.TakeID, sum, sha, job.WavSHA)
	}
	if _, err := wav.Validate(wavBytes); err != nil {
		return "", jobs.ErrWavInvalid, err
	}
	if err := os.WriteFile(capturePath, wavBytes, 0o644); err != nil {
		return "", jobs.ErrScratch, err
	}
	if job.Kind != jobs.KindTrainMore {
		return "", "", nil // a from-scratch train/probe — no checkpoint to resume from
	}
	if job.StartEpoch == nil {
		// Belt: the schema CHECKs start_epoch on a train_more; a row without one can
		// only come from a buggy writer, and running it from scratch is the one
		// behavior this path forbids.
		return "", jobs.ErrBaseUnavailable, errors.New("train_more without start_epoch — refusing to run from scratch")
	}
	ckpt, ok, err := p.store.ResumeCkpt(ctx, job.ID)
	if err != nil {
		return "", jobs.ErrMaterialize, err
	}
	if !ok {
		return "", jobs.ErrBaseUnavailable, errors.New("job_resume snapshot missing")
	}
	resumePath := filepath.Join(scratch, "resume.ckpt")
	if err := os.WriteFile(resumePath, ckpt, 0o644); err != nil {
		return "", jobs.ErrScratch, err
	}
	return resumePath, "", nil
}

func (p *Pool) register(id int64, e *procEntry) {
	p.mu.Lock()
	p.procs[id] = e
	p.mu.Unlock()
}

// unregister removes the entry ONLY if it is still e — a compare-and-delete, so a
// lagging teardown of an earlier attempt can never drop a newer attempt's entry.
func (p *Pool) unregister(id int64, e *procEntry) {
	p.mu.Lock()
	if p.procs[id] == e {
		delete(p.procs, id)
	}
	p.mu.Unlock()
}

// publishStats recomputes the queue counters and the moving-average
// seconds/epoch and publishes them. Serialized so a stale snapshot from one lane
// worker can't overwrite a fresher one from another.
func (p *Pool) publishStats() {
	p.countsMu.Lock()
	defer p.countsMu.Unlock()
	ctx := context.Background()
	if p.onCounts != nil {
		// On a read error keep the last published counts rather than reporting a
		// fabricated 0 (which would wrongly drop the keep-awake assertion mid-train).
		if r, q, err := p.store.CountByState(ctx, p.workerName); err == nil {
			p.onCounts(r, q)
		} else {
			p.log.Printf("publish queue counts: %v", err)
		}
	}
	if p.onAvg != nil {
		if avg, err := p.store.AvgSPerEpoch(ctx, p.workerName); err == nil {
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

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
