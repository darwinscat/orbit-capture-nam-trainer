// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package worker owns the training subprocess lifecycle: a pool of `cap` workers
// drains the SHARED queue (jobs rows, claimed per lane with FOR UPDATE SKIP
// LOCKED), each spawning ONE trainer child in its own process group, streaming its
// output into progress + job_log, answering the app's commands on the row
// (a cancel or a stop, both heard at the checkpoint write), and enforcing
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
	OnCounts     func(running, queued int) // publish live counts (keep-awake); may be nil
	OnAvg        func(*float64)            // publish the moving-average s/epoch; may be nil
	Now          func() time.Time
	Ready        func() bool // workers idle until this is true (runtime provisioned AND schema ok); nil → always ready
	// Profile returns the provenance once the runtime is provisioned; ok=false
	// before that (nothing is claimed — the claim filter needs the nam version).
	Profile func() (Provenance, bool)
	// PauseStatePath is where a pause is REMEMBERED across restarts. A pause used to live only in
	// this process, so restarting the daemon resumed it — and restarting is what an upgrade, a
	// config re-read and a crash all are. The person who paused it was usually sitting at that
	// machine wanting their GPU, and got it taken back by a relaunch they did not ask for. A pause
	// is lifted by a hand and by nothing else.
	//
	// A FILE, not a column: the tray must be able to give somebody their machine back while the
	// library is unreachable, and it must still be given back after the reboot that follows. Empty
	// disables the memory (tests that want a clean pool).
	PauseStatePath string
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
	onCounts    func(running, queued int)
	onAvg       func(*float64)
	now         func() time.Time
	ready       func() bool
	profile     func() (Provenance, bool)
	pauseFile   string

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// paused gates claiming only (the menu-bar control): queued jobs stay
	// queued, running ones are untouched unless Pause killed them explicitly.
	// REMEMBERED ACROSS RESTARTS when Options.PauseStatePath is set — see there
	// for why. pauseKill covers the claim→register window a kill-Pause's procs
	// snapshot cannot see: a worker that claimed before the gate closed
	// re-checks it right after registering and self-kills.
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
// It also carries this attempt's scratch directory, which is where the checkpoint saver looks for the
// pair an epoch just wrote. Per attempt and not per job: a re-claimed job gets a fresh entry, so it
// can never be handed the previous attempt's files.
type procEntry struct {
	mu      sync.Mutex
	pgid    int
	reason  string
	reaping bool

	scratch string // per-attempt scratch dir
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
		stall:     stall,
		onCounts:  o.OnCounts,
		onAvg:     o.OnAvg,
		now:       now,
		ready:     ready,
		profile:   profile,
		pauseFile: o.PauseStatePath,
		procs:     make(map[int64]*procEntry),
		wake:      make(chan struct{}, 1),
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
	p.rememberPause(true)
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
	p.rememberPause(false)
	p.Notify()
}

// rememberPause writes the gate to disk so a restart does not lift it. A failure here is logged and
// not returned: the pause itself has already taken effect in this process, and refusing to pause
// because a file could not be written would be the wrong half to give up.
func (p *Pool) rememberPause(paused bool) {
	if p.pauseFile == "" {
		return
	}
	if !paused {
		if err := os.Remove(p.pauseFile); err != nil && !os.IsNotExist(err) {
			p.log.Printf("could not forget the pause (%s): %v", p.pauseFile, err)
		}
		return
	}
	tmp := p.pauseFile + ".tmp"
	if err := os.WriteFile(tmp, []byte("paused by hand; delete this file or press Resume\n"), 0o644); err != nil {
		p.log.Printf("could not remember the pause (%s): %v", p.pauseFile, err)
		return
	}
	if err := os.Rename(tmp, p.pauseFile); err != nil {
		p.log.Printf("could not remember the pause (%s): %v", p.pauseFile, err)
	}
}

// PauseWasRemembered reports whether a previous run left this trainer paused. Read once at startup,
// before anything is claimed.
func PauseWasRemembered(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
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
	resumeCkpt, _, code, err := p.materialize(job, scratch, capturePath, outdir)
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

	// scratch is unique per attempt and carried on the entry so the saver can glob
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
	oc := p.supervise(job, proc, entry, outdir)

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
func (p *Pool) supervise(job jobs.Job, proc *Proc, entry *procEntry, outdir string) outcome {
	var oc outcome
	var tracker epochTracker
	var lastWrite time.Time
	var liveESR *float64      // the driver's newest per-epoch ESR, ridden along on the next progress write
	var lastEpochAt time.Time // when the previous epoch finished — this epoch's cost is the gap

	watchdog := time.AfterFunc(p.stall, func() { entry.kill(reasonStall) })
	defer watchdog.Stop()

	// EVERY FINISHED EPOCH GOES INTO THE LIBRARY. Only the train lane has anything worth keeping — a
	// probe is three seconds and produces no checkpoint — and only while a job is running. See
	// migration 0006 for what this buys: a requeue costs one unfinished epoch instead of the run.
	var saver *ckptSaver
	if job.Lane == jobs.LaneTrain {
		saver = p.startCkptSaver(context.Background(), job, filepath.Dir(outdir), entry)
		defer saver.stop()
	}

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
			if saver != nil {
				saver.nudge(int(ep), &v) // …and the weights themselves, on their own goroutine
			}
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
		// lose the run.
		//
		// THERE IS NOTHING TO HARVEST ANY MORE. This used to go hunting through the scratch directory
		// at the exact moment of the kill for the newest pair that was not half-written, with a
		// fallback for the torn one a SIGKILL leaves behind — all of it because the weights lived
		// only there. They are in the library now, put there epoch by epoch as the run went, and the
		// saver's last look ran before this did. So stopping is reading a row.
		if reason == reasonStop || reason == reasonPause {
			// …THIS RUN'S ROW, not merely the take's. A continuation stopped in its first minutes has
			// written nothing yet, and what sits on the take is its PARENT's pair — harvesting that
			// closed the continuation `succeeded`, filed the parent's weights as its output and
			// stamped them with the parent's epoch count, so the library grew a duplicate model
			// nobody trained. The row says which run put it there; anything else is not ours to
			// finish with.
			if c, ok, err := p.store.Checkpoint(ctx, job.TakeID); err == nil && ok && c.JobID != nil && *c.JobID == job.ID {
				okRow, err := p.store.FinishSucceeded(ctx, job.ID, job.ClaimToken,
					store.Result{Reached: int64(c.Reached), ESR: c.ESR, Nam: c.Nam}, p.prov())
				p.done(okRow, err, job.ID, fmt.Sprintf("stopped at reached=%d", c.Reached))
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
		// (There was a rule here that threw the take's weights away when a run died before its first
		// epoch while resuming from them — the worry being a checkpoint the driver cannot open taking
		// every later attempt down with it. It is gone. Deciding that weights are poison is a guess,
		// and a wrong guess destroys the only copy of somebody's run; the take's history is a list of
		// every place it has been, and a person picks another point from it in one gesture. Nothing
		// discards a take's weights except a cancel, and even that only moves them aside.)
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
	// THE TAKE GETS THE FINISHED MODEL, not merely the last epoch the saver happened to catch. The
	// driver exports model.nam and model.ckpt at the end, and that pair IS the run's answer — the
	// take's row has to say the same thing, because it is the one place anybody asks.
	if ckpt := readOptionalCkpt(filepath.Join(outdir, "model.ckpt")); len(ckpt) > 0 {
		if _, e := p.store.PutCheckpoint(ctx, job.ID, job.ClaimToken, job.TakeID,
			store.Pair{Reached: job.Epochs, ESR: esr, Nam: nam, Ckpt: ckpt,
				NamSHA: sha256Hex(nam)}, nil); e != nil {
			p.log.Printf("job %d: keep the finished model on the take: %v", job.ID, e)
		}
	}
	ok, err := p.store.FinishSucceeded(ctx, job.ID, job.ClaimToken,
		store.Result{Reached: int64(job.Epochs), ESR: esr, Nam: nam}, p.prov())
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
// (signal_mismatch otherwise). The take's own weights, if it has any, are materialized to
// <scratch>/resume.ckpt and the path returned (Spec.ResumeCkpt) — so every run of a take continues
// from where the take is. A continuation with nothing to continue from is a clean base_unavailable,
// never a from-scratch run. code is the job_error to fail with on error.
func (p *Pool) materialize(job jobs.Job, scratch, capturePath, outdir string) (resumeCkpt string, resumedFrom int, code string, err error) {
	ctx := context.Background()
	if err := os.MkdirAll(outdir, 0o755); err != nil {
		return "", 0, jobs.ErrScratch, err
	}
	prof, _ := p.profile()
	sig, ok, err := p.store.TakeSignalSHA(ctx, job.TakeID)
	if err != nil {
		return "", 0, jobs.ErrMaterialize, err
	}
	if !ok {
		return "", 0, jobs.ErrTakeGone, fmt.Errorf("take %d not found", job.TakeID)
	}
	if sig != prof.SignalSHA256 {
		return "", 0, jobs.ErrSignalMismatch,
			fmt.Errorf("take %d was recorded against signal %.12s…, this daemon trains with %.12s…", job.TakeID, sig, prof.SignalSHA256)
	}
	wavBytes, sha, ok, err := p.store.TakeAudio(ctx, job.TakeID)
	if err != nil {
		return "", 0, jobs.ErrMaterialize, err
	}
	if !ok {
		return "", 0, jobs.ErrTakeGone, fmt.Errorf("take %d has no audio", job.TakeID)
	}
	if sum := sha256Hex(wavBytes); sum != sha || sum != job.WavSHA {
		return "", 0, jobs.ErrMaterialize,
			fmt.Errorf("take %d audio sha %.12s… does not match take_audio %.12s… / job %.12s…", job.TakeID, sum, sha, job.WavSHA)
	}
	if _, err := wav.Validate(wavBytes); err != nil {
		return "", 0, jobs.ErrWavInvalid, err
	}
	if err := os.WriteFile(capturePath, wavBytes, 0o644); err != nil {
		return "", 0, jobs.ErrScratch, err
	}
	// A RUN THIS JOB HAS ALREADY DONE, WHEREVER IT WAS DONE. The library holds the newest completed
	// epoch of any job that was training (migration 0006), so a row here means this is not a fresh
	// start — it is a run being picked up after its trainer died, was killed, was upgraded over, or
	// simply never came back. It is checked BEFORE the kind, because it outranks both answers below:
	// a train continues instead of starting over, and a train_more continues from where IT got to
	// rather than from the parent model it was launched off.
	if job.Lane == jobs.LaneTrain {
		if c, ok, err := p.store.Checkpoint(ctx, job.TakeID); err != nil {
			return "", 0, jobs.ErrMaterialize, err
		} else if ok {
			resumePath := filepath.Join(scratch, "resume.ckpt")
			if err := os.WriteFile(resumePath, c.Ckpt, 0o644); err != nil {
				return "", 0, jobs.ErrScratch, err
			}
			p.log.Printf("job %d: resuming take %d from epoch %d kept in the library",
				job.ID, job.TakeID, c.Reached)
			return resumePath, c.Reached, "", nil
		}
	}
	// A CONTINUATION WITH NOTHING TO CONTINUE FROM DOES NOT RUN FROM SCRATCH. The take's row above is
	// the only source now — job_resume was a copy the app made of the parent's checkpoint, and a copy
	// is what the take's own row replaced. Reaching here on a train_more means the take has no
	// weights: the run it was launched off was cancelled, or its history was picked apart. Running it
	// empty would quietly retrain from zero under a gesture that said "add epochs".
	if job.Kind == jobs.KindTrainMore {
		return "", 0, jobs.ErrBaseUnavailable, errors.New("this take has no weights to continue from")
	}
	return "", 0, "", nil // a from-scratch train/probe
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
