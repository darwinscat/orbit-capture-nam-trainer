// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"context"
	"sync"
	"time"

	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/store"
)

// ckptSaver carries each finished epoch's checkpoint pair from the scratch directory into the
// library, so the run survives this process — see migration 0006.
//
// IT COALESCES, IT DOES NOT QUEUE. The reader of the trainer's output must never wait on a database:
// an epoch says "there is something new", and this goroutine writes whatever is newest WHEN IT GETS
// THERE. If the library is slow, or an epoch is quicker than a round trip, the epochs in between are
// simply skipped rather than piling up behind each other — the row is meant to be recent, never
// complete. Skipping costs the difference between two epochs; a backlog would cost memory, ordering,
// and eventually a write of stale weights over fresh ones.
//
// One goroutine per running job, so writes for a job are ordered by construction.
type ckptSaver struct {
	mu       sync.Mutex
	pending  bool
	epoch    int      // the epoch the reader just saw finish…
	esr      *float64 // …and the figure it printed for it
	wake     chan struct{}
	done     chan struct{}
	finished chan struct{} // closed by the goroutine once its last look is over
	once     sync.Once
	// The best pair changes when validation improves, which after the first dozens of epochs is
	// almost never — and it was being read off disk and pushed through the wire on EVERY epoch. On an
	// eight-hundred-epoch run that is hundreds of megabytes of identical bytes through libpq, TOAST
	// and the write-ahead log. Touched only by the saver's own goroutine.
	sentBestSHA string
}

func (p *Pool) startCkptSaver(ctx context.Context, job jobs.Job, scratch string, child *procEntry) *ckptSaver {
	s := &ckptSaver{wake: make(chan struct{}, 1), done: make(chan struct{}), finished: make(chan struct{})}
	go func() {
		defer close(s.finished)
		for {
			select {
			case <-s.done:
				// ONE LAST LOOK ON THE WAY OUT, AND stop() WAITS FOR IT. On a natural finish this
				// writes nothing that lasts — the job turns terminal and the trigger takes the row —
				// but on the requeue paths it is the only writer of the final epoch, and that is what
				// this table exists for. It matters more than it looks: the reader nudges on the
				// driver's per-epoch ESR line, and Lightning writes the checkpoint pair AFTER that,
				// in a later callback. So a nudge usually stores the PREVIOUS epoch's pair and the
				// row runs one epoch behind the run. This last look is where it catches up.
				//
				// There is no ctx.Done() arm beside this one on purpose. Go picks among ready arms at
				// random, so an arm that returned without writing would silently drop the last epoch
				// on some shutdowns and not others — the worst kind of bug to be handed. The write
				// below carries its own deadline instead.
				_, epoch, esr := s.take()
				p.storeNewestPair(ctx, job, scratch, child, s, epoch, esr)
				return
			case <-s.wake:
				if had, epoch, esr := s.take(); had {
					p.storeNewestPair(ctx, job, scratch, child, s, epoch, esr)
				}

			}
		}
	}()
	return s
}

// nudge says "an epoch finished". Never blocks, never allocates a backlog.
func (s *ckptSaver) nudge(epoch int, esr *float64) {
	s.mu.Lock()
	s.pending = true
	s.epoch, s.esr = epoch, esr
	s.mu.Unlock()
	select {
	case s.wake <- struct{}{}:
	default: // a wake is already in flight; it will see the flag
	}
}

func (s *ckptSaver) take() (bool, int, *float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	was, epoch, esr := s.pending, s.epoch, s.esr
	s.pending = false
	return was, epoch, esr
}

// stop ends the saver AND WAITS for its last look to finish. The wait is the point: without it the
// final write races the two things that happen next — the requeue, which takes the claim away and
// makes the write a no-op, and the removal of the scratch directory, which takes the file away. Both
// win easily over a database round trip, so "the last epoch is kept" was true only by luck.
//
// Bounded, because a caller waiting here is holding up the reaping of a child. The write inside
// carries a thirty-second deadline; this allows a little more than that and then gives up, which is
// the same answer as before this wait existed.
func (s *ckptSaver) stop() {
	s.once.Do(func() { close(s.done) })
	select {
	case <-s.finished:
	case <-time.After(40 * time.Second):
	}
}

// storeNewestPair puts this run's newest work in the library and obeys what the library says back.
//
// TWO PAIRS, because two readers want two different answers and both are already on disk. The LAST
// completed epoch is what a continuation must resume from — resuming from an older one silently
// redoes work. The BEST by validation ESR is what a person wants to listen to. Storing one meant
// serving the other badly.
//
// Newest-first with the torn one skipped: a checkpoint is written while the next epoch is already
// running, so the newest file is sometimes half a file. Unlike at kill time there is nothing to do
// about that but take the one behind it — and this runs again in seconds.
//
// AND IF THE LIBRARY SAYS THE RUN IS NOT OURS, WE STOP. Not an error, not a retry: the fence names
// the claim and `paused_at`, so a false answer means the take has been handed to another machine or
// marked claimable because this task went silent. Whatever is training here is training over
// somebody else's work.
//
// epoch/esr are what the reader just saw finish, and they are used for the pair being stored ONLY
// when it IS that epoch — a torn newest checkpoint sends this one back, the best-pair fallback sends
// it further, and attaching one epoch's figure to another's weights would be a lie. For those,
// job_epochs answers in one indexed row. The reader's own value is preferred where it applies because
// it is exact: job_epochs.esr is a `real`, and the round trip through it costs precision the job's
// own double-precision column would otherwise keep.
func (p *Pool) storeNewestPair(ctx context.Context, job jobs.Job, scratch string, child *procEntry, s *ckptSaver, epoch int, esr *float64) {
	var best *Pair
	if b, found := selectBestCkpt(scratch); found {
		if bp, good := readPair([]ckptChoice{b}, nil); good {
			if b.esr > 0 {
				e := b.esr
				bp.ESR = &e
			}
			best = &bp
		}
	}
	last, ok := readPair(selectLastCkpt(scratch), nil)
	if !ok {
		// No whole `last` pair — mid-rotation, or the trainer only ever wrote `best` names. The best
		// one is then the newest whole thing on disk, and it is what the driver's own final export
		// falls back to as well. Without this a stop landing in that window finds nothing at all.
		if best == nil {
			return // nothing whole yet; the next epoch is another look
		}
		last = *best
	}
	if stored := last.Reached - 1; stored == epoch && esr != nil {
		last.ESR = esr
	} else {
		ectx, ecancel := context.WithTimeout(ctx, 30*time.Second) // bounded like the write below
		last.ESR = p.store.EpochESR(ectx, job.ID, stored)
		ecancel()
	}
	// The library already holds these exact bytes — send nothing rather than the same megabyte again.
	// It is the same pair by content, not by name: sha256 of the .nam is what the row stores too.
	if best != nil && s != nil && best.NamSHA == s.sentBestSHA {
		best = nil
	}
	// BOUNDED, because an unbounded write is the worst failure available here. A socket to the library
	// that dies without an RST leaves Exec waiting for minutes, and this is a single goroutine: while
	// it waits nothing is kept and the training goes right on, arriving quietly at exactly the loss
	// this table exists to prevent. Thirty seconds is many times a healthy write, and the next epoch
	// is another try.
	wctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	v, err := p.store.PutCheckpoint(wctx, job.ID, job.ClaimToken, job.TakeID, last.pair(), bestPair(best))
	cancel()
	if err != nil {
		if ctx.Err() == nil {
			p.log.Printf("job %d: keep epoch %d: %v", job.ID, last.Reached, err)
		}
		return
	}
	if best != nil && s != nil {
		s.sentBestSHA = best.NamSHA
	}
	p.obey(job, child, v)
}

// readPair takes the newest choice whose files are both whole.
func readPair(choices []ckptChoice, esr *float64) (Pair, bool) {
	for _, c := range choices {
		if nam, ckpt, ok := qualifyPair(c.path); ok {
			return Pair{Reached: int(c.epoch) + 1, ESR: esr, Nam: nam, Ckpt: ckpt,
				NamSHA: sha256Hex(nam)}, true
		}
	}
	return Pair{}, false
}

// Pair mirrors store.Pair so this file does not have to know the store's shape at every call site.
type Pair struct {
	Reached int
	ESR     *float64
	Nam     []byte
	Ckpt    []byte
	NamSHA  string
}

func (p Pair) pair() store.Pair {
	return store.Pair{Reached: p.Reached, ESR: p.ESR, Nam: p.Nam, Ckpt: p.Ckpt, NamSHA: p.NamSHA}
}

func bestPair(b *Pair) *store.Pair {
	if b == nil {
		return nil
	}
	sp := b.pair()
	return &sp
}

// obey does what the library said. EVERYTHING THE RUN NEEDS TO KNOW arrives this way — there is no
// second channel. It used to be a poll of three control flags every two seconds, running beside the
// checkpoint write and answering the same question in its own words.
//
// THE KILL GOES TO THIS ATTEMPT'S OWN CHILD, never to whatever is registered under the job id. The
// two are not the same process: a paused row may be re-claimed by an IDLE WORKER OF THIS VERY POOL
// (nothing in ClaimNext excludes the same machine), and its register() replaces the entry under that
// id. The displaced attempt then reads its own fence, learns it lost the row — and, looking the
// child up by id, killed the new attempt's trainer instead of its own. The old child went on burning
// the card with every write fenced out, murdering each successive re-claim at each epoch boundary,
// while the row sat 'running' with no process behind it.
func (p *Pool) obey(job jobs.Job, child *procEntry, v store.Verdict) {
	switch {
	case !v.Mine:
		p.log.Printf("job %d: this run is no longer ours (claimed elsewhere, or paused) — stopping", job.ID)
		child.kill(reasonLost)
	case v.Cancel:
		p.log.Printf("job %d: cancel requested — killing", job.ID)
		child.kill(reasonCancel)
	case v.Stop:
		p.log.Printf("job %d: stop requested — what is in the library is what it keeps", job.ID)
		child.kill(reasonStop)
	}
}
