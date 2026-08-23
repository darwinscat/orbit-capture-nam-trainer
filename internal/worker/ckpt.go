// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"context"
	"sync"
	"time"

	"orbit-capture-nam-trainer/internal/jobs"
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
	esr      *float64 // the figure the driver printed for that epoch, carried in from the reader
	wake     chan struct{}
	done     chan struct{}
	finished chan struct{} // closed by the goroutine once its last look is over
	once     sync.Once
}

func (p *Pool) startCkptSaver(ctx context.Context, job jobs.Job, scratch string) *ckptSaver {
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
				_, esr := s.take()
				p.storeNewestPair(ctx, job, scratch, esr)
				return
			case <-s.wake:
				if had, esr := s.take(); had {
					p.storeNewestPair(ctx, job, scratch, esr)
				}
			}
		}
	}()
	return s
}

// nudge says "an epoch finished". Never blocks, never allocates a backlog.
func (s *ckptSaver) nudge(esr *float64) {
	s.mu.Lock()
	s.pending = true
	s.esr = esr
	s.mu.Unlock()
	select {
	case s.wake <- struct{}{}:
	default: // a wake is already in flight; it will see the flag
	}
}

func (s *ckptSaver) take() (bool, *float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	was, esr := s.pending, s.esr
	s.pending = false
	return was, esr
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

// storeNewestPair finds the newest INTACT pair on disk and puts it in the library. Newest-first with
// the torn one skipped, exactly as the stop harvest reads it: a checkpoint is written while the next
// epoch is already running, so the newest file on disk is sometimes half a file. There is nothing to
// do about that but take the one behind it — and, unlike at kill time, this runs again in seconds.
// esr is the figure the reader already had in its hand for the epoch that triggered this. It used to
// be looked up here with stopESR, which reads the job's ENTIRE log to find one line — once per epoch,
// over the network, on a log that grows with every epoch. That is O(n²) traffic across a run, for a
// number that was three lines away.
func (p *Pool) storeNewestPair(ctx context.Context, job jobs.Job, scratch string, esr *float64) {
	for _, c := range selectLastCkpt(scratch) {
		nam, ckpt, ok := qualifyPair(c.path)
		if !ok {
			continue
		}
		// BOUNDED, because an unbounded one is the worst possible failure here. A TCP socket to the
		// library that dies without an RST — a sleeping machine, a Wi-Fi drop — leaves Exec waiting
		// for minutes, and this is a single goroutine: while it waits, no epoch is kept, and the
		// training goes right on. The run would reach epoch 120 with the library still holding
		// epoch 4, which is precisely the loss this table exists to prevent, arrived at quietly.
		// Thirty seconds is many times a healthy write of 800 kB, and the next epoch is another try.
		wctx, cancel := context.WithTimeout(ctx, 30*time.Second)
		err := p.store.PutCheckpoint(wctx, job.ID, job.ClaimToken, int(c.epoch)+1,
			esr, nam, ckpt, sha256Hex(nam))
		cancel()
		if err != nil && ctx.Err() == nil {
			// Not fatal and not retried here: the next epoch is another attempt, and a run that
			// cannot reach the library has bigger news than a missed checkpoint.
			p.log.Printf("job %d: keep epoch %d: %v", job.ID, c.epoch+1, err)
		}
		return
	}
}
