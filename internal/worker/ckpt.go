// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"context"
	"sync"

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
	mu      sync.Mutex
	pending bool
	wake    chan struct{}
	done    chan struct{}
	once    sync.Once
}

func (p *Pool) startCkptSaver(ctx context.Context, job jobs.Job, scratch string) *ckptSaver {
	s := &ckptSaver{wake: make(chan struct{}, 1), done: make(chan struct{})}
	go func() {
		for {
			select {
			case <-s.done:
				// One last look on the way out: the final epoch of a run that ends normally would
				// otherwise be dropped exactly when the answer is most complete.
				s.take()
				p.storeNewestPair(ctx, job, scratch)
				return
			case <-ctx.Done():
				return
			case <-s.wake:
				if s.take() {
					p.storeNewestPair(ctx, job, scratch)
				}
			}
		}
	}()
	return s
}

// nudge says "an epoch finished". Never blocks, never allocates a backlog.
func (s *ckptSaver) nudge() {
	s.mu.Lock()
	s.pending = true
	s.mu.Unlock()
	select {
	case s.wake <- struct{}{}:
	default: // a wake is already in flight; it will see the flag
	}
}

func (s *ckptSaver) take() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	was := s.pending
	s.pending = false
	return was
}

func (s *ckptSaver) stop() { s.once.Do(func() { close(s.done) }) }

// storeNewestPair finds the newest INTACT pair on disk and puts it in the library. Newest-first with
// the torn one skipped, exactly as the stop harvest reads it: a checkpoint is written while the next
// epoch is already running, so the newest file on disk is sometimes half a file. There is nothing to
// do about that but take the one behind it — and, unlike at kill time, this runs again in seconds.
func (p *Pool) storeNewestPair(ctx context.Context, job jobs.Job, scratch string) {
	for _, c := range selectLastCkpt(scratch) {
		nam, ckpt, ok := qualifyPair(c.path)
		if !ok {
			continue
		}
		err := p.store.PutCheckpoint(ctx, job.ID, job.ClaimToken, int(c.epoch)+1,
			p.stopESR(ctx, job.ID, c.epoch), nam, ckpt, sha256Hex(nam))
		if err != nil && ctx.Err() == nil {
			// Not fatal and not retried here: the next epoch is another attempt, and a run that
			// cannot reach the library has bigger news than a missed checkpoint.
			p.log.Printf("job %d: keep epoch %d: %v", job.ID, c.epoch+1, err)
		}
		return
	}
}
