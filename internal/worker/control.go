// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"context"
	"errors"
	"time"

	"orbit-capture-nam-trainer/internal/jobs"
)

// controlLoop answers the app's commands on a RUNNING row, polled every
// controlPoll until the child's output closes (done) or the pool stops:
//
//   - the row is no longer ours (cancelled by the app after a stale heartbeat, or
//     reclaimed) → kill the child with reasonLost; nothing is written;
//   - cancel_requested_at → kill the group (reasonCancel); classify writes
//     state='cancelled'. Cancel beats stop when both are seen on one poll;
//   - stop_requested_at → on a train-lane job: stop_state='pending' (acknowledged,
//     no whole checkpoint pair yet) until selectLastCkpt finds one, then
//     stop_state='armed' and the kill (reasonStop) — classify harvests the last
//     pair and finishes succeeded (stop_state='done') or stop_failed ('refused').
//     A probe is never stopped: it is acknowledged 'refused' and runs to its
//     verdict (seconds);
//   - live_requested_at newer than live_served_at → ExportLive; a snapshot lands in
//     job_snapshots with live_served_at = now(), or live_served_at + live_error
//     (no_checkpoint | transient) when there is nothing to serve yet.
//
// Each ack is written ONCE per transition (in-memory flags); the reads are fenced
// on the claim token like every other write.
func (p *Pool) controlLoop(job jobs.Job, entry *procEntry, done <-chan struct{}) {
	t := time.NewTicker(p.controlPoll)
	defer t.Stop()
	var stopAcked, stopArmed bool
	for {
		select {
		case <-done:
			return
		case <-p.ctx.Done():
			return
		case <-t.C:
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		c, ok, err := p.store.Control(ctx, job.ID, job.ClaimToken)
		cancel()
		if err != nil {
			p.log.Printf("job %d: read control flags: %v", job.ID, err)
			continue
		}
		if !ok {
			entry.kill(reasonLost)
			return
		}
		if c.CancelRequestedAt != nil {
			p.log.Printf("job %d: cancel requested — killing", job.ID)
			entry.kill(reasonCancel)
			return
		}
		if c.StopRequestedAt != nil && !stopArmed {
			switch {
			case job.Lane != jobs.LaneTrain:
				if !stopAcked {
					p.log.Printf("job %d: stop requested on a probe — refused, it runs to its verdict", job.ID)
					p.setStopState(job, jobs.StopRefused)
					stopAcked = true
				}
			case len(selectLastCkpt(entry.scratch)) > 0:
				p.log.Printf("job %d: stop requested — a checkpoint pair exists, stopping", job.ID)
				p.setStopState(job, jobs.StopArmed)
				stopArmed = true
				entry.kill(reasonStop)
			case !stopAcked:
				p.log.Printf("job %d: stop requested — no checkpoint yet, will stop at the first one", job.ID)
				p.setStopState(job, jobs.StopPending)
				stopAcked = true
			}
		}
		if c.LiveRequestedAt != nil && (c.LiveServedAt == nil || c.LiveRequestedAt.After(*c.LiveServedAt)) {
			p.serveLive(job, entry, *c.LiveRequestedAt)
		}
	}
}

func (p *Pool) setStopState(job jobs.Job, state string) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := p.store.SetStopState(ctx, job.ID, job.ClaimToken, state); err != nil {
		p.log.Printf("job %d: stop_state=%s: %v", job.ID, state, err)
	}
}

// serveLive answers one live request: a job_snapshots row with the best-so-far
// checkpoint's .nam, or the reason there is none.
func (p *Pool) serveLive(job jobs.Job, entry *procEntry, requestedAt time.Time) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	nam, epoch, esr, err := p.exportLiveEntry(ctx, entry, job.ID)
	switch {
	case err == nil:
		if err := p.store.ServeLive(ctx, job.ID, job.ClaimToken, requestedAt, epoch, esr, nam); err != nil {
			p.log.Printf("job %d: serve live snapshot: %v", job.ID, err)
			return
		}
		p.log.Printf("job %d: live snapshot served (epoch %d, esr %.6g)", job.ID, epoch, esr)
	case errors.Is(err, ErrNoCheckpoint):
		if err := p.store.LiveServed(ctx, job.ID, job.ClaimToken, jobs.LiveNoCheckpoint); err != nil {
			p.log.Printf("job %d: live served (no_checkpoint): %v", job.ID, err)
		}
	default:
		if err := p.store.LiveServed(ctx, job.ID, job.ClaimToken, jobs.LiveTransient); err != nil {
			p.log.Printf("job %d: live served (transient): %v", job.ID, err)
		}
	}
}
