// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"database/sql"
	"fmt"

	"orbit-capture-nam-trainer/internal/jobs"
)

// QueueEntry is one job's view for POST /v1/queue: the full row plus its lane
// scheduling numbers. Position is nil for a running/terminal job; EpochsAhead is
// 0 for a running job and nil for a terminal one. Found is false only for an
// unknown key — a job whose MODEL was GC'd keeps its row, so it is still found:true
// (with has_model:false).
type QueueEntry struct {
	Job         jobs.Job
	Found       bool
	Position    *int
	EpochsAhead *int64
}

// QueueView returns, for each requested key, the job's status plus its lane
// position and epochs_ahead. Positions and epochs_ahead are derived from ONE
// queue snapshot, so they can never mutually contradict (no A-ahead-of-B AND
// B-ahead-of-A) under a concurrent claim/delete/patch. Per-key status is a point
// read branched on the job's own current state, so a job claimed between the
// snapshot and its lookup is reported running (position dropped), not as a queued
// job with a stale position.
//
// epochs_ahead is the lane epochs ahead of this job: the remaining epochs of every
// RUNNING job in the lane plus the full epochs of every QUEUED job ahead of it
// (same priority/created_at/key order the scheduler claims by). It spans EVERY
// caller's jobs, not just the requested keys. It is a serial-drain sum — exact
// wall-work at cap=1 (the default); at cap>1 the client divides by cap for an ETA,
// an estimate, since an uneven running remainder frees a worker sooner than the sum
// implies.
func (s *Store) QueueView(ctx context.Context, keys []string) (map[string]QueueEntry, error) {
	pos, ahead, err := s.scheduleSnapshot(ctx)
	if err != nil {
		return nil, err
	}

	out := make(map[string]QueueEntry, len(keys))
	for _, key := range keys {
		if _, done := out[key]; done {
			continue // a repeated key is one entry
		}
		j, found, err := s.GetJob(ctx, key)
		if err != nil {
			return nil, fmt.Errorf("queue view %s: %w", key, err)
		}
		if !found {
			out[key] = QueueEntry{Found: false}
			continue
		}
		e := QueueEntry{Job: j, Found: true}
		switch j.State {
		case jobs.StateRunning:
			zero := int64(0)
			e.EpochsAhead = &zero // running: no position, nothing ahead of it
		case jobs.StateQueued:
			if p, ok := pos[key]; ok {
				pp := p
				a := ahead[key]
				e.Position = &pp
				e.EpochsAhead = &a
			}
		} // terminal: both nil
		out[key] = e
	}
	return out, nil
}

// scheduleSnapshot reads all queued+running rows once and computes, per lane, each
// queued job's 1-based position and epochs_ahead. Lanes (train / probe_self /
// probe_e10) are scoped separately because they drain concurrently — a train
// job's ETA must not count probe epochs; train_more shares the train lane
// (jobs.Lane). Each job's remaining work is epochs − COALESCE(start_epoch,0), so a
// resumed train_more only ever counts the epochs it will actually compute. The
// queue is tens of rows; one indexed pass is cheap.
func (s *Store) scheduleSnapshot(ctx context.Context) (pos map[string]int, ahead map[string]int64, err error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT key, kind, state, epochs, epoch, start_epoch FROM jobs
		 WHERE state IN ('queued','running')
		 ORDER BY priority, created_at, key`)
	if err != nil {
		return nil, nil, fmt.Errorf("schedule snapshot: %w", err)
	}
	defer rows.Close()

	type qjob struct {
		key       string
		remaining int64
	}
	runSum := map[string]int64{}  // lane -> Σ remaining epochs of running jobs
	queued := map[string][]qjob{} // lane -> queued jobs, already in claim order
	for rows.Next() {
		var (
			key, kind, state  string
			epochs            int64
			epoch, startEpoch sql.NullInt64
		)
		if err := rows.Scan(&key, &kind, &state, &epochs, &epoch, &startEpoch); err != nil {
			return nil, nil, fmt.Errorf("schedule scan: %w", err)
		}
		lane := jobs.Lane(kind)
		rem := remainingEpochs(epochs, epoch, startEpoch)
		if state == jobs.StateRunning {
			runSum[lane] += rem
		} else {
			queued[lane] = append(queued[lane], qjob{key, rem})
		}
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("schedule rows: %w", err)
	}

	pos = map[string]int{}
	ahead = map[string]int64{}
	for lane, list := range queued {
		cum := runSum[lane] // every running job in the lane is ahead of all queued ones
		for i, q := range list {
			pos[q.key] = i + 1
			ahead[q.key] = cum
			cum += q.remaining
		}
	}
	return pos, ahead, nil
}

// QueueTotals returns the live queue totals the macOS menu-bar tray displays: the
// running/queued job counts and, per lane, the total remaining epochs (the
// remainder of every running job plus the remaining epochs of every queued one) —
// the same serial-drain arithmetic as scheduleSnapshot, summed to the end of the
// lane instead of per job. The map is keyed by lane (jobs.Lane), so train and
// train_more accumulate together. One indexed pass over tens of rows.
func (s *Store) QueueTotals(ctx context.Context) (running, queued int, remaining map[string]int64, err error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT kind, state, epochs, epoch, start_epoch FROM jobs WHERE state IN ('queued','running')`)
	if err != nil {
		return 0, 0, nil, fmt.Errorf("queue totals: %w", err)
	}
	defer rows.Close()

	remaining = map[string]int64{}
	for rows.Next() {
		var (
			kind, state       string
			epochs            int64
			epoch, startEpoch sql.NullInt64
		)
		if err := rows.Scan(&kind, &state, &epochs, &epoch, &startEpoch); err != nil {
			return 0, 0, nil, fmt.Errorf("queue totals scan: %w", err)
		}
		if state == jobs.StateRunning {
			running++
		} else {
			queued++
		}
		remaining[jobs.Lane(kind)] += remainingEpochs(epochs, epoch, startEpoch)
	}
	if err := rows.Err(); err != nil {
		return 0, 0, nil, fmt.Errorf("queue totals rows: %w", err)
	}
	return running, queued, remaining, nil
}

// QueueRow is one line of the menu-bar queue list: a running or queued job in
// display order (running first, then queued in claim order). Epoch is the last
// reported 0-based epoch of a running job, nil until it prints one (or for a
// queued job).
type QueueRow struct {
	Key     string
	Kind    string
	Running bool
	Epochs  int64
	Epoch   *int64
}

// QueueRows returns up to limit rows for the menu-bar queue list: running jobs
// first, then queued ones in the exact order the scheduler will claim them
// (priority, created_at, key). Lanes are interleaved here — the list answers
// "what is the daemon doing / about to do", not per-lane ETA arithmetic.
func (s *Store) QueueRows(ctx context.Context, limit int) ([]QueueRow, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT key, kind, state, epochs, epoch FROM jobs
		 WHERE state IN ('queued','running')
		 ORDER BY (state='running') DESC, priority, created_at, key
		 LIMIT ?`, limit)
	if err != nil {
		return nil, fmt.Errorf("queue rows: %w", err)
	}
	defer rows.Close()

	var out []QueueRow
	for rows.Next() {
		var (
			r     QueueRow
			state string
			epoch sql.NullInt64
		)
		if err := rows.Scan(&r.Key, &r.Kind, &state, &r.Epochs, &epoch); err != nil {
			return nil, fmt.Errorf("queue rows scan: %w", err)
		}
		r.Running = state == jobs.StateRunning
		if epoch.Valid {
			e := epoch.Int64
			r.Epoch = &e
		}
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("queue rows err: %w", err)
	}
	return out, nil
}

// remainingEpochs is how many epochs a job still has to compute. With no live epoch
// reported yet (a QUEUED job, or a just-claimed one silent for minutes during torch
// import) it is epochs − COALESCE(start_epoch,0): a plain job runs the full count,
// a resumed train_more only the epochs past its parent's. A naive SUM(epochs-epoch)
// would wrongly drop an epoch-NULL running job to zero, hence the explicit branch.
// Once an epoch is reported it is ABSOLUTE (Lightning keeps numbering across a
// resume), so the remainder is epochs-(epoch+1) with no start_epoch term, clamped
// at 0. The +1 matches the 0-based epoch and the eta_s convention in the HTTP layer.
func remainingEpochs(epochs int64, epoch, startEpoch sql.NullInt64) int64 {
	var done int64
	switch {
	case epoch.Valid:
		done = epoch.Int64 + 1
	case startEpoch.Valid:
		done = startEpoch.Int64
	}
	if r := epochs - done; r > 0 {
		return r
	}
	return 0
}
