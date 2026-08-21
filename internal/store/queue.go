// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"fmt"

	"orbit-capture-nam-trainer/internal/jobs"
)

// QueueTotals returns the live queue totals the macOS menu-bar tray displays: the
// running/queued job counts and, per lane, the total remaining epochs (the
// remainder of every running job plus the remaining epochs of every queued one).
// Whole-queue numbers (every worker's), keyed by lane (jobs.LaneTrain /
// jobs.LaneProbe), so train and train_more accumulate together. Position/ETA per
// job is the app's business now (a window function on its side).
func (s *Store) QueueTotals(ctx context.Context) (running, queued int, remaining map[string]int64, err error) {
	rows, err := s.pool.Query(ctx,
		`SELECT lane, state, epochs, epoch, start_epoch FROM jobs WHERE state IN ('queued', 'running')`)
	if err != nil {
		return 0, 0, nil, fmt.Errorf("queue totals: %w", err)
	}
	defer rows.Close()

	remaining = map[string]int64{}
	for rows.Next() {
		var (
			lane, state       string
			epochs            int64
			epoch, startEpoch *int64
		)
		if err := rows.Scan(&lane, &state, &epochs, &epoch, &startEpoch); err != nil {
			return 0, 0, nil, fmt.Errorf("queue totals scan: %w", err)
		}
		if state == jobs.StateRunning {
			running++
		} else {
			queued++
		}
		remaining[lane] += remainingEpochs(epochs, epoch, startEpoch)
	}
	if err := rows.Err(); err != nil {
		return 0, 0, nil, fmt.Errorf("queue totals rows: %w", err)
	}
	return running, queued, remaining, nil
}

// QueueRow is one line of the menu-bar queue list: a running or queued job in
// display order. Epoch is the last reported 0-based epoch of a running job, nil
// until it prints one (or for a queued job).
type QueueRow struct {
	ID      int64
	Label   string // take_label — the handle people use
	Kind    string
	Running bool
	Epochs  int64
	Epoch   *int64
}

// QueueRows returns up to limit rows for the menu-bar queue list: running jobs
// first, then queued ones in the exact order the scheduler claims them (priority,
// queued_at, id). Lanes are interleaved — the list answers "what is the queue
// doing / about to do", not per-lane ETA arithmetic.
func (s *Store) QueueRows(ctx context.Context, limit int) ([]QueueRow, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT id, take_label, kind, state, epochs, epoch FROM jobs
		 WHERE state IN ('queued', 'running')
		 ORDER BY (state = 'running') DESC, priority, queued_at, id
		 LIMIT $1`, limit)
	if err != nil {
		return nil, fmt.Errorf("queue rows: %w", err)
	}
	defer rows.Close()

	var out []QueueRow
	for rows.Next() {
		var (
			r     QueueRow
			state string
		)
		if err := rows.Scan(&r.ID, &r.Label, &r.Kind, &state, &r.Epochs, &r.Epoch); err != nil {
			return nil, fmt.Errorf("queue rows scan: %w", err)
		}
		r.Running = state == jobs.StateRunning
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
// a resumed train_more only the epochs past its parent's. Once an epoch is reported
// it is ABSOLUTE (Lightning keeps numbering across a resume), so the remainder is
// epochs-(epoch+1), clamped at 0.
func remainingEpochs(epochs int64, epoch, startEpoch *int64) int64 {
	var done int64
	switch {
	case epoch != nil:
		done = *epoch + 1
	case startEpoch != nil:
		done = *startEpoch
	}
	if r := epochs - done; r > 0 {
		return r
	}
	return 0
}
