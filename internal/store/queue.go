// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"fmt"

	"orbit-capture-nam-trainer/internal/jobs"
)

// QueueTotals returns what the macOS menu-bar tray displays: the running/queued
// counts over the WHOLE shared queue — the list under them is that queue too —
// and `mine`, the epochs still to compute in each job THIS trainer is running.
//
// The counts are everybody's and the estimate is not, on purpose. How long the
// whole queue takes is not knowable from one box: the other trainers' speed is
// not this one's, and which of them claims the next row is decided when it is
// claimed. What this box can honestly say is when IT is done with what it holds,
// so only its own running train-lane jobs are measured — queued work is somebody
// else's until it is claimed. (Before the queue became shared, "the queue" and
// "mine" were the same rows, and the title quietly promised the whole thing at
// this box's cap: with two trainers at cap 1 it read about twice the truth.)
func (s *Store) QueueTotals(ctx context.Context, me string) (running, queued int, mine []int64, err error) {
	rows, err := s.pool.Query(ctx,
		`SELECT lane, state, epochs, epoch, start_epoch, worker FROM jobs WHERE state IN ('queued', 'running')`)
	if err != nil {
		return 0, 0, nil, fmt.Errorf("queue totals: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var (
			lane, state       string
			epochs            int64
			epoch, startEpoch *int64
			worker            *string
		)
		if err := rows.Scan(&lane, &state, &epochs, &epoch, &startEpoch, &worker); err != nil {
			return 0, 0, nil, fmt.Errorf("queue totals scan: %w", err)
		}
		if state != jobs.StateRunning {
			queued++
			continue
		}
		running++
		if lane == jobs.LaneTrain && worker != nil && *worker == me {
			mine = append(mine, remainingEpochs(epochs, epoch, startEpoch))
		}
	}
	if err := rows.Err(); err != nil {
		return 0, 0, nil, fmt.Errorf("queue totals rows: %w", err)
	}
	return running, queued, mine, nil
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
	Worker  string // who is running it (empty while queued — nobody holds it yet)
}

// QueueRows returns up to limit rows for the menu-bar queue list: running jobs
// first, then queued ones in the exact order the scheduler claims them (priority,
// queued_at, id). Lanes are interleaved — the list answers "what is the queue
// doing / about to do", not per-lane ETA arithmetic.
func (s *Store) QueueRows(ctx context.Context, limit int) ([]QueueRow, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT id, take_label, kind, state, epochs, epoch, worker FROM jobs
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
			r      QueueRow
			state  string
			worker *string
		)
		if err := rows.Scan(&r.ID, &r.Label, &r.Kind, &state, &r.Epochs, &r.Epoch, &worker); err != nil {
			return nil, fmt.Errorf("queue rows scan: %w", err)
		}
		r.Running = state == jobs.StateRunning
		if worker != nil {
			r.Worker = *worker
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
