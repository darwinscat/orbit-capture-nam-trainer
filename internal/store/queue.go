// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"fmt"
)

// MyRun is one job THIS trainer is running: what the menu lists, and the numbers
// the title and the estimate are made of. Epoch is the last reported 0-based epoch,
// nil until the run prints one.
type MyRun struct {
	Label     string // take_label — the handle people use
	Kind      string // train | train_more | probe_self
	Lane      string // jobs.LaneTrain / jobs.LaneProbe (the schema's generated column)
	Epochs    int64
	Epoch     *int64
	Remaining int64 // epochs still to compute
}

// MyRuns returns what this worker is running, oldest claim first.
//
// The menu used to read the whole shared queue and present it as this machine's:
// counts, list and estimate all came from `jobs` with no worker filter. While there
// was one trainer, "the queue" and "mine" were the same rows; with two, the menu on
// one box listed the other's work and timed the pile at this box's speed. A menu bar
// answers one question — what is THIS machine doing — and the app is where the shared
// queue is looked at, with every worker's name on it.
func (s *Store) MyRuns(ctx context.Context, me string) ([]MyRun, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT take_label, kind, lane, epochs, epoch, start_epoch FROM jobs
		 WHERE state = 'running' AND worker = $1
		 ORDER BY claimed_at, id`, me)
	if err != nil {
		return nil, fmt.Errorf("my runs: %w", err)
	}
	defer rows.Close()

	var out []MyRun
	for rows.Next() {
		var (
			r          MyRun
			startEpoch *int64
		)
		if err := rows.Scan(&r.Label, &r.Kind, &r.Lane, &r.Epochs, &r.Epoch, &startEpoch); err != nil {
			return nil, fmt.Errorf("my runs scan: %w", err)
		}
		r.Remaining = remainingEpochs(r.Epochs, r.Epoch, startEpoch)
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("my runs rows: %w", err)
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
