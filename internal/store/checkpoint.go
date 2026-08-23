// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

// RunningCheckpoint is what the library holds for a job that is TRAINING RIGHT NOW: the newest
// intact checkpoint pair and the epoch count it carries. See migration 0006 — the point is that a
// crash, a kill, a reboot or a machine that never came back cost one unfinished epoch instead of the
// whole run, and that the next trainer to claim the row can be a different machine.
type RunningCheckpoint struct {
	Reached int
	ESR     *float64
	Nam     []byte
	Ckpt    []byte
}

// PutCheckpoint stores this attempt's newest completed epoch, replacing whatever was there.
//
// FENCED, AND MONOTONIC. The insert draws its own values through a SELECT over jobs, so a claim that
// has already been taken away writes nothing at all rather than writing something that looks current;
// and the update refuses to go backwards, so a straggler cannot replace two hundred epochs with two.
func (s *Store) PutCheckpoint(ctx context.Context, id int64, token string, reached int,
	esr *float64, nam, ckpt []byte, namSHA string) error {
	_, err := s.pool.Exec(ctx,
		`INSERT INTO job_checkpoint (job_id, claim_token, reached, esr, nam_sha256, nam_size, nam, ckpt_size, ckpt)
		 SELECT j.id, j.claim_token, $3, $4, $5, $6, $7, $8, $9
		   FROM jobs j WHERE j.id = $1 AND j.claim_token = $2::uuid AND j.state = 'running'
		 ON CONFLICT (job_id) DO UPDATE SET
		   claim_token = EXCLUDED.claim_token, reached = EXCLUDED.reached, esr = EXCLUDED.esr,
		   nam_sha256 = EXCLUDED.nam_sha256, nam_size = EXCLUDED.nam_size, nam = EXCLUDED.nam,
		   ckpt_size = EXCLUDED.ckpt_size, ckpt = EXCLUDED.ckpt, at = now()
		 WHERE EXCLUDED.reached > job_checkpoint.reached`,
		id, token, reached, esr, namSHA, len(nam), nam, len(ckpt), ckpt)
	if err != nil {
		return fmt.Errorf("put checkpoint for job %d: %w", id, err)
	}
	return nil
}

// Checkpoint reads what a job has trained so far, whoever trained it. Read at CLAIM time, before the
// child is spawned: a row here means this is not a fresh run, it is a run being picked up.
func (s *Store) Checkpoint(ctx context.Context, id int64) (RunningCheckpoint, bool, error) {
	var c RunningCheckpoint
	err := s.pool.QueryRow(ctx,
		`SELECT reached, esr, nam, ckpt FROM job_checkpoint WHERE job_id = $1`, id).
		Scan(&c.Reached, &c.ESR, &c.Nam, &c.Ckpt)
	if errors.Is(err, pgx.ErrNoRows) {
		return RunningCheckpoint{}, false, nil
	}
	if err != nil {
		return RunningCheckpoint{}, false, fmt.Errorf("read checkpoint for job %d: %w", id, err)
	}
	return c, true, nil
}

// DropCheckpoint removes a job's running checkpoint.
//
// The schema drops it by trigger the moment a job turns terminal, which is the rule that matters and
// the one no writer can forget. This exists for the one case that is NOT a state change: a resume
// that could not be used — a checkpoint the driver refused — where leaving the row would make every
// later attempt refuse in the same way, for ever.
func (s *Store) DropCheckpoint(ctx context.Context, id int64) error {
	if _, err := s.pool.Exec(ctx, `DELETE FROM job_checkpoint WHERE job_id = $1`, id); err != nil {
		return fmt.Errorf("drop checkpoint for job %d: %w", id, err)
	}
	return nil
}
