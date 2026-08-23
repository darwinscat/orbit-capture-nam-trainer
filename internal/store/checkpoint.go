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
//
// AND THE FENCE IS TAKEN UNDER A LOCK, which is not decoration. ON CONFLICT DO UPDATE, on finding its
// conflicting row deleted by a transaction that committed while it waited, RETRIES THE INSERT — and
// it does not re-evaluate the SELECT while doing so, so the fence is re-used from a snapshot in which
// the job was still running. The job turning terminal is exactly what deletes that row (the trigger
// in migration 0006), and a run's last write races that transition on EVERY natural finish: the row
// would come back, on a job that is over, and stay for ever, because the trigger only fires on a
// change of state and there will not be another one. FOR UPDATE makes the fence read the newest
// version of the jobs row under its lock, so the terminal transaction cannot slip inside the window.
// The lock order here — jobs, then job_checkpoint — is the order the terminal write takes as well.
func (s *Store) PutCheckpoint(ctx context.Context, id int64, token string, reached int,
	esr *float64, nam, ckpt []byte, namSHA string) error {
	_, err := s.pool.Exec(ctx,
		`INSERT INTO job_checkpoint (job_id, claim_token, reached, esr, nam_sha256, nam_size, nam, ckpt_size, ckpt)
		 SELECT j.id, j.claim_token, $3, $4, $5, $6, $7, $8, $9
		   FROM jobs j WHERE j.id = $1 AND j.claim_token = $2::uuid AND j.state = 'running' FOR UPDATE
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

// DropCheckpointNoNewerThan removes a job's running checkpoint, but ONLY if it has not moved past
// the epoch the caller resumed from.
//
// The schema drops the row by trigger the moment a job turns terminal (migration 0006) — one rule, in
// one place, that no writer can forget. This covers the case that is NOT a state change: an attempt
// that resumed from a checkpoint and died before its first epoch, whose failure the fence then
// refuses to record because the row had already been requeued out from under it. The job sits at
// 'queued', the trigger never fires, and a checkpoint the driver cannot open would swallow every
// later attempt in the same silence.
//
// AND THE GUARD IS THE EPOCH, NOT THE CLAIM. A token cannot answer this: the row being deleted was
// written by an EARLIER attempt — that is what "resumed from" means — so the caller never holds its
// token. What the caller can say is "the weights I could not use were at N", and a row that has since
// reached further belongs to a successor who is training right now. Deleting that is how a daemon
// takes a live two-hundred-epoch run down with it, and this is the guard against it. A successor that
// resumed from the same N and has not written yet does lose the row — and re-creates it one epoch
// later, which is the whole cost.
func (s *Store) DropCheckpointNoNewerThan(ctx context.Context, id int64, reached int) error {
	if _, err := s.pool.Exec(ctx,
		`DELETE FROM job_checkpoint WHERE job_id = $1 AND reached <= $2`, id, reached); err != nil {
		return fmt.Errorf("drop checkpoint for job %d: %w", id, err)
	}
	return nil
}
