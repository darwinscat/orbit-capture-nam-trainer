// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

// TakeCheckpoint is what the library holds for a take: the newest weights it has, whoever trained
// them and whether or not a run is going right now. See migration 0007 — one row per TAKE, because
// "what are this take's newest weights" is a question about the take, and the answer has to survive
// the run that produced it. A finished run is a pause; a failed one keeps what it trained; only a
// cancel discards, and even then the bytes are moved aside rather than dropped.
type TakeCheckpoint struct {
	Reached int      // the last completed epoch — what a continuation resumes from
	ESR     *float64 // that epoch's validation figure, if the driver named one
	Nam     []byte
	Ckpt    []byte
}

// Pair is one checkpoint on disk: the trainer writes a .ckpt and its .nam sibling every epoch.
type Pair struct {
	Reached int
	ESR     *float64
	Nam     []byte
	Ckpt    []byte
	NamSHA  string
}

// PutCheckpoint stores this run's newest epoch as the take's weights, and says whether it may go on.
//
// THE WRITE IS ALSO THE OWNERSHIP CHECK, and that is the whole design: the fence names the job, the
// claim and `paused_at IS NULL`, so a run that has been handed to another machine — or marked
// claimable because its task went silent — writes nothing and learns it in the same breath. ok=false
// means exactly one thing, "this run is no longer mine to continue", and the answer to it is to stop.
// There is no monotonic guard: a repeat of the same epoch is a harmless overwrite, and a guard that
// refused it made ok=false ambiguous, which cost the caller the only signal it has.
//
// FOR UPDATE is not decoration. ON CONFLICT DO UPDATE, on finding its conflicting row deleted by a
// transaction that committed while it waited, RETRIES THE INSERT without re-evaluating the SELECT —
// so without the lock a write racing a cancel could put the weights back after the cancel discarded
// them, on a take whose run is over, for ever.
//
// last is required; best may be empty until the trainer has named one.
func (s *Store) PutCheckpoint(ctx context.Context, jobID int64, token string, takeID int64,
	last Pair, best *Pair) (ok bool, err error) {
	var bestReached *int
	var bestESR *float64
	var bestSHA *string
	var bestNam, bestCkpt []byte
	if best != nil {
		bestReached, bestESR, bestSHA = &best.Reached, best.ESR, &best.NamSHA
		bestNam, bestCkpt = best.Nam, best.Ckpt
	}
	tag, err := s.pool.Exec(ctx,
		`INSERT INTO take_checkpoint (take_id, job_id, claim_token, reached, esr,
		        nam_sha256, nam_size, nam, ckpt_size, ckpt,
		        best_reached, best_esr, best_nam_sha256, best_nam_size, best_nam, best_ckpt_size, best_ckpt)
		 SELECT $3, j.id, j.claim_token, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
		   FROM jobs j
		  WHERE j.id = $1 AND j.claim_token = $2::uuid AND j.state = 'running' AND j.paused_at IS NULL
		    FOR UPDATE
		 ON CONFLICT (take_id) DO UPDATE SET
		   job_id = EXCLUDED.job_id, claim_token = EXCLUDED.claim_token, reached = EXCLUDED.reached,
		   esr = EXCLUDED.esr, nam_sha256 = EXCLUDED.nam_sha256, nam_size = EXCLUDED.nam_size,
		   nam = EXCLUDED.nam, ckpt_size = EXCLUDED.ckpt_size, ckpt = EXCLUDED.ckpt,
		   best_reached = EXCLUDED.best_reached, best_esr = EXCLUDED.best_esr,
		   best_nam_sha256 = EXCLUDED.best_nam_sha256, best_nam_size = EXCLUDED.best_nam_size,
		   best_nam = EXCLUDED.best_nam, best_ckpt_size = EXCLUDED.best_ckpt_size,
		   best_ckpt = EXCLUDED.best_ckpt, at = now()`,
		jobID, token, takeID, last.Reached, last.ESR, last.NamSHA, len(last.Nam), last.Nam,
		len(last.Ckpt), last.Ckpt,
		bestReached, bestESR, bestSHA, sizeOrNil(bestNam), bestNam, sizeOrNil(bestCkpt), bestCkpt)
	if err != nil {
		return false, fmt.Errorf("put checkpoint for take %d: %w", takeID, err)
	}
	return tag.RowsAffected() > 0, nil
}

func sizeOrNil(b []byte) *int {
	if b == nil {
		return nil
	}
	n := len(b)
	return &n
}

// Checkpoint reads a take's newest weights, whoever trained them. Read at CLAIM time, before the
// child is spawned: a row here means this is not a fresh start — the take has been trained this far
// already, by this run, by an earlier one, or by another machine entirely.
func (s *Store) Checkpoint(ctx context.Context, takeID int64) (TakeCheckpoint, bool, error) {
	var c TakeCheckpoint
	err := s.pool.QueryRow(ctx,
		`SELECT reached, esr, nam, ckpt FROM take_checkpoint WHERE take_id = $1`, takeID).
		Scan(&c.Reached, &c.ESR, &c.Nam, &c.Ckpt)
	if errors.Is(err, pgx.ErrNoRows) {
		return TakeCheckpoint{}, false, nil
	}
	if err != nil {
		return TakeCheckpoint{}, false, fmt.Errorf("read checkpoint for take %d: %w", takeID, err)
	}
	return c, true, nil
}

// EpochESR is the validation figure the trainer printed for one epoch of one run, or nil when it
// printed none. One indexed row: job_epochs is keyed by (job_id, epoch), written as the epochs land.
//
// The saver asks this about the epoch it is ABOUT TO STORE, which is not always the epoch that just
// finished — a torn newest checkpoint sends it one back, and a run with no `last` pair at all sends
// it to the best one. Carrying the reader's figure along would then attach one epoch's ESR to
// another's weights. (It used to read the whole job_log to find the line, once per epoch, over the
// network, on a log that grows all run — this is the same answer without that.)
func (s *Store) EpochESR(ctx context.Context, jobID int64, epoch int) *float64 {
	var esr *float64
	if err := s.pool.QueryRow(ctx,
		`SELECT esr FROM job_epochs WHERE job_id = $1 AND epoch = $2`, jobID, epoch).Scan(&esr); err != nil {
		return nil
	}
	return esr
}
