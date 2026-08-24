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
// Verdict is what the library says back when a run writes an epoch: everything the run needs to know
// about whether to carry on, in the write it was making anyway.
//
// THIS IS THE ONLY CHANNEL. There used to be a second — a poll of the row's control flags every two
// seconds — and two channels answering the same question is two things to keep in agreement. The
// price is latency: a cancel now arrives at the end of the epoch in progress rather than within two
// seconds. On a workshop whose epochs are ten to thirty seconds that is a fair trade for one
// mechanism instead of two.
type Verdict struct {
	Mine   bool // the fence passed — this run is still ours to continue
	Stop   bool // …but somebody asked it to stop and keep what it has
	Cancel bool // …or to throw it away
}

func (s *Store) PutCheckpoint(ctx context.Context, jobID int64, token string, takeID int64,
	last Pair, best *Pair) (Verdict, error) {
	var bestReached *int
	var bestESR *float64
	var bestSHA *string
	var bestNam, bestCkpt []byte
	if best != nil {
		bestReached, bestESR, bestSHA = &best.Reached, best.ESR, &best.NamSHA
		bestNam, bestCkpt = best.Nam, best.Ckpt
	}
	var v Verdict
	err := s.pool.QueryRow(ctx,
		`WITH mine AS (
		     SELECT j.id, j.claim_token,
		            j.stop_requested_at   IS NOT NULL AS stop,
		            j.cancel_requested_at IS NOT NULL AS cancel
		       FROM jobs j
		      WHERE j.id = $1 AND j.claim_token = $2::uuid AND j.state = 'running'
		        AND j.paused_at IS NULL
		        FOR UPDATE),
		 kept AS (
		     INSERT INTO take_checkpoint (take_id, job_id, claim_token, reached, esr,
		            nam_sha256, nam_size, nam, ckpt_size, ckpt,
		            best_reached, best_esr, best_nam_sha256, best_nam_size, best_nam, best_ckpt_size, best_ckpt)
		     SELECT $3, m.id, m.claim_token, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
		       FROM mine m
		     ON CONFLICT (take_id) DO UPDATE SET
		       job_id = EXCLUDED.job_id, claim_token = EXCLUDED.claim_token, reached = EXCLUDED.reached,
		       esr = EXCLUDED.esr, nam_sha256 = EXCLUDED.nam_sha256, nam_size = EXCLUDED.nam_size,
		       nam = EXCLUDED.nam, ckpt_size = EXCLUDED.ckpt_size, ckpt = EXCLUDED.ckpt,
		       -- THE BEST PAIR IS THE BEST ONE THIS TAKE HAS SEEN, whoever saw it. It used to be
		       -- assigned from EXCLUDED like the rest, which meant every writer that did not bring one
		       -- ERASED it: a run that finished naturally wrote its final pair with no best beside it
		       -- and left best_* NULL, so the app's COALESCE(best, last) quietly served the last epoch
		       -- as the best — on a run whose ESR swings an order of magnitude between epochs, that is
		       -- the whole point of keeping two pairs, thrown away at the moment the run succeeded.
		       -- Measured on a live library: two takes finished at 400 epochs had no best pair at all,
		       -- while a take stopped short kept one ten times better than its last epoch.
		       --
		       -- Whoever offers a lower ESR wins; nobody's silence takes anything away. The same rule
		       -- covers a handover (the new attempt only displaces the old attempt's best by beating
		       -- it) and the saver's own moment of catching a checkpoint mid-rotation.
		       best_reached = CASE WHEN $18::real IS NOT NULL
		                            AND (take_checkpoint.best_esr IS NULL OR $18::real < take_checkpoint.best_esr)
		                           THEN EXCLUDED.best_reached ELSE take_checkpoint.best_reached END,
		       best_esr = CASE WHEN $18::real IS NOT NULL
		                        AND (take_checkpoint.best_esr IS NULL OR $18::real < take_checkpoint.best_esr)
		                       THEN EXCLUDED.best_esr ELSE take_checkpoint.best_esr END,
		       best_nam_sha256 = CASE WHEN $18::real IS NOT NULL
		                               AND (take_checkpoint.best_esr IS NULL OR $18::real < take_checkpoint.best_esr)
		                              THEN EXCLUDED.best_nam_sha256 ELSE take_checkpoint.best_nam_sha256 END,
		       best_nam_size = CASE WHEN $18::real IS NOT NULL
		                             AND (take_checkpoint.best_esr IS NULL OR $18::real < take_checkpoint.best_esr)
		                            THEN EXCLUDED.best_nam_size ELSE take_checkpoint.best_nam_size END,
		       best_nam = CASE WHEN $18::real IS NOT NULL
		                        AND (take_checkpoint.best_esr IS NULL OR $18::real < take_checkpoint.best_esr)
		                       THEN EXCLUDED.best_nam ELSE take_checkpoint.best_nam END,
		       best_ckpt_size = CASE WHEN $18::real IS NOT NULL
		                              AND (take_checkpoint.best_esr IS NULL OR $18::real < take_checkpoint.best_esr)
		                             THEN EXCLUDED.best_ckpt_size ELSE take_checkpoint.best_ckpt_size END,
		       best_ckpt = CASE WHEN $18::real IS NOT NULL
		                         AND (take_checkpoint.best_esr IS NULL OR $18::real < take_checkpoint.best_esr)
		                        THEN EXCLUDED.best_ckpt ELSE take_checkpoint.best_ckpt END,
		       at = now()
		     RETURNING take_id)
		 SELECT EXISTS (SELECT 1 FROM kept),
		        COALESCE((SELECT stop   FROM mine), false),
		        COALESCE((SELECT cancel FROM mine), false)`,
		jobID, token, takeID, last.Reached, last.ESR, last.NamSHA, len(last.Nam), last.Nam,
		len(last.Ckpt), last.Ckpt,
		bestReached, bestESR, bestSHA, sizeOrNil(bestNam), bestNam, sizeOrNil(bestCkpt), bestCkpt,
		bestESR). // $18: the offered best ESR, read once by the CASEs above
		Scan(&v.Mine, &v.Stop, &v.Cancel)
	if err != nil {
		return Verdict{}, fmt.Errorf("put checkpoint for take %d: %w", takeID, err)
	}
	return v, nil
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

// RunVerdict asks the same question PutCheckpoint answers, without writing anything: is this run
// still ours, and has anybody asked it to stop or to be thrown away.
//
// IT IS THE SAME CHANNEL, ASKED ON A TIMER. The write answers it for free once an epoch, and that is
// how it is normally heard — but a run that has stopped producing epochs produces no writes either,
// and a person who wants a hung trainer stopped should not have to wait for the stall watchdog.
// One indexed row, once every half minute per running job.
func (s *Store) RunVerdict(ctx context.Context, jobID int64, token string) (Verdict, error) {
	var v Verdict
	err := s.pool.QueryRow(ctx,
		`SELECT true, stop_requested_at IS NOT NULL, cancel_requested_at IS NOT NULL
		   FROM jobs
		  WHERE id = $1 AND claim_token = $2::uuid AND state = 'running' AND paused_at IS NULL`,
		jobID, token).Scan(&v.Mine, &v.Stop, &v.Cancel)
	if errors.Is(err, pgx.ErrNoRows) {
		return Verdict{}, nil // not ours any more, which is an answer and not an error
	}
	if err != nil {
		return Verdict{}, fmt.Errorf("read the verdict for job %d: %w", jobID, err)
	}
	return v, nil
}
