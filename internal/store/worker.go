// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/jackc/pgx/v5"

	"orbit-capture-nam-trainer/internal/jobs"
)

// Provenance is what a run was made with — stamped on the row at every terminal
// transition (jobs.nam_version / driver_sha256 / signal_sha256).
type Provenance struct {
	NamVersion   string
	DriverSHA256 string
	SignalSHA256 string
}

// fence is the WHERE clause every post-claim write carries: the row must still be
// THIS attempt's (claim_token) and still running. A requeue, a recovery by another
// instance, or an app-side cancel of a stale row all change one of the three, and
// the straggler's write then affects 0 rows.
const fence = ` WHERE id = $1 AND claim_token = $2::uuid AND state = 'running'`

// ClaimNext atomically pops the next queued job of one lane and flips it to
// running, stamping this worker's identity and a fresh claim_token. The pick is
// the schema's drain order (priority enum, queued_at, id) over rows that carry no
// stop/cancel request and pin the nam version this worker runs; FOR UPDATE SKIP
// LOCKED keeps concurrent claimers from ever taking the same row. Live progress
// carried over from a recovered attempt is cleared. ok=false means nothing
// claimable.
func (s *Store) ClaimNext(ctx context.Context, lane, worker, instance, namVersion string) (jobs.Job, bool, error) {
	j, err := scanJob(s.pool.QueryRow(ctx,
		`UPDATE jobs SET state = 'running', worker = $1, worker_instance = $2,
		        claim_token = gen_random_uuid(), claimed_at = now(), started_at = now(),
		        epoch = NULL, s_per_epoch = NULL, pgid = NULL
		 WHERE id = (
		   SELECT id FROM jobs
		   WHERE state = 'queued' AND lane = $3
		     AND cancel_requested_at IS NULL AND stop_requested_at IS NULL
		     AND take_id IS NOT NULL   -- a job whose take was wiped is nobody's; the app closes it
		     AND required_nam_version = $4
		   ORDER BY priority, queued_at, id
		   LIMIT 1 FOR UPDATE SKIP LOCKED)
		 RETURNING `+jobColumns,
		worker, instance, lane, namVersion))
	if errors.Is(err, pgx.ErrNoRows) {
		return jobs.Job{}, false, nil
	}
	if err != nil {
		return jobs.Job{}, false, fmt.Errorf("claim next (%s): %w", lane, err)
	}
	return j, true, nil
}

// SetJobPGID records the process-group leader of a running attempt. ok=false means
// the row is no longer this attempt's (cancelled/requeued in the claim→spawn
// window) — the caller must then kill the child it just spawned.
func (s *Store) SetJobPGID(ctx context.Context, id int64, token string, pgid int) (bool, error) {
	tag, err := s.pool.Exec(ctx, `UPDATE jobs SET pgid = $3`+fence, id, token, pgid)
	if err != nil {
		return false, fmt.Errorf("set pgid: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}

// RequeueJob returns a running attempt to the queue (a graceful shutdown or a
// kill-pause — never a failure): the claim is released, progress and pgid cleared.
// stop/cancel flags stay on the row; the claim filter skips them and the app
// resolves. Fenced like every other write.
func (s *Store) RequeueJob(ctx context.Context, id int64, token string) error {
	_, err := s.pool.Exec(ctx,
		`UPDATE jobs SET state = 'queued', worker = NULL, worker_instance = NULL, claim_token = NULL,
		        pgid = NULL, claimed_at = NULL, started_at = NULL, epoch = NULL, s_per_epoch = NULL`+fence,
		id, token)
	if err != nil {
		return fmt.Errorf("requeue job: %w", err)
	}
	return nil
}

// UpdateProgress records live epoch + s/epoch (throttled by the caller). A
// non-positive s/epoch is stored NULL (the column CHECKs > 0).
// UpdateProgress writes the live epoch, its seconds-per-epoch, and — when the driver
// has printed one for that epoch — the ESR. Without the ESR here, a watching app can
// only learn how a run is going by re-parsing the whole log; jobs.esr stays NULL for
// the entire run and only appears at the end, which is exactly when it stops mattering.
func (s *Store) UpdateProgress(ctx context.Context, id int64, token string, epoch int, sPerEpoch float64, esr *float64) error {
	var sp *float64
	if sPerEpoch > 0 && !math.IsInf(sPerEpoch, 0) && !math.IsNaN(sPerEpoch) {
		sp = &sPerEpoch
	}
	_, err := s.pool.Exec(ctx, `UPDATE jobs SET epoch = $3, s_per_epoch = $4, esr = COALESCE($5, esr)`+fence,
		id, token, epoch, sp, esrArg(esr))
	if err != nil {
		return fmt.Errorf("update progress: %w", err)
	}
	return nil
}

// RecordEpoch writes one epoch's numbers — the run as data, beside job_log's story. Fenced like every
// other write after a claim, and an upsert because a requeued attempt starts its epochs again and the
// newest attempt is the truth.
func (s *Store) RecordEpoch(ctx context.Context, id int64, token string, epoch int, esr *float64, sPerEpoch float64) error {
	var sp *float64
	if sPerEpoch > 0 && !math.IsInf(sPerEpoch, 0) && !math.IsNaN(sPerEpoch) {
		sp = &sPerEpoch
	}
	_, err := s.pool.Exec(ctx,
		`INSERT INTO job_epochs (job_id, claim_token, epoch, esr, s_per_epoch)
		 SELECT $1, $2::uuid, $3, $4, $5
		  WHERE EXISTS (SELECT 1 FROM jobs WHERE id = $1 AND claim_token = $2::uuid AND state = 'running')
		 ON CONFLICT (job_id, epoch) DO UPDATE
		    SET claim_token = EXCLUDED.claim_token, esr = EXCLUDED.esr,
		        s_per_epoch = EXCLUDED.s_per_epoch, at = now()`,
		id, token, epoch, esrArg(esr), sp)
	if err != nil {
		return fmt.Errorf("record epoch: %w", err)
	}
	return nil
}

// AppendLog appends one stdout line, but ONLY while this attempt is still running:
// the EXISTS gate inserts nothing for a straggler (no FK error, no scribble onto a
// newer attempt).
func (s *Store) AppendLog(ctx context.Context, id int64, token, line string) error {
	_, err := s.pool.Exec(ctx,
		`INSERT INTO job_log (job_id, claim_token, line)
		 SELECT $1, $2::uuid, $3 WHERE EXISTS (SELECT 1 FROM jobs`+fence+`)`,
		id, token, line)
	if err != nil {
		return fmt.Errorf("append log: %w", err)
	}
	return nil
}

// Control is the command block of a running row — what the app asked for since
// the last poll.
type Control struct {
	StopRequestedAt   *time.Time
	CancelRequestedAt *time.Time
	LiveRequestedAt   *time.Time
	LiveServedAt      *time.Time
	StopState         *string
}

// Control reads the command flags of a running attempt. ok=false means the row is
// no longer this attempt's (cancelled by the app after our heartbeat went stale,
// or requeued) — the caller then kills its child and writes nothing more.
func (s *Store) Control(ctx context.Context, id int64, token string) (Control, bool, error) {
	var c Control
	err := s.pool.QueryRow(ctx,
		`SELECT stop_requested_at, cancel_requested_at, live_requested_at, live_served_at, stop_state FROM jobs`+fence,
		id, token).Scan(&c.StopRequestedAt, &c.CancelRequestedAt, &c.LiveRequestedAt, &c.LiveServedAt, &c.StopState)
	if errors.Is(err, pgx.ErrNoRows) {
		return Control{}, false, nil
	}
	if err != nil {
		return Control{}, false, fmt.Errorf("read control: %w", err)
	}
	return c, true, nil
}

// SetStopState acknowledges a stop request: stop_seen_at is stamped once, stop_state
// moves to pending | armed | refused (done lands with the terminal write).
func (s *Store) SetStopState(ctx context.Context, id int64, token, state string) (bool, error) {
	tag, err := s.pool.Exec(ctx,
		`UPDATE jobs SET stop_seen_at = COALESCE(stop_seen_at, now()), stop_state = $3`+fence, id, token, state)
	if err != nil {
		return false, fmt.Errorf("set stop_state %s: %w", state, err)
	}
	return tag.RowsAffected() > 0, nil
}

// LiveServed answers a live request WITHOUT a snapshot: live_served_at = now() and
// the reason (jobs.LiveNoCheckpoint | LiveTransient).
func (s *Store) LiveServed(ctx context.Context, id int64, token, liveError string) error {
	_, err := s.pool.Exec(ctx, `UPDATE jobs SET live_served_at = now(), live_error = $3`+fence, id, token, liveError)
	if err != nil {
		return fmt.Errorf("live served (%s): %w", liveError, err)
	}
	return nil
}

// ServeLive answers a live request WITH a snapshot, in one transaction: the
// job_snapshots row (fenced by the EXISTS gate) and live_served_at = now(),
// live_error = NULL. requestedAt is the live_requested_at this answers.
func (s *Store) ServeLive(ctx context.Context, id int64, token string, requestedAt time.Time, epoch int64, esr float64, nam []byte) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin serve live: %w", err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck // no-op after commit
	tag, err := tx.Exec(ctx,
		`INSERT INTO job_snapshots (job_id, claim_token, requested_at, epoch, esr, sha256, size, bytes)
		 SELECT $1, $2::uuid, $3, $4, $5, $6, $7, $8 WHERE EXISTS (SELECT 1 FROM jobs`+fence+`)`,
		id, token, requestedAt, epoch, esrArg(&esr), sha256Hex(nam), len(nam), nam)
	if err != nil {
		return fmt.Errorf("insert snapshot: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return nil // no longer ours — nothing to answer
	}
	if _, err := tx.Exec(ctx, `UPDATE jobs SET live_served_at = now(), live_error = NULL`+fence, id, token); err != nil {
		return fmt.Errorf("mark live served: %w", err)
	}
	return tx.Commit(ctx)
}

// Result is what a succeeded train-lane run keeps: the .nam bytes, the torch
// checkpoint (nil when the run left none — a stop&keep may lack one), the epochs
// the weights HAVE (reached — passed EXPLICITLY from what the trainer was spawned
// with or what the stop harvested, never read back from the row) and the ESR.
type Result struct {
	Reached int64
	ESR     *float64
	Nam     []byte
	Ckpt    []byte
}

// FinishSucceeded writes the terminal state of a train/train_more run AND its
// job_result row in ONE transaction. A stop that was requested is answered
// stop_state='done' in the same write. ok=false when the row was no longer this
// attempt's (nothing written).
func (s *Store) FinishSucceeded(ctx context.Context, id int64, token string, r Result, prov Provenance) (bool, error) {
	if len(r.Nam) == 0 {
		return false, errors.New("finish succeeded: empty model")
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return false, fmt.Errorf("begin finish: %w", err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	// A cancel that arrived after the last control poll must still WIN here: the user
	// asked for this run to be thrown away, and a model that lands anyway would quietly
	// replace the one they kept. Resolving it in the terminal transaction is the only
	// place where "cancelled" and "finished" cannot both be true.
	var state string
	if err := tx.QueryRow(ctx,
		`UPDATE jobs SET state = CASE WHEN cancel_requested_at IS NOT NULL THEN 'cancelled'::job_state ELSE 'succeeded'::job_state END,
		        finished_at = now(), pgid = NULL,
		        reached = $3, esr = $4,
		        error_code = CASE WHEN cancel_requested_at IS NOT NULL THEN 'cancelled'::job_error ELSE NULL END,
		        error_message = NULL,
		        nam_version = $5, driver_sha256 = $6, signal_sha256 = $7,
		        stop_state = CASE WHEN stop_requested_at IS NOT NULL THEN 'done' ELSE stop_state END`+fence+
			` RETURNING state::text`,
		id, token, r.Reached, esrArg(r.ESR), strArg(prov.NamVersion), shaArg(prov.DriverSHA256), shaArg(prov.SignalSHA256)).Scan(&state); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return false, nil
		}
		return false, fmt.Errorf("finish update: %w", err)
	}
	if state == "cancelled" {
		// nothing is kept: no weights, no checkpoint. The log stays as the story.
		if err := tx.Commit(ctx); err != nil {
			return false, fmt.Errorf("finish commit: %w", err)
		}
		return true, nil
	}
	var ckptSize *int
	if len(r.Ckpt) > 0 {
		n := len(r.Ckpt)
		ckptSize = &n
	} else {
		r.Ckpt = nil
	}
	if _, err := tx.Exec(ctx,
		`INSERT INTO job_result (job_id, claim_token, sha256, size, bytes, epochs, esr, ckpt_size, ckpt)
		 VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8, $9)`,
		id, token, sha256Hex(r.Nam), len(r.Nam), r.Nam, r.Reached, esrArg(r.ESR), ckptSize, r.Ckpt); err != nil {
		return false, fmt.Errorf("insert result: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return false, fmt.Errorf("finish commit: %w", err)
	}
	return true, nil
}

// FinishProbeSelf marks a probe_self succeeded with its verdict (+ ESR if known).
// No result row — probes keep nothing.
func (s *Store) FinishProbeSelf(ctx context.Context, id int64, token, verdict string, esr *float64, prov Provenance) (bool, error) {
	tag, err := s.pool.Exec(ctx,
		`UPDATE jobs SET state = CASE WHEN cancel_requested_at IS NOT NULL THEN 'cancelled'::job_state ELSE 'succeeded'::job_state END,
		        finished_at = now(), pgid = NULL,
		        verdict = $3, esr = $4,
		        error_code = CASE WHEN cancel_requested_at IS NOT NULL THEN 'cancelled'::job_error ELSE NULL END,
		        error_message = NULL,
		        nam_version = $5, driver_sha256 = $6, signal_sha256 = $7,
		        stop_state = CASE WHEN stop_requested_at IS NOT NULL THEN 'done' ELSE stop_state END`+fence,
		id, token, verdict, esrArg(esr), strArg(prov.NamVersion), shaArg(prov.DriverSHA256), shaArg(prov.SignalSHA256))
	if err != nil {
		return false, fmt.Errorf("finish probe: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}

// FinishFailed marks an attempt failed with a code from the job_error domain. A
// stop_failed additionally answers the stop request with stop_state='refused'.
func (s *Store) FinishFailed(ctx context.Context, id int64, token, code, message string, prov Provenance) (bool, error) {
	tag, err := s.pool.Exec(ctx,
		`UPDATE jobs SET state = 'failed', finished_at = now(), pgid = NULL,
		        error_code = $3, error_message = $4,
		        nam_version = $5, driver_sha256 = $6, signal_sha256 = $7,
		        stop_state = CASE WHEN $8 THEN 'refused' ELSE stop_state END`+fence,
		id, token, code, strArg(message), strArg(prov.NamVersion), shaArg(prov.DriverSHA256), shaArg(prov.SignalSHA256),
		code == jobs.ErrStopFailed)
	if err != nil {
		return false, fmt.Errorf("finish failed (%s): %w", code, err)
	}
	return tag.RowsAffected() > 0, nil
}

// FinishCancelled answers cancel_requested_at on a running attempt: the child was
// killed, nothing is kept.
func (s *Store) FinishCancelled(ctx context.Context, id int64, token string, prov Provenance) (bool, error) {
	tag, err := s.pool.Exec(ctx,
		`UPDATE jobs SET state = 'cancelled', finished_at = now(), pgid = NULL,
		        error_code = 'cancelled', error_message = NULL,
		        nam_version = $3, driver_sha256 = $4, signal_sha256 = $5`+fence,
		id, token, strArg(prov.NamVersion), shaArg(prov.DriverSHA256), shaArg(prov.SignalSHA256))
	if err != nil {
		return false, fmt.Errorf("finish cancelled: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}

// RecoverRunning is the restart-recovery pass: every row left running by a
// previous process OF THIS WORKER goes back to the queue (claim released, progress
// cleared) and its recorded pgid is returned for the argv-guarded kill. Only this
// worker's rows — another box's running jobs are its own business. stop/cancel
// flags stay on the rows (the claim filter skips them, the app resolves).
func (s *Store) RecoverRunning(ctx context.Context, worker string) ([]int, error) {
	// RETURNING sees the NEW row, so the old pgid is captured by the CTE first.
	rows, err := s.pool.Query(ctx,
		`WITH mine AS (SELECT id, pgid FROM jobs WHERE state = 'running' AND worker = $1 FOR UPDATE)
		 UPDATE jobs j SET state = 'queued', worker = NULL, worker_instance = NULL, claim_token = NULL,
		        pgid = NULL, claimed_at = NULL, started_at = NULL, epoch = NULL, s_per_epoch = NULL
		 FROM mine WHERE j.id = mine.id
		 RETURNING mine.pgid`, worker)
	if err != nil {
		return nil, fmt.Errorf("recover running: %w", err)
	}
	defer rows.Close()
	var pgids []int
	for rows.Next() {
		var pgid *int
		if err := rows.Scan(&pgid); err != nil {
			return nil, fmt.Errorf("scan pgid: %w", err)
		}
		if pgid != nil {
			pgids = append(pgids, *pgid)
		}
	}
	return pgids, rows.Err()
}

// AvgSPerEpochWindow is how many recently-computed training epochs the heartbeat
// average spans.
const AvgSPerEpochWindow = 30

// AvgSPerEpoch returns the seconds-per-epoch averaged over the last
// AvgSPerEpochWindow computed training epochs ON THIS WORKER, weighted by epoch
// count: each terminal train-lane job contributes the epochs it actually computed
// (epoch + 1 − COALESCE(start_epoch,0) — a continuation counts only its own), the
// one oldest job straddling the window edge is clipped to the epochs that fall
// inside it, and a pathological row is clamped to weight 1 (GREATEST). nil when
// there is no such history. The app reads it from workers.avg_s_per_epoch for a
// queue ETA that is stable even when idle.
func (s *Store) AvgSPerEpoch(ctx context.Context, worker string) (*float64, error) {
	var avg *float64
	err := s.pool.QueryRow(ctx, `
		WITH recent AS (
		  SELECT spe, ep, SUM(ep) OVER (ORDER BY finished_at DESC, id) AS cum FROM (
		    SELECT s_per_epoch AS spe, GREATEST(1, epoch + 1 - COALESCE(start_epoch, 0))::bigint AS ep, finished_at, id
		    FROM jobs
		    WHERE lane = 'train' AND worker = $1 AND state IN ('succeeded', 'failed')
		      AND s_per_epoch IS NOT NULL AND epoch IS NOT NULL AND finished_at IS NOT NULL
		    ORDER BY finished_at DESC, id
		    LIMIT $2
		  ) x
		),
		windowed AS (
		  SELECT spe, LEAST(ep, $2::bigint - (cum - ep)) AS w FROM recent WHERE cum - ep < $2::bigint
		)
		SELECT SUM(spe * w) / SUM(w) FROM windowed`,
		worker, int64(AvgSPerEpochWindow)).Scan(&avg)
	if err != nil {
		return nil, fmt.Errorf("avg s/epoch: %w", err)
	}
	return avg, nil
}

// CountByState returns how many jobs THIS worker is running and how many queued
// rows are claimable by anyone (no stop/cancel pending). The keep-awake assertion
// and the heartbeat's `running` hang off it.
func (s *Store) CountByState(ctx context.Context, worker string) (running, queued int, err error) {
	err = s.pool.QueryRow(ctx,
		`SELECT (SELECT count(*) FROM jobs WHERE state = 'running' AND worker = $1),
		        (SELECT count(*) FROM jobs WHERE state = 'queued'
		           AND cancel_requested_at IS NULL AND stop_requested_at IS NULL AND take_id IS NOT NULL)`, worker).Scan(&running, &queued)
	if err != nil {
		return 0, 0, fmt.Errorf("count by state: %w", err)
	}
	return running, queued, nil
}

// esrArg sanitizes an ESR for the `esr >= 0` CHECKs: nil, NaN, ±Inf or a negative
// value become NULL ("not said") rather than a rejected transaction.
func esrArg(v *float64) *float64 {
	if v == nil || math.IsNaN(*v) || math.IsInf(*v, 0) || *v < 0 {
		return nil
	}
	return v
}

// strArg renders "" as NULL — the schema's rule: an optional text is NULL when
// unstated, never ”.
func strArg(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// shaArg renders a sha256 hex for a sha256_hex column: NULL unless it is exactly
// 64 lower-case hex characters (the domain would reject anything else).
func shaArg(s string) *string {
	if len(s) != 64 {
		return nil
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return nil
		}
	}
	return &s
}

// sha256Hex is the lower-case hex sha256 of b.
func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
