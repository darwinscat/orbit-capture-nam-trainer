// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"

	"orbit-capture-nam-trainer/internal/jobs"
)

// jobColumns is the explicit column list scanned into a jobs.Job, kept in one
// place so every SELECT/RETURNING and the scanner never drift. The enum and the
// uuid are read as text so the scanner needs no driver-specific types.
// take_id is nullable since the library learned to outlive a wiped take: the journal row stays with
// take_id NULL. 0 means "the take is gone" — the claim filters those out, nothing else may run one.
const jobColumns = `id, coalesce(take_id, 0) AS take_id, take_label, kind, lane, epochs, arch, priority::text, latency_samples,
	wav_sha, required_nam_version, base_job_id, start_epoch,
	stop_requested_at, stop_reason, cancel_requested_at, live_requested_at, queued_at,
	state, worker, worker_instance, claim_token::text, pgid, claimed_at, started_at, finished_at,
	epoch, reached, s_per_epoch, esr, verdict, error_code, error_message,
	stop_seen_at, stop_state, live_served_at, live_error, nam_version, driver_sha256, signal_sha256`

func scanJob(row pgx.Row) (jobs.Job, error) {
	var (
		j          jobs.Job
		claimToken *string
	)
	err := row.Scan(
		&j.ID, &j.TakeID, &j.TakeLabel, &j.Kind, &j.Lane, &j.Epochs, &j.Arch, &j.Priority, &j.Latency,
		&j.WavSHA, &j.RequiredNamVersion, &j.BaseJobID, &j.StartEpoch,
		&j.StopRequestedAt, &j.StopReason, &j.CancelRequestedAt, &j.LiveRequestedAt, &j.QueuedAt,
		&j.State, &j.Worker, &j.WorkerInstance, &claimToken, &j.PGID, &j.ClaimedAt, &j.StartedAt, &j.FinishedAt,
		&j.Epoch, &j.Reached, &j.SPerEpoch, &j.ESR, &j.Verdict, &j.ErrorCode, &j.ErrorMessage,
		&j.StopSeenAt, &j.StopState, &j.LiveServedAt, &j.LiveError, &j.NamVersion, &j.DriverSHA256, &j.SignalSHA256,
	)
	if err != nil {
		return jobs.Job{}, err
	}
	if claimToken != nil {
		j.ClaimToken = *claimToken
	}
	return j, nil
}

// GetJob returns the job row; ok=false when the id is unknown.
func (s *Store) GetJob(ctx context.Context, id int64) (jobs.Job, bool, error) {
	j, err := scanJob(s.pool.QueryRow(ctx, `SELECT `+jobColumns+` FROM jobs WHERE id = $1`, id))
	if errors.Is(err, pgx.ErrNoRows) {
		return jobs.Job{}, false, nil
	}
	if err != nil {
		return jobs.Job{}, false, fmt.Errorf("get job %d: %w", id, err)
	}
	return j, true, nil
}

// JobLog returns the job's training stdout, one row per line, in insert order —
// the live-ESR scan (ExportLive / the stop harvest) reads it.
func (s *Store) JobLog(ctx context.Context, id int64) ([]string, error) {
	rows, err := s.pool.Query(ctx, `SELECT line FROM job_log WHERE job_id = $1 ORDER BY id`, id)
	if err != nil {
		return nil, fmt.Errorf("query log: %w", err)
	}
	defer rows.Close()
	var lines []string
	for rows.Next() {
		var line string
		if err := rows.Scan(&line); err != nil {
			return nil, fmt.Errorf("scan log: %w", err)
		}
		lines = append(lines, line)
	}
	return lines, rows.Err()
}

// TakeAudio returns the take's wav bytes VERBATIM and the sha256 the library
// recorded for them. ok=false when the take has no audio row (the take is gone —
// the job's FK normally forbids that, so it is a belt).
func (s *Store) TakeAudio(ctx context.Context, takeID int64) (wav []byte, sha string, ok bool, err error) {
	err = s.pool.QueryRow(ctx, `SELECT bytes, sha256 FROM take_audio WHERE take_id = $1`, takeID).Scan(&wav, &sha)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, "", false, nil
	}
	if err != nil {
		return nil, "", false, fmt.Errorf("read take_audio %d: %w", takeID, err)
	}
	return wav, sha, true, nil
}

// TakeSignalSHA returns the sha256 of the stimulus the take was recorded against
// (signals.sha256 via takes.signal_id). ok=false when the take is unknown.
func (s *Store) TakeSignalSHA(ctx context.Context, takeID int64) (sha string, ok bool, err error) {
	err = s.pool.QueryRow(ctx,
		`SELECT s.sha256 FROM takes t JOIN signals s ON s.id = t.signal_id WHERE t.id = $1`, takeID).Scan(&sha)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		return "", false, fmt.Errorf("read take signal %d: %w", takeID, err)
	}
	return sha, true, nil
}

// ResumeCkpt returns the parent-checkpoint snapshot the app stored in job_resume
// when it queued a train_more (the worker materializes it into scratch and resumes
// from it). ok=false when absent — the child must then fail base_unavailable,
// never run from scratch.
func (s *Store) ResumeCkpt(ctx context.Context, jobID int64) ([]byte, bool, error) {
	var ckpt []byte
	err := s.pool.QueryRow(ctx, `SELECT ckpt FROM job_resume WHERE job_id = $1`, jobID).Scan(&ckpt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("read job_resume %d: %w", jobID, err)
	}
	return ckpt, true, nil
}
