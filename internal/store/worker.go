// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"orbit-capture-nam-trainer/internal/jobs"
)

// ClaimNextQueued atomically pops the highest-priority, oldest queued job and
// flips it to running (started_at set; pid recorded later once the child is
// spawned). When kinds are given it only considers those job kinds — that is how
// the worker runs separate lanes, probes draining in parallel with training
// rather than queueing behind it; empty kinds means any kind. ok=false means the
// queue is empty OR another worker won the row (the caller treats both as
// "nothing claimed" and loops). The CAS on state='queued' in the UPDATE is what
// makes the pop safe between racing workers.
func (s *Store) ClaimNextQueued(ctx context.Context, startedAt int64, kinds ...string) (jobs.Job, bool, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return jobs.Job{}, false, fmt.Errorf("begin claim: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // no-op after commit

	where := "j.state = 'queued'"
	var args []any
	if len(kinds) > 0 {
		ph := make([]string, len(kinds))
		for i, k := range kinds {
			ph[i] = "?"
			args = append(args, k)
		}
		where += " AND j.kind IN (" + strings.Join(ph, ", ") + ")"
	}
	// The key tiebreak matches QueuedPosition's ordering, so the pop order and the
	// reported position never disagree for same-second inserts.
	row := tx.QueryRowContext(ctx,
		`SELECT `+jobColumns+` FROM jobs j LEFT JOIN results r ON r.job_key = j.key
		 WHERE `+where+` ORDER BY j.priority, j.created_at, j.key LIMIT 1`, args...)
	j, err := scanJob(row)
	if errors.Is(err, sql.ErrNoRows) {
		return jobs.Job{}, false, nil
	}
	if err != nil {
		return jobs.Job{}, false, fmt.Errorf("select queued: %w", err)
	}

	// Clear any live-progress carried over from a prior (recovered) run so the row
	// doesn't report a stale epoch during the silent torch-import preamble.
	res, err := tx.ExecContext(ctx,
		`UPDATE jobs SET state='running', started_at=?, epoch=NULL, s_per_epoch=NULL
		 WHERE key=? AND state='queued'`,
		startedAt, j.Key)
	if err != nil {
		return jobs.Job{}, false, fmt.Errorf("claim update: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return jobs.Job{}, false, fmt.Errorf("claim rows: %w", err)
	}
	if n == 0 {
		return jobs.Job{}, false, nil // lost the race — caller retries
	}
	if err := tx.Commit(); err != nil {
		return jobs.Job{}, false, fmt.Errorf("claim commit: %w", err)
	}

	j.State = jobs.StateRunning
	j.StartedAt = &startedAt
	j.Epoch = nil
	j.SPerEpoch = nil
	return j, true, nil
}

// SetJobPID records the process-group leader of a running job. ok=false means the
// job was no longer running (deleted in the claim→register window) — the caller
// must then kill the child it just spawned, since no DELETE could reach it.
func (s *Store) SetJobPID(ctx context.Context, key string, pid int) (ok bool, err error) {
	res, err := s.db.ExecContext(ctx,
		`UPDATE jobs SET pid = ? WHERE key = ? AND state = 'running'`, pid, key)
	if err != nil {
		return false, fmt.Errorf("set pid: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("set pid rows: %w", err)
	}
	return n > 0, nil
}

// RequeueJob returns a running job to the queue, clearing live progress and the
// pid. Used when a job's child was killed by a graceful shutdown (not a failure):
// the job must run again next start, so it must NOT be written terminal.
func (s *Store) RequeueJob(ctx context.Context, key string) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE jobs SET state='queued', pid=NULL, started_at=NULL, epoch=NULL, s_per_epoch=NULL
		 WHERE key=? AND state='running'`, key)
	if err != nil {
		return fmt.Errorf("requeue job: %w", err)
	}
	return nil
}

// UpdateProgress records live epoch + s/epoch for a running job (throttled by the
// caller). The started_at guard ensures a lagging worker from a since-deleted run
// can never write onto a DIFFERENT run that reused the same content key.
func (s *Store) UpdateProgress(ctx context.Context, key string, epoch int, sPerEpoch float64, startedAt int64) error {
	var sp any
	if sPerEpoch > 0 {
		sp = sPerEpoch
	}
	_, err := s.db.ExecContext(ctx,
		`UPDATE jobs SET epoch = ?, s_per_epoch = ? WHERE key = ? AND state = 'running' AND started_at = ?`,
		epoch, sp, key, startedAt)
	if err != nil {
		return fmt.Errorf("update progress: %w", err)
	}
	return nil
}

// AppendLog appends one training stdout line, but ONLY while this exact run (same
// key AND started_at) is still running — so a lagging worker from a deleted run
// can't scribble old lines onto a new run that reused the key, and a deleted row
// simply drops the line (the EXISTS gate inserts nothing, so no FK error fires).
func (s *Store) AppendLog(ctx context.Context, key, line string, startedAt int64) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO job_log(job_key, line)
		 SELECT ?, ? WHERE EXISTS(
		   SELECT 1 FROM jobs WHERE key = ? AND started_at = ? AND state = 'running')`,
		key, line, key, startedAt)
	if err != nil {
		return fmt.Errorf("append log: %w", err)
	}
	return nil
}

// AudioBlob returns the stored capture wav for a job. ok=false when absent
// (already dropped at a terminal state, or the job is unknown).
func (s *Store) AudioBlob(ctx context.Context, key string) ([]byte, bool, error) {
	var content []byte
	err := s.db.QueryRowContext(ctx,
		`SELECT content FROM audio_blobs WHERE job_key = ?`, key).Scan(&content)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("get audio blob: %w", err)
	}
	return content, true, nil
}

// FinishTrainSuccess marks a train job succeeded, stores the model + train.json +
// the final validation ESR, and drops the (now redundant) capture blob — all in
// one transaction. It returns ok=false if the row was no longer running (deleted
// mid-flight): the caller then just wipes scratch and moves on, never
// resurrecting the row.
func (s *Store) FinishTrainSuccess(ctx context.Context, key string, finishedAt int64, nam []byte, trainJSON string, esr *float64) (bool, error) {
	return s.finish(ctx, key, func(tx *sql.Tx) (sql.Result, error) {
		return tx.ExecContext(ctx,
			`UPDATE jobs SET state='succeeded', finished_at=?, pid=NULL, esr=?, error_code=NULL, error_msg=NULL
			 WHERE key=? AND state='running'`, finishedAt, floatArg(esr), key)
	}, func(tx *sql.Tx) error {
		_, err := tx.ExecContext(ctx,
			`INSERT INTO results(job_key, nam, train_json) VALUES(?, ?, ?)
			 ON CONFLICT(job_key) DO UPDATE SET nam=excluded.nam, train_json=excluded.train_json`,
			key, nam, trainJSON)
		return err
	})
}

// FinishProbeSelf marks a probe_self succeeded with its verdict (+ ESR if known).
// No model is stored.
func (s *Store) FinishProbeSelf(ctx context.Context, key string, finishedAt int64, verdict string, esr *float64) (bool, error) {
	return s.finish(ctx, key, func(tx *sql.Tx) (sql.Result, error) {
		return tx.ExecContext(ctx,
			`UPDATE jobs SET state='succeeded', finished_at=?, pid=NULL, verdict=?, esr=?,
			 error_code=NULL, error_msg=NULL WHERE key=? AND state='running'`,
			finishedAt, verdict, floatArg(esr), key)
	}, nil)
}

// FinishProbeE10 marks a probe_e10 succeeded with its E@10 ESR. No model stored.
func (s *Store) FinishProbeE10(ctx context.Context, key string, finishedAt int64, esr float64) (bool, error) {
	return s.finish(ctx, key, func(tx *sql.Tx) (sql.Result, error) {
		return tx.ExecContext(ctx,
			`UPDATE jobs SET state='succeeded', finished_at=?, pid=NULL, esr=?,
			 error_code=NULL, error_msg=NULL WHERE key=? AND state='running'`,
			finishedAt, esr, key)
	}, nil)
}

// FinishFailed marks a job failed (terminal — retry is client DELETE + resubmit),
// keeping the row and its job_log as history but dropping the capture blob.
func (s *Store) FinishFailed(ctx context.Context, key string, finishedAt int64, errorCode, errorMsg string) (bool, error) {
	return s.finish(ctx, key, func(tx *sql.Tx) (sql.Result, error) {
		return tx.ExecContext(ctx,
			`UPDATE jobs SET state='failed', finished_at=?, pid=NULL, error_code=?, error_msg=?
			 WHERE key=? AND state='running'`, finishedAt, errorCode, errorMsg, key)
	}, nil)
}

// finish runs the terminal state transition in one transaction: the guarded UPDATE
// (via update, which must be exactly one `UPDATE jobs ... WHERE key=? AND
// state='running'` and returns its sql.Result), then — only if it affected the
// still-running row — an optional extra step (extra, e.g. storing results) and the
// capture-blob delete. Returns ok=false when the row was not running (deleted
// mid-run).
func (s *Store) finish(ctx context.Context, key string, update func(*sql.Tx) (sql.Result, error), extra func(*sql.Tx) error) (bool, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return false, fmt.Errorf("begin finish: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	// The guarded UPDATE is the gate: RowsAffected tells us whether the job was
	// still running; a deleted job affects 0 rows and we bail cleanly.
	res, err := update(tx)
	if err != nil {
		return false, fmt.Errorf("finish update: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("finish rows: %w", err)
	}
	if n == 0 {
		return false, nil // job left running (deleted mid-run) — nothing to finish
	}

	if extra != nil {
		if err := extra(tx); err != nil {
			return false, fmt.Errorf("finish extra: %w", err)
		}
	}
	// The capture blob is the big object; it is done the moment the job is terminal.
	if _, err := tx.ExecContext(ctx, `DELETE FROM audio_blobs WHERE job_key = ?`, key); err != nil {
		return false, fmt.Errorf("drop blob: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return false, fmt.Errorf("finish commit: %w", err)
	}
	return true, nil
}

// RecoverRunning is the restart-recovery pass (the design notes): it returns
// the recorded pids of every job left running by a previous process (for the
// caller to SIGKILL) and requeues those jobs (order preserved by created_at),
// clearing their pid/started_at. Runs in one transaction.
func (s *Store) RecoverRunning(ctx context.Context) ([]int, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("begin recover: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	rows, err := tx.QueryContext(ctx,
		`SELECT pid FROM jobs WHERE state='running' AND pid IS NOT NULL`)
	if err != nil {
		return nil, fmt.Errorf("select running pids: %w", err)
	}
	var pids []int
	for rows.Next() {
		var pid int64
		if err := rows.Scan(&pid); err != nil {
			rows.Close()
			return nil, fmt.Errorf("scan pid: %w", err)
		}
		pids = append(pids, int(pid))
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return nil, err
	}
	rows.Close()

	if _, err := tx.ExecContext(ctx,
		`UPDATE jobs SET state='queued', pid=NULL, started_at=NULL, epoch=NULL, s_per_epoch=NULL
		 WHERE state='running'`); err != nil {
		return nil, fmt.Errorf("requeue running: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("recover commit: %w", err)
	}
	return pids, nil
}

// AvgSPerEpochWindow is how many recently-computed training epochs the health
// average spans.
const AvgSPerEpochWindow = 30

// AvgSPerEpoch returns the seconds-per-epoch averaged over the last
// AvgSPerEpochWindow computed training epochs on this machine, weighted by epoch
// count: each terminal train job contributes its epochs, and the one oldest job
// that straddles the window edge is clipped to the epochs that fall inside it, so
// the weights sum to exactly the window (or to all history, when there is less).
// The number therefore tracks the machine's recent speed — a device change or a
// thermal slowdown moves it — rather than being dominated by one old long run. It
// looks only at terminal train jobs with a recorded s_per_epoch, newest first, and
// is nil when there is no such history. The app uses it for a queue ETA that is
// stable and available even when idle.
func (s *Store) AvgSPerEpoch(ctx context.Context) (*float64, error) {
	var avg sql.NullFloat64
	err := s.db.QueryRowContext(ctx, `
		WITH recent AS (
		  SELECT spe, ep, SUM(ep) OVER (ORDER BY finished_at DESC, key) AS cum FROM (
		    SELECT s_per_epoch AS spe, (epoch + 1) AS ep, finished_at, key
		    FROM jobs
		    WHERE kind = 'train' AND state IN ('succeeded','failed')
		      AND s_per_epoch IS NOT NULL AND epoch IS NOT NULL AND finished_at IS NOT NULL
		    ORDER BY finished_at DESC, key
		    LIMIT ?
		  )
		),
		windowed AS (
		  SELECT spe, MIN(ep, ? - (cum - ep)) AS w FROM recent WHERE cum - ep < ?
		)
		SELECT SUM(spe * w) / SUM(w) FROM windowed`,
		// LIMIT, MIN-clip, and cum-filter are all the window: each qualifying job
		// contributes >=1 epoch, so the newest AvgSPerEpochWindow rows always cover
		// the AvgSPerEpochWindow-epoch window — no silent truncation on a bump.
		AvgSPerEpochWindow, AvgSPerEpochWindow, AvgSPerEpochWindow).Scan(&avg)
	if err != nil {
		return nil, fmt.Errorf("avg s/epoch: %w", err)
	}
	if !avg.Valid {
		return nil, nil
	}
	v := avg.Float64
	return &v, nil
}

// CountByState returns the number of running and queued jobs. It seeds the
// in-memory /v1/health counters at startup and is re-read on every queue
// transition (worker publishStats) — the steady-state reconcile that both the
// health counts and the keep-awake assertion hang off.
func (s *Store) CountByState(ctx context.Context) (running, queued int, err error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT state, COUNT(*) FROM jobs WHERE state IN ('running','queued') GROUP BY state`)
	if err != nil {
		return 0, 0, fmt.Errorf("count by state: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var st string
		var n int
		if err := rows.Scan(&st, &n); err != nil {
			return 0, 0, err
		}
		switch st {
		case jobs.StateRunning:
			running = n
		case jobs.StateQueued:
			queued = n
		}
	}
	return running, queued, rows.Err()
}

// floatArg renders a *float64 as a NULL-able SQL argument.
func floatArg(f *float64) any {
	if f == nil {
		return nil
	}
	return *f
}

