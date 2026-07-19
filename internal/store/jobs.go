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

// ErrExists is returned by InsertJob when a row with the same key already exists.
// The key is identity (same key ⇒ same work), so the handler turns this into an
// idempotent 200 rather than an error.
var ErrExists = errors.New("job already exists")

// jobColumns is the explicit column list scanned into a jobs.Job, kept in one
// place so the SELECTs and the scanner never drift.
const jobColumns = `j.key, j.kind, j.state, j.priority, j.epochs, j.arch, j.created_at,
	j.started_at, j.finished_at, j.pid, j.epoch, j.s_per_epoch, j.verdict, j.esr,
	j.error_code, j.error_msg, (r.nam IS NOT NULL) AS has_model`

// InsertJob writes the job row and its capture blob in ONE transaction — the
// atomicity guarantee behind storing WAVs in the DB (a row is never visible
// without its blob, and vice-versa). A duplicate key yields ErrExists.
func (s *Store) InsertJob(ctx context.Context, j jobs.Job, wav []byte) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // rollback after commit is a no-op

	_, err = tx.ExecContext(ctx,
		`INSERT INTO jobs(key, kind, state, priority, epochs, arch, created_at)
		 VALUES(?, ?, ?, ?, ?, ?, ?)`,
		j.Key, j.Kind, jobs.StateQueued, j.Priority, j.Epochs, j.Arch, j.CreatedAt)
	if err != nil {
		if isUniqueViolation(err) {
			return ErrExists
		}
		return fmt.Errorf("insert job: %w", err)
	}
	if _, err = tx.ExecContext(ctx,
		`INSERT INTO audio_blobs(job_key, content) VALUES(?, ?)`, j.Key, wav); err != nil {
		return fmt.Errorf("insert blob: %w", err)
	}
	if err = tx.Commit(); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	return nil
}

// GetJob returns the job (with has_model) and ok=false if the key is unknown.
func (s *Store) GetJob(ctx context.Context, key string) (jobs.Job, bool, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT `+jobColumns+` FROM jobs j LEFT JOIN results r ON r.job_key = j.key WHERE j.key = ?`, key)
	j, err := scanJob(row)
	if errors.Is(err, sql.ErrNoRows) {
		return jobs.Job{}, false, nil
	}
	if err != nil {
		return jobs.Job{}, false, fmt.Errorf("get job: %w", err)
	}
	return j, true, nil
}

// DeleteJob removes the job row (CASCADE removes its blob, results and log). It
// returns whether a row existed. This FREES the key (delete-then-resubmit is the
// retry path).
func (s *Store) DeleteJob(ctx context.Context, key string) (bool, error) {
	res, err := s.db.ExecContext(ctx, `DELETE FROM jobs WHERE key = ?`, key)
	if err != nil {
		return false, fmt.Errorf("delete job: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("rows affected: %w", err)
	}
	return n > 0, nil
}

// DeleteIfQueued atomically removes a job only if it is still queued, returning
// whether it did. This is the DELETE fast path: a queued job has no process, so
// removing it in one statement avoids the check-then-kill-then-delete race where
// a worker claims the row between the handler's lookup and its delete.
func (s *Store) DeleteIfQueued(ctx context.Context, key string) (bool, error) {
	res, err := s.db.ExecContext(ctx, `DELETE FROM jobs WHERE key = ? AND state = 'queued'`, key)
	if err != nil {
		return false, fmt.Errorf("delete if queued: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("rows affected: %w", err)
	}
	return n > 0, nil
}

// QueuedPosition returns the 1-based position of a QUEUED job in its lane's drain
// order (priority asc, created_at asc, key asc as a stable tiebreak). It is
// LANE-SCOPED: only jobs of the same kind count ahead, because the scheduler
// claims per lane (train / probe_self / probe_e10 drain concurrently), so a
// cross-lane count would disagree with the actual pop order. ok=false when the key
// is unknown or the job is not queued (running/terminal have no position).
func (s *Store) QueuedPosition(ctx context.Context, key string) (int, bool, error) {
	var (
		priority  int
		createdAt int64
		state     string
		kind      string
	)
	err := s.db.QueryRowContext(ctx,
		`SELECT priority, created_at, state, kind FROM jobs WHERE key = ?`, key).
		Scan(&priority, &createdAt, &state, &kind)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, fmt.Errorf("lookup job: %w", err)
	}
	if state != jobs.StateQueued {
		return 0, false, nil
	}

	var ahead int
	err = s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM jobs
		 WHERE state = 'queued' AND kind = ?
		   AND ( priority < ?
		      OR (priority = ? AND created_at < ?)
		      OR (priority = ? AND created_at = ? AND key < ?) )`,
		kind, priority, priority, createdAt, priority, createdAt, key).Scan(&ahead)
	if err != nil {
		return 0, false, fmt.Errorf("count ahead: %w", err)
	}
	return ahead + 1, true, nil
}

// SetPriorityIfQueued updates priority only for a QUEUED job (running/terminal is
// a documented no-op). existed reports whether the key is known at all, so the
// handler can 404 an unknown key while 204-ing every real one.
func (s *Store) SetPriorityIfQueued(ctx context.Context, key string, priority int) (existed bool, err error) {
	var state string
	err = s.db.QueryRowContext(ctx, `SELECT state FROM jobs WHERE key = ?`, key).Scan(&state)
	if errors.Is(err, sql.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("lookup job: %w", err)
	}
	if state == jobs.StateQueued {
		if _, err = s.db.ExecContext(ctx,
			`UPDATE jobs SET priority = ? WHERE key = ? AND state = 'queued'`, priority, key); err != nil {
			return true, fmt.Errorf("update priority: %w", err)
		}
	}
	return true, nil
}

// ModelBytes returns the stored .nam for a succeeded job. ok=false when the key
// is unknown or the model blob is absent (never produced, a probe, or GC'd).
func (s *Store) ModelBytes(ctx context.Context, key string) ([]byte, bool, error) {
	var nam []byte
	err := s.db.QueryRowContext(ctx,
		`SELECT nam FROM results WHERE job_key = ? AND nam IS NOT NULL`, key).Scan(&nam)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("get model: %w", err)
	}
	return nam, true, nil
}

// JobExists reports whether a key is known (used by the log endpoint to
// distinguish an unknown job from one with no log lines yet).
func (s *Store) JobExists(ctx context.Context, key string) (bool, error) {
	var one int
	err := s.db.QueryRowContext(ctx, `SELECT 1 FROM jobs WHERE key = ?`, key).Scan(&one)
	if errors.Is(err, sql.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("job exists: %w", err)
	}
	return true, nil
}

// JobLog returns the job's training stdout, one row per line, in insert order —
// one code path for both live tailing and historical reads.
func (s *Store) JobLog(ctx context.Context, key string) ([]string, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT line FROM job_log WHERE job_key = ? ORDER BY id`, key)
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

// scanner is satisfied by both *sql.Row and *sql.Rows.
type scanner interface {
	Scan(dest ...any) error
}

func scanJob(sc scanner) (jobs.Job, error) {
	var (
		j          jobs.Job
		startedAt  sql.NullInt64
		finishedAt sql.NullInt64
		pid        sql.NullInt64
		epoch      sql.NullInt64
		sPerEpoch  sql.NullFloat64
		verdict    sql.NullString
		esr        sql.NullFloat64
		errorCode  sql.NullString
		errorMsg   sql.NullString
		hasModel   bool
	)
	if err := sc.Scan(
		&j.Key, &j.Kind, &j.State, &j.Priority, &j.Epochs, &j.Arch, &j.CreatedAt,
		&startedAt, &finishedAt, &pid, &epoch, &sPerEpoch, &verdict, &esr,
		&errorCode, &errorMsg, &hasModel,
	); err != nil {
		return jobs.Job{}, err
	}
	j.StartedAt = nullInt(startedAt)
	j.FinishedAt = nullInt(finishedAt)
	j.PID = nullInt(pid)
	j.Epoch = nullInt(epoch)
	j.SPerEpoch = nullFloat(sPerEpoch)
	j.Verdict = nullStr(verdict)
	j.ESR = nullFloat(esr)
	j.ErrorCode = nullStr(errorCode)
	j.ErrorMsg = nullStr(errorMsg)
	j.HasModel = hasModel
	return j, nil
}

func nullInt(n sql.NullInt64) *int64 {
	if !n.Valid {
		return nil
	}
	v := n.Int64
	return &v
}

func nullFloat(n sql.NullFloat64) *float64 {
	if !n.Valid {
		return nil
	}
	v := n.Float64
	return &v
}

func nullStr(n sql.NullString) *string {
	if !n.Valid {
		return nil
	}
	v := n.String
	return &v
}

// isUniqueViolation reports whether err is a SQLite UNIQUE/PRIMARY KEY conflict.
// modernc surfaces these as a message; matching on the text keeps the store free
// of a driver-specific error-code import.
func isUniqueViolation(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToUpper(err.Error())
	return strings.Contains(msg, "UNIQUE CONSTRAINT") || strings.Contains(msg, "PRIMARY KEY")
}
