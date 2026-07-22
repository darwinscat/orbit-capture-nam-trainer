// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"orbit-capture-nam-trainer/internal/jobkey"
	"orbit-capture-nam-trainer/internal/jobs"
)

// ErrExists is returned by InsertJob when a row with the same key already exists.
// The key is identity (same key ⇒ same work), so the handler turns this into an
// idempotent 200 rather than an error.
var ErrExists = errors.New("job already exists")

// ErrBaseUnavailable is the sentinel InsertJob returns for a kind=train_more whose
// parent cannot seed a resume; errors.Is(err, ErrBaseUnavailable) matches it. The
// concrete *BaseUnavailableError (errors.As) carries the human-readable Reason the
// HTTP layer surfaces as the 409 base_unavailable body (a later step).
var ErrBaseUnavailable = errors.New("base unavailable")

// BaseUnavailableError names which train_more eligibility check failed. Every
// instance unwraps to ErrBaseUnavailable.
type BaseUnavailableError struct {
	Reason string
}

func (e *BaseUnavailableError) Error() string { return "base unavailable: " + e.Reason }
func (e *BaseUnavailableError) Unwrap() error { return ErrBaseUnavailable }

// jobColumns is the explicit column list scanned into a jobs.Job, kept in one
// place so the SELECTs and the scanner never drift.
const jobColumns = `j.key, j.kind, j.state, j.priority, j.epochs, j.arch, j.created_at,
	j.started_at, j.finished_at, j.pid, j.epoch, j.s_per_epoch, j.verdict, j.esr,
	j.error_code, j.error_msg, j.wav_sha, j.base_key, j.start_epoch, j.reached,
	(r.nam IS NOT NULL) AS has_model`

// InsertJob writes the job row and its capture blob in ONE transaction — the
// atomicity guarantee behind storing WAVs in the DB (a row is never visible
// without its blob, and vice-versa). The capture's sha256 is always recorded in
// wav_sha. A duplicate key yields ErrExists.
//
// For a kind=train_more job (j.BaseKey set) the parent is validated and its
// checkpoint is snapshotted into resume_ckpts INSIDE this same transaction, so the
// child is self-contained the instant it commits (the parent may then be deleted or
// GC'd freely). BEGIN IMMEDIATE (the DSN's _txlock) makes that atomic against a
// concurrent parent DELETE: an ineligible parent rolls the whole insert back —
// there is never a child row without its snapshot, nor a dangling snapshot. An
// ineligible parent yields ErrBaseUnavailable.
func (s *Store) InsertJob(ctx context.Context, j jobs.Job, wav []byte) error {
	// The HTTP layer has already hashed the body for its key recompute and passes
	// the hex in j.WavSHA — don't hash a multi-MB capture twice per PUT. The
	// fallback keeps direct callers (tests) honest.
	var wavSHA string
	if j.WavSHA != nil {
		wavSHA = *j.WavSHA
	} else {
		wavSHA = jobkey.SHA256Hex(wav)
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // rollback after commit is a no-op

	// The child row goes in FIRST so a duplicate key is reported as ErrExists
	// regardless of the parent's (train_more) eligibility. start_epoch is filled in
	// by snapshotParent below once the parent's epochs are read; a plain job leaves
	// it NULL.
	_, err = tx.ExecContext(ctx,
		`INSERT INTO jobs(key, kind, state, priority, epochs, arch, created_at, wav_sha, base_key)
		 VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		j.Key, j.Kind, jobs.StateQueued, j.Priority, j.Epochs, j.Arch, j.CreatedAt, wavSHA, strArg(j.BaseKey))
	if err != nil {
		if isUniqueViolation(err) {
			return ErrExists
		}
		return fmt.Errorf("insert job: %w", err)
	}
	if j.BaseKey != nil {
		if err := snapshotParent(ctx, tx, j, wavSHA); err != nil {
			return err // ErrBaseUnavailable or a wrapped DB error; rollback drops the child
		}
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

// snapshotParent validates a train_more child's parent and, on success, sets the
// child's start_epoch to the parent's reached count and copies the parent's
// checkpoint into resume_ckpts(child). The numbering origin is COALESCE(reached,
// epochs): reached equals epochs for a natural finish, is the harvested count for
// an early stop, and is NULL for a probe or a pre-v3 row — in which case we fall
// back to epochs (a probe's fixed count; a pre-v3 train's full requested count).
// Eligibility is KIND-AGNOSTIC — any succeeded job with a stored ckpt qualifies (a
// probe_e10 that ran to completion does; a probe_self, killed before epoch 0, never
// has one). All checks run inside the caller's tx.
func snapshotParent(ctx context.Context, tx *sql.Tx, child jobs.Job, childWavSHA string) error {
	var (
		state       string
		parentEp    int64
		parentReach sql.NullInt64
		parentArc   string
		parentWav   sql.NullString
		hasCkpt     bool
	)
	err := tx.QueryRowContext(ctx,
		`SELECT j.state, j.epochs, j.reached, j.arch, j.wav_sha, (r.ckpt IS NOT NULL)
		 FROM jobs j LEFT JOIN results r ON r.job_key = j.key WHERE j.key = ?`, *child.BaseKey).
		Scan(&state, &parentEp, &parentReach, &parentArc, &parentWav, &hasCkpt)
	if errors.Is(err, sql.ErrNoRows) {
		return &BaseUnavailableError{Reason: "parent job not found"}
	}
	if err != nil {
		return fmt.Errorf("read parent: %w", err)
	}
	// origin = COALESCE(parent.reached, parent.epochs): where this child's numbering
	// begins and the floor its epochs must exceed.
	origin := parentEp
	if parentReach.Valid {
		origin = parentReach.Int64
	}
	switch {
	case state != jobs.StateSucceeded:
		return &BaseUnavailableError{Reason: "parent job has not succeeded"}
	case !hasCkpt: // results.ckpt IS NULL — never produced, a probe_self, or GC'd
		return &BaseUnavailableError{Reason: "parent job has no stored checkpoint"}
	case !parentWav.Valid || parentWav.String != childWavSHA:
		return &BaseUnavailableError{Reason: "capture does not match the parent"}
	case parentArc != child.Arch:
		return &BaseUnavailableError{Reason: "arch does not match the parent"}
	case int64(child.Epochs) <= origin:
		// Accurate whether the parent finished naturally (reached == epochs) or was
		// stopped early (reached < epochs): the child must exceed the parent's
		// reached count to have new epochs to compute.
		return &BaseUnavailableError{Reason: "epochs must exceed the parent's reached count"}
	}
	if _, err := tx.ExecContext(ctx,
		`UPDATE jobs SET start_epoch = ? WHERE key = ?`, origin, child.Key); err != nil {
		return fmt.Errorf("set start_epoch: %w", err)
	}
	// The ~505 KB blob is copied SQL-side (INSERT…SELECT): it never crosses into Go
	// while we hold the write lock. The eligibility check above ran in this same tx,
	// so exactly one row must be copied — anything else is a store invariant break.
	res, err := tx.ExecContext(ctx,
		`INSERT INTO resume_ckpts(job_key, content)
		 SELECT ?, ckpt FROM results WHERE job_key = ? AND ckpt IS NOT NULL`,
		child.Key, *child.BaseKey)
	if err != nil {
		return fmt.Errorf("snapshot ckpt: %w", err)
	}
	if n, err := res.RowsAffected(); err != nil || n != 1 {
		return fmt.Errorf("snapshot ckpt: copied %d rows (err=%v), want 1", n, err)
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
// LANE-SCOPED: only jobs in the same lane count ahead, because the scheduler claims
// per lane (train+train_more / probe_self / probe_e10 drain concurrently), so a
// cross-lane count would disagree with the actual pop order. train and train_more
// share the train lane, so each counts the other ahead of it — matching
// ClaimNextQueued, which claims both from that lane. ok=false when the key is
// unknown or the job is not queued (running/terminal have no position).
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

	frag, args := kindsIn("kind", jobs.LaneKinds(kind))
	args = append(args, priority, priority, createdAt, priority, createdAt, key)

	var ahead int
	err = s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM jobs
		 WHERE state = 'queued' AND `+frag+`
		   AND ( priority < ?
		      OR (priority = ? AND created_at < ?)
		      OR (priority = ? AND created_at = ? AND key < ?) )`,
		args...).Scan(&ahead)
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
		wavSHA     sql.NullString
		baseKey    sql.NullString
		startEpoch sql.NullInt64
		reached    sql.NullInt64
		hasModel   bool
	)
	if err := sc.Scan(
		&j.Key, &j.Kind, &j.State, &j.Priority, &j.Epochs, &j.Arch, &j.CreatedAt,
		&startedAt, &finishedAt, &pid, &epoch, &sPerEpoch, &verdict, &esr,
		&errorCode, &errorMsg, &wavSHA, &baseKey, &startEpoch, &reached, &hasModel,
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
	j.WavSHA = nullStr(wavSHA)
	j.BaseKey = nullStr(baseKey)
	j.StartEpoch = nullInt(startEpoch)
	j.Reached = nullInt(reached)
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

// strArg renders a *string as a NULL-able SQL argument.
func strArg(s *string) any {
	if s == nil {
		return nil
	}
	return *s
}

// kindsIn renders an SQL "<col> IN (?, …)" fragment plus its bind args for a set
// of job kinds, so every lane-scoped query binds the same jobs.LaneKinds source
// instead of hand-rolling placeholders (or worse, string literals that the
// compiler can't tie back to the kind constants).
func kindsIn(col string, kinds []string) (string, []any) {
	ph := make([]string, len(kinds))
	args := make([]any, len(kinds))
	for i, k := range kinds {
		ph[i] = "?"
		args[i] = k
	}
	return col + " IN (" + strings.Join(ph, ", ") + ")", args
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
