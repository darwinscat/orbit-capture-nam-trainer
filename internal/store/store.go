// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package store owns the SQLite database: the schema from the design notes and
// the PRAGMAs it must run under. WAVs live in the DB as blobs (atomicity over
// everything — a job row and its capture arrive in one transaction), so the file
// churns 27 MB objects; auto_vacuum=INCREMENTAL keeps it from growing forever.
package store

import (
	"context"
	"database/sql"
	"fmt"

	_ "modernc.org/sqlite" // pure-Go driver, no cgo (keeps cross-compilation trivial)
)

// schema is the DDL from the design notes, verbatim in shape. IF NOT EXISTS
// makes Open idempotent across restarts.
const schema = `
CREATE TABLE IF NOT EXISTS jobs (
  key         TEXT PRIMARY KEY,              -- sha256 hex, client-computed
  kind        TEXT NOT NULL,                 -- train | train_more | probe_self | probe_e10
  state       TEXT NOT NULL,                 -- queued | running | succeeded | failed
  priority    INTEGER NOT NULL DEFAULT 1,    -- 0 high / 1 med / 2 low
  epochs      INTEGER NOT NULL,              -- train: requested; probe_self: 1; probe_e10: 10
  arch        TEXT NOT NULL DEFAULT 'standard',
  created_at  INTEGER NOT NULL,              -- unix seconds
  started_at  INTEGER, finished_at INTEGER,
  pid         INTEGER,                       -- python process-GROUP leader while running
  epoch       INTEGER, s_per_epoch REAL,     -- live progress (raw numbers; client formats)
  verdict     TEXT,                          -- probe_self: pass | fail (NULL otherwise)
  esr         REAL,                          -- probe_self: replicate ESR; probe_e10: E@10
  error_code  TEXT, error_msg TEXT,
  wav_sha     TEXT,                          -- sha256 hex of the capture (every new PUT)
  base_key    TEXT,                          -- train_more: parent key (provenance)
  start_epoch INTEGER                        -- train_more: parent's epochs (numbering origin)
);
CREATE TABLE IF NOT EXISTS audio_blobs (     -- the capture wav; deleted at terminal state
  job_key TEXT PRIMARY KEY REFERENCES jobs(key) ON DELETE CASCADE,
  content BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS results (         -- small
  job_key    TEXT PRIMARY KEY REFERENCES jobs(key) ON DELETE CASCADE,
  nam        BLOB,                           -- the .nam (NULL until succeeded / for probes)
  train_json TEXT,
  ckpt       BLOB                            -- last .ckpt of a succeeded train/train_more/probe_e10
);
CREATE TABLE IF NOT EXISTS resume_ckpts (    -- a train_more child's private parent-ckpt snapshot
  job_key TEXT PRIMARY KEY REFERENCES jobs(key) ON DELETE CASCADE,
  content BLOB NOT NULL                       -- deleted at terminal state (run-input, like audio_blobs)
);
CREATE TABLE IF NOT EXISTS job_log (         -- training stdout, one row per line
  id      INTEGER PRIMARY KEY,               -- append, stable order, live-tailable
  job_key TEXT NOT NULL REFERENCES jobs(key) ON DELETE CASCADE,
  line    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS job_log_key ON job_log(job_key);
CREATE INDEX IF NOT EXISTS jobs_pop ON jobs(state, priority, created_at);
CREATE INDEX IF NOT EXISTS jobs_train_finished ON jobs(kind, finished_at);
`

// Store wraps the database handle.
type Store struct {
	db *sql.DB
}

// DB exposes the underlying handle for packages that run their own queries.
func (s *Store) DB() *sql.DB { return s.db }

// Open opens (creating if absent) the database at path, applies the PRAGMAs, and
// ensures the schema. foreign_keys and busy_timeout are per-connection, so they
// ride the DSN and apply to every pooled connection; auto_vacuum and journal_mode
// are database-level and persistent, so they are set once on a dedicated init
// connection BEFORE any table exists (auto_vacuum can only be chosen on an empty
// database).
//
// _txlock=immediate makes every database/sql BeginTx issue BEGIN IMMEDIATE. This
// is essential under WAL with concurrent writers: a DEFERRED transaction that
// reads and then upgrades to a write (our claim: SELECT queued → UPDATE running)
// takes SQLITE_BUSY *immediately*, bypassing busy_timeout, whenever another
// connection committed a write in between (the read snapshot is invalidated and
// no amount of waiting can cure it). With IMMEDIATE the write lock is taken at
// BEGIN, so racing writers serialize cleanly under busy_timeout instead. Every
// transaction in this codebase writes, so IMMEDIATE is correct blanket policy.
func Open(ctx context.Context, path string) (*Store, error) {
	dsn := "file:" + path + "?_pragma=busy_timeout(5000)&_pragma=foreign_keys(on)&_txlock=immediate"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	db.SetMaxOpenConns(8) // bound per-connection page caches; hygiene, not correctness

	conn, err := db.Conn(ctx)
	if err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("acquire init conn: %w", err)
	}
	defer conn.Close()

	// Order matters: auto_vacuum before any table, before WAL, on the empty file.
	for _, pragma := range []string{
		"PRAGMA auto_vacuum=INCREMENTAL",
		"PRAGMA journal_mode=WAL",
	} {
		if _, err := conn.ExecContext(ctx, pragma); err != nil {
			_ = db.Close()
			return nil, fmt.Errorf("%s: %w", pragma, err)
		}
	}
	if err := migrate(ctx, conn); err != nil {
		_ = db.Close()
		return nil, err
	}

	return &Store{db: db}, nil
}

// schemaVersion is the current on-disk schema version (PRAGMA user_version). There
// was no versioning before this, so every field database starts at 0 and is carried
// to 1 (the base schema) on first open. Bump this and add a step in migrate()
// whenever the DDL changes on an existing database.
//
// v2 (train_more): additive jobs.wav_sha/base_key/start_epoch + results.ckpt +
// the resume_ckpts table. A fresh database gets the whole v2 shape from `schema`
// in one shot; an existing v1 file is carried forward by the guarded ALTERs below.
const schemaVersion = 2

// migrate brings the database up to schemaVersion. Step 1 is the base schema
// (create-if-absent, idempotent). Additive changes to an existing DB (a new column
// or index) become the next numbered step; SQLite DDL has no transactional
// rollback, so each step must be self-contained and safe to re-run after a crash —
// ALTER TABLE ADD COLUMN in particular is not idempotent, so every add is guarded
// by a pragma table_info check (a fresh v2 database already has the columns from
// `schema`; a re-run after a crash mid-step is a no-op).
func migrate(ctx context.Context, conn *sql.Conn) error {
	var v int
	if err := conn.QueryRowContext(ctx, "PRAGMA user_version").Scan(&v); err != nil {
		return fmt.Errorf("read schema version: %w", err)
	}
	if v < 1 {
		if _, err := conn.ExecContext(ctx, schema); err != nil {
			return fmt.Errorf("apply base schema: %w", err)
		}
	}
	if v < 2 {
		for _, c := range []struct{ table, column, decl string }{
			{"jobs", "wav_sha", "TEXT"},
			{"jobs", "base_key", "TEXT"},
			{"jobs", "start_epoch", "INTEGER"},
			{"results", "ckpt", "BLOB"},
		} {
			if err := addColumnIfMissing(ctx, conn, c.table, c.column, c.decl); err != nil {
				return err
			}
		}
		if _, err := conn.ExecContext(ctx,
			`CREATE TABLE IF NOT EXISTS resume_ckpts (
			   job_key TEXT PRIMARY KEY REFERENCES jobs(key) ON DELETE CASCADE,
			   content BLOB NOT NULL)`); err != nil {
			return fmt.Errorf("create resume_ckpts: %w", err)
		}
	}
	if v < schemaVersion {
		if _, err := conn.ExecContext(ctx, fmt.Sprintf("PRAGMA user_version=%d", schemaVersion)); err != nil {
			return fmt.Errorf("set schema version: %w", err)
		}
	}
	return nil
}

// addColumnIfMissing runs ALTER TABLE ADD COLUMN only when the column is absent,
// so the (non-idempotent) ALTER is safe to re-run after a crash. table and column
// are compile-time constants from migrate(), never caller input.
func addColumnIfMissing(ctx context.Context, conn *sql.Conn, table, column, decl string) error {
	var n int
	if err := conn.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM pragma_table_info('"+table+"') WHERE name = ?", column).Scan(&n); err != nil {
		return fmt.Errorf("table_info %s: %w", table, err)
	}
	if n > 0 {
		return nil // already present — a crashed prior attempt (or fresh v2 schema) got here first
	}
	if _, err := conn.ExecContext(ctx,
		fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", table, column, decl)); err != nil {
		return fmt.Errorf("add %s.%s: %w", table, column, err)
	}
	return nil
}

// Close closes the database.
func (s *Store) Close() error { return s.db.Close() }

// IncrementalVacuum reclaims free pages left by deleted blobs. Run after GC.
func (s *Store) IncrementalVacuum(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, "PRAGMA incremental_vacuum")
	return err
}

// GCExpiredModels NULLs out the .nam blob AND its resume checkpoint for terminal
// jobs whose finished_at is older than cutoff (unix seconds) — the re-download
// window has closed, and the ckpt shares that window (it IS the continuation
// window). Job rows and job_log stay indefinitely as portable history; only the
// big blobs expire. Returns the number of rows freed. A subsequent GET .../model
// then answers 404 (has_model:false) and the parent is no longer resumable — both
// the client's cue to DELETE + resubmit.
func (s *Store) GCExpiredModels(ctx context.Context, cutoff int64) (int64, error) {
	res, err := s.db.ExecContext(ctx,
		`UPDATE results SET nam = NULL, ckpt = NULL
		 WHERE (nam IS NOT NULL OR ckpt IS NOT NULL) AND job_key IN (
		   SELECT key FROM jobs
		   WHERE state IN ('succeeded','failed') AND finished_at IS NOT NULL AND finished_at < ?)`,
		cutoff)
	if err != nil {
		return 0, fmt.Errorf("gc expired models: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("gc rows: %w", err)
	}
	return n, nil
}
