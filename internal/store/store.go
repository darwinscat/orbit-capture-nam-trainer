// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package store is the daemon's view of the SHARED PostgreSQL library + queue
// (orbit-nam-capture, app/assets/migrations/0001_init.sql). The daemon owns no
// schema: it touches ONLY workers, jobs (its run columns), job_log, job_result,
// job_snapshots, reads job_resume, and READS take_audio / signals of the take it is
// pointed at. Every write after a claim is fenced on (id, claim_token,
// state='running') so a straggler of an earlier attempt can never write onto a
// newer one. That fence — and the closed vocabularies in package jobs — are the
// whole contract; the daemon changes only when a column does.
package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// SupportedQueueContract is the library.queue_contract this build was written
// for. The heartbeat compares it with the live value on every beat and reports
// ready=false (with a note) when the library moved past it — the daemon then stops
// claiming but lets running jobs finish.
const SupportedQueueContract = 1

// Store wraps the connection pool.
type Store struct {
	pool *pgxpool.Pool
}

// Open connects to the shared database. maxConns bounds the pool (the train lane
// width plus a few for the heartbeat, the tray and the per-job control pollers).
func Open(ctx context.Context, dsn string, maxConns int32) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("parse dsn: %w", err)
	}
	return OpenConfig(ctx, cfg, maxConns)
}

// OpenConfig is Open for a prepared pool config (tests pin search_path on it).
func OpenConfig(ctx context.Context, cfg *pgxpool.Config, maxConns int32) (*Store, error) {
	if maxConns < 2 {
		maxConns = 2
	}
	cfg.MaxConns = maxConns
	cfg.MinConns = 1
	cfg.MaxConnIdleTime = 5 * time.Minute
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("open pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping database: %w", err)
	}
	return &Store{pool: pool}, nil
}

// Pool exposes the underlying pool for callers that run their own queries (tests).
func (s *Store) Pool() *pgxpool.Pool { return s.pool }

// Close releases the pool.
func (s *Store) Close() { s.pool.Close() }

// QueueContract reads library.queue_contract — the version of the daemon's tables
// the library is at. A missing library row is an error (the database was never
// initialised by the app).
func (s *Store) QueueContract(ctx context.Context) (int, error) {
	var v int
	err := s.pool.QueryRow(ctx, `SELECT queue_contract FROM library WHERE id = 1`).Scan(&v)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, errors.New("library row missing (database not initialised by the app)")
	}
	if err != nil {
		return 0, fmt.Errorf("read queue_contract: %w", err)
	}
	return v, nil
}

// isUniqueViolation reports whether err is a PostgreSQL unique_violation (23505).
func isUniqueViolation(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "23505"
}

// ClaimIdentity takes a session-scoped advisory lock on this daemon's worker name
// and HOLDS it for the process lifetime, on a connection of its own.
//
// The worker name is the hostname (ONCT_WORKER_NAME overrides it, for verification
// runs that need a second claimant on one box), and everything the daemon owns hangs
// off it: the workers row, the recovery sweep that requeues "my" running jobs at boot,
// the cap and pause the app asks of it. Two daemons sharing ONE NAME would therefore
// requeue each other's live runs on startup and fight over one row's cap forever.
// There is no column that can express that — the second process would simply overwrite
// the first. So it does not start: this returns ok=false and the caller says so.
//
// Two daemons with DIFFERENT names are a different thing entirely, and a legitimate
// one: each owns its own row, and each sweeps only its own jobs.
//
// The lock is database-wide rather than schema-scoped, which is what makes it a real
// answer — a private schema for a test run does not hide a name collision.
//
// The lock is session-scoped, so a killed daemon releases it the moment PostgreSQL
// notices the connection is gone — no stale lock survives a crash.
func (s *Store) ClaimIdentity(ctx context.Context, name string) (release func(), ok bool, err error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, false, fmt.Errorf("acquire identity conn: %w", err)
	}
	var got bool
	if err := conn.QueryRow(ctx,
		`SELECT pg_try_advisory_lock(hashtext('orbitnam.worker:' || $1))`, name).Scan(&got); err != nil {
		conn.Release()
		return nil, false, fmt.Errorf("claim identity: %w", err)
	}
	if !got {
		conn.Release()
		return nil, false, nil
	}
	return func() { conn.Release() }, true, nil
}
