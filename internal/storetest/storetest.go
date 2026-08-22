// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package storetest opens a PRIVATE schema in the shared dev database for one
// test, applies the app's DDL (the contract, 0001_init.sql) inside it, seeds the
// library rows a job needs (signal → device → combo → shoot → take → take_audio),
// and drops the schema when the test ends. It is imported only by _test files.
//
// Every test is skipped unless ORBITNAM_TEST_PG_DSN is set; the schema is EVERY migration the app
// ships, in version order, read from ORBITNAM_TEST_DDL (a file or a directory) or, by default, from
// the sibling orbit-nam-capture checkout.
// Nothing here ever touches the public schema.
package storetest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"orbit-capture-nam-trainer/internal/store"
)

// Environment variables the harness reads.
const (
	DSNEnv = "ORBITNAM_TEST_PG_DSN" // e.g. "host=… dbname=orbitnam_dev user=orbitnam_dev"
	DDLEnv = "ORBITNAM_TEST_DDL"    // path to 0001_init.sql; default: the sibling app checkout
)

// SignalSHA is the sha256 the seeded training-signal row carries; a pool whose
// profile reports the same sha passes the signal_mismatch check.
const SignalSHA = "70f8ec7f25686a1bd77f25973de8e51a6721e957e81eec121822e5e53366bc41"

// NamVersion is the required_nam_version every seeded job pins (and the profile a
// test pool reports), unless a test overrides it.
const NamVersion = "0.13.0"

var schemaSeq atomic.Int64

// Open creates a private schema, applies the DDL, and returns a Store bound to it
// (search_path pinned on every pooled connection). Skips when DSNEnv is unset.
func Open(t testing.TB) *store.Store {
	t.Helper()
	dsn := os.Getenv(DSNEnv)
	if dsn == "" {
		t.Skipf("skipped: %s unset (no PostgreSQL test database)", DSNEnv)
	}
	ddl := loadDDL(t)
	schema := fmt.Sprintf("trainertest_%d_%d", os.Getpid(), schemaSeq.Add(1))
	ctx := context.Background()

	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		t.Fatalf("storetest: parse %s: %v", DSNEnv, err)
	}
	if cfg.ConnConfig.RuntimeParams == nil {
		cfg.ConnConfig.RuntimeParams = map[string]string{}
	}
	cfg.ConnConfig.RuntimeParams["search_path"] = schema
	st, err := store.OpenConfig(ctx, cfg, 12)
	if err != nil {
		t.Fatalf("storetest: open: %v", err)
	}
	if _, err := st.Pool().Exec(ctx, "CREATE SCHEMA "+schema); err != nil {
		st.Close()
		t.Fatalf("storetest: create schema %s: %v", schema, err)
	}
	t.Cleanup(func() {
		dctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if _, err := st.Pool().Exec(dctx, "DROP SCHEMA "+schema+" CASCADE"); err != nil {
			t.Errorf("storetest: drop schema %s: %v", schema, err)
		}
		st.Close()
	})
	// One multi-statement Exec (simple protocol): every unqualified name lands in
	// the first schema of search_path — ours.
	if _, err := st.Pool().Exec(ctx, ddl); err != nil {
		t.Fatalf("storetest: apply DDL in %s: %v", schema, err)
	}
	// The library row the daemon's heartbeat reads. The FIRST migration creates the singleton itself
	// (a daemon must always have a queue_contract to read, including before any app has opened the
	// library), so this is an update of what is already there — inserting it collided on the primary
	// key and failed every test in this package at setup.
	if _, err := st.Pool().Exec(ctx,
		`INSERT INTO library (id, created_by_app, queue_contract) VALUES (1, 'storetest', $1)
		 ON CONFLICT (id) DO UPDATE SET created_by_app = EXCLUDED.created_by_app,
		                                queue_contract = EXCLUDED.queue_contract`,
		store.SupportedQueueContract); err != nil {
		t.Fatalf("storetest: seed library: %v", err)
	}
	return st
}

// loadDDL reads the contract: EVERY migration the app ships, in version order, concatenated.
//
// It used to read 0001_init.sql alone, which meant this daemon was tested against a schema the app
// stopped producing the moment a second migration existed — green here, and possibly broken against
// the library it actually runs on. The app applies them in order and so does this.
//
// Candidates: DDLEnv (a file OR a directory of them), then the sibling checkout of orbit-nam-capture
// next to this repo (and one level up, for a worktree layout).
func loadDDL(t testing.TB) string {
	t.Helper()
	var candidates []string
	if p := os.Getenv(DDLEnv); p != "" {
		candidates = append(candidates, p)
	}
	_, here, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(here), "..", ".."))
	rel := filepath.Join("orbit-nam-capture", "app", "assets", "migrations")
	candidates = append(candidates,
		filepath.Join(root, "..", rel),
		filepath.Join(root, "..", "..", rel))
	for _, p := range candidates {
		if sql, ok := readMigrations(p); ok {
			return sql
		}
	}
	t.Fatalf("storetest: migrations not found (set %s); tried %v", DDLEnv, candidates)
	return ""
}

// readMigrations reads one .sql file, or every NNNN_*.sql in a directory in version order.
func readMigrations(path string) (string, bool) {
	info, err := os.Stat(path)
	if err != nil {
		return "", false
	}
	if !info.IsDir() {
		b, err := os.ReadFile(path)
		return string(b), err == nil
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		return "", false
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".sql") {
			names = append(names, e.Name())
		}
	}
	if len(names) == 0 {
		return "", false
	}
	sort.Strings(names) // NNNN_ prefixes are zero-padded, so lexical order IS version order
	var sql strings.Builder
	for _, n := range names {
		b, err := os.ReadFile(filepath.Join(path, n))
		if err != nil {
			return "", false
		}
		sql.WriteString("\n-- ---- ")
		sql.WriteString(n)
		sql.WriteString(" ----\n")
		sql.Write(b)
	}
	return sql.String(), true
}

// Take is a seeded take with everything a job row needs to reference it.
type Take struct {
	ID       int64
	DeviceID int64
	ComboID  int64
	ShootID  int64
	SignalID int64
	Label    string // takes.name — what jobs.take_label carries
	WavSHA   string // take_audio.sha256 — what jobs.wav_sha pins
}

var takeSeq atomic.Int64

// SeedTake inserts a signal (upserted by name), a fresh device/combo/shoot and one
// take whose audio is wav VERBATIM. The take's declared frames satisfy the
// library's 30 s..20 min CHECK regardless of the bytes (the store never validates
// them; the worker does, via package wav).
func SeedTake(t testing.TB, st *store.Store, wav []byte) Take {
	t.Helper()
	ctx := context.Background()
	n := takeSeq.Add(1)
	var tk Take
	if err := st.Pool().QueryRow(ctx,
		`INSERT INTO signals (name, kind, sha256, sample_rate, frames) VALUES ('v3_0_0.wav', 'train', $1, 48000, 1)
		 ON CONFLICT (name) DO UPDATE SET sha256 = EXCLUDED.sha256 RETURNING id`, SignalSHA).Scan(&tk.SignalID); err != nil {
		t.Fatalf("seed signal: %v", err)
	}
	if err := st.Pool().QueryRow(ctx,
		`INSERT INTO devices (name, rig_id, signal_id) VALUES ($1, $2, $3) RETURNING id`,
		fmt.Sprintf("Device %d", n), fmt.Sprintf("rig-%d", n), tk.SignalID).Scan(&tk.DeviceID); err != nil {
		t.Fatalf("seed device: %v", err)
	}
	if err := st.Pool().QueryRow(ctx,
		`INSERT INTO combos (device_id, settings) VALUES ($1, '{}') RETURNING id`, tk.DeviceID).Scan(&tk.ComboID); err != nil {
		t.Fatalf("seed combo: %v", err)
	}
	if err := st.Pool().QueryRow(ctx,
		`INSERT INTO shoots (device_id, started_at) VALUES ($1, now()) RETURNING id`, tk.DeviceID).Scan(&tk.ShootID); err != nil {
		t.Fatalf("seed shoot: %v", err)
	}
	tk.Label = fmt.Sprintf("Take %d-0001", n)
	if err := st.Pool().QueryRow(ctx,
		`INSERT INTO takes (device_id, combo_id, shoot_id, name, serial, version, recorded_at, signal_id,
		                    sample_rate, bits, frames, peak_dbfs)
		 VALUES ($1, $2, $3, $4, 1, 1, now(), $5, 48000, 16, 1440000, -20.0::double precision) RETURNING id`,
		tk.DeviceID, tk.ComboID, tk.ShootID, tk.Label, tk.SignalID).Scan(&tk.ID); err != nil {
		t.Fatalf("seed take: %v", err)
	}
	sum := sha256.Sum256(wav)
	tk.WavSHA = hex.EncodeToString(sum[:])
	if _, err := st.Pool().Exec(ctx,
		`INSERT INTO take_audio (take_id, sha256, size, bytes) VALUES ($1, $2, $3, $4)`,
		tk.ID, tk.WavSHA, len(wav), wav); err != nil {
		t.Fatalf("seed take_audio: %v", err)
	}
	return tk
}

// JobSpec is what the APP writes when it queues a job (the request block).
type JobSpec struct {
	Take       Take
	Kind       string // default train
	Epochs     int    // default 100 (probe_self: 1)
	Arch       string // default standard
	Priority   string // default normal
	Latency    *int64
	NamVersion string // default NamVersion
	BaseJobID  *int64 // train_more: provenance
	StartEpoch *int64 // train_more: required by the schema
	QueuedAt   *time.Time
}

// InsertJob queues a job as the app would and returns its id.
func InsertJob(t testing.TB, st *store.Store, s JobSpec) int64 {
	t.Helper()
	if s.Kind == "" {
		s.Kind = "train"
	}
	if s.Epochs == 0 {
		s.Epochs = 100
	}
	if s.Arch == "" {
		s.Arch = "standard"
	}
	if s.Priority == "" {
		s.Priority = "normal"
	}
	if s.NamVersion == "" {
		s.NamVersion = NamVersion
	}
	queuedAt := time.Now()
	if s.QueuedAt != nil {
		queuedAt = *s.QueuedAt
	}
	var id int64
	if err := st.Pool().QueryRow(context.Background(),
		`INSERT INTO jobs (take_id, take_label, kind, epochs, arch, priority, latency_samples, wav_sha,
		                   required_nam_version, base_job_id, start_epoch, queued_at)
		 VALUES ($1, $2, $3, $4, $5, $6::priority, $7, $8, $9, $10, $11, $12) RETURNING id`,
		s.Take.ID, s.Take.Label, s.Kind, s.Epochs, s.Arch, s.Priority, s.Latency, s.Take.WavSHA,
		s.NamVersion, s.BaseJobID, s.StartEpoch, queuedAt).Scan(&id); err != nil {
		t.Fatalf("insert job: %v", err)
	}
	return id
}

// InsertResumeFromResult snapshots the base job's job_result.ckpt into job_resume
// for the child — the server-side INSERT…SELECT the app runs when it queues a
// train_more.
func InsertResumeFromResult(t testing.TB, st *store.Store, childID, baseID int64) {
	t.Helper()
	tag, err := st.Pool().Exec(context.Background(),
		`INSERT INTO job_resume (job_id, base_job_id, ckpt_sha256, ckpt_size, ckpt)
		 SELECT $1, $2, encode(sha256(ckpt), 'hex'), ckpt_size, ckpt FROM job_result
		 WHERE job_id = $2 AND ckpt IS NOT NULL`, childID, baseID)
	if err != nil {
		t.Fatalf("insert job_resume: %v", err)
	}
	if tag.RowsAffected() != 1 {
		t.Fatalf("insert job_resume: base %d has no checkpoint", baseID)
	}
}

// InsertResumeBytes stores an explicit checkpoint for a train_more child.
func InsertResumeBytes(t testing.TB, st *store.Store, childID, baseID int64, ckpt []byte) {
	t.Helper()
	sum := sha256.Sum256(ckpt)
	if _, err := st.Pool().Exec(context.Background(),
		`INSERT INTO job_resume (job_id, base_job_id, ckpt_sha256, ckpt_size, ckpt) VALUES ($1, $2, $3, $4, $5)`,
		childID, baseID, hex.EncodeToString(sum[:]), len(ckpt), ckpt); err != nil {
		t.Fatalf("insert job_resume bytes: %v", err)
	}
}

// Result returns the job_result row's bytes; ok=false when there is none.
func Result(t testing.TB, st *store.Store, id int64) (nam, ckpt []byte, ok bool) {
	t.Helper()
	rows, err := st.Pool().Query(context.Background(), `SELECT bytes, ckpt FROM job_result WHERE job_id = $1`, id)
	if err != nil {
		t.Fatalf("read job_result: %v", err)
	}
	defer rows.Close()
	if !rows.Next() {
		return nil, nil, false
	}
	if err := rows.Scan(&nam, &ckpt); err != nil {
		t.Fatalf("scan job_result: %v", err)
	}
	return nam, ckpt, true
}

// Count runs a COUNT(*) query with args.
func Count(t testing.TB, st *store.Store, sql string, args ...any) int {
	t.Helper()
	var n int
	if err := st.Pool().QueryRow(context.Background(), sql, args...).Scan(&n); err != nil {
		t.Fatalf("count (%s): %v", sql, err)
	}
	return n
}

// Exec runs a statement, failing the test on error.
func Exec(t testing.TB, st *store.Store, sql string, args ...any) {
	t.Helper()
	if _, err := st.Pool().Exec(context.Background(), sql, args...); err != nil {
		t.Fatalf("exec (%s): %v", sql, err)
	}
}
