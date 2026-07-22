// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"bytes"
	"context"
	"database/sql"
	"path/filepath"
	"strings"
	"testing"

	"orbit-capture-nam-trainer/internal/jobs"
)

func openTest(t *testing.T) *Store {
	t.Helper()
	st, err := Open(context.Background(), filepath.Join(t.TempDir(), "trainer.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { st.Close() })
	return st
}

func TestOpenAppliesPragmas(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	var journal string
	if err := st.db.QueryRowContext(ctx, "PRAGMA journal_mode").Scan(&journal); err != nil {
		t.Fatalf("journal_mode: %v", err)
	}
	if !strings.EqualFold(journal, "wal") {
		t.Errorf("journal_mode = %q, want wal", journal)
	}

	var autovac int
	if err := st.db.QueryRowContext(ctx, "PRAGMA auto_vacuum").Scan(&autovac); err != nil {
		t.Fatalf("auto_vacuum: %v", err)
	}
	if autovac != 2 { // 2 = INCREMENTAL
		t.Errorf("auto_vacuum = %d, want 2 (INCREMENTAL)", autovac)
	}

	var fk int
	if err := st.db.QueryRowContext(ctx, "PRAGMA foreign_keys").Scan(&fk); err != nil {
		t.Fatalf("foreign_keys: %v", err)
	}
	if fk != 1 {
		t.Errorf("foreign_keys = %d, want 1", fk)
	}
}

func TestSchemaTablesExist(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	for _, tbl := range []string{"jobs", "audio_blobs", "results", "resume_ckpts", "job_log"} {
		var name string
		err := st.db.QueryRowContext(ctx,
			"SELECT name FROM sqlite_master WHERE type='table' AND name=?", tbl).Scan(&name)
		if err != nil {
			t.Errorf("table %q missing: %v", tbl, err)
		}
	}
}

// v1Schema is the base schema exactly as it shipped at user_version=1 — before the
// train_more (v2) columns and the resume_ckpts table. The migration test builds a
// populated database at this shape and reopens it with the current code, proving an
// existing field database is carried to v2 without data loss.
const v1Schema = `
CREATE TABLE jobs (
  key TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 1, epochs INTEGER NOT NULL,
  arch TEXT NOT NULL DEFAULT 'standard', created_at INTEGER NOT NULL,
  started_at INTEGER, finished_at INTEGER, pid INTEGER,
  epoch INTEGER, s_per_epoch REAL, verdict TEXT, esr REAL,
  error_code TEXT, error_msg TEXT);
CREATE TABLE audio_blobs (job_key TEXT PRIMARY KEY REFERENCES jobs(key) ON DELETE CASCADE, content BLOB NOT NULL);
CREATE TABLE results (job_key TEXT PRIMARY KEY REFERENCES jobs(key) ON DELETE CASCADE, nam BLOB, train_json TEXT);
CREATE TABLE job_log (id INTEGER PRIMARY KEY, job_key TEXT NOT NULL REFERENCES jobs(key) ON DELETE CASCADE, line TEXT NOT NULL);
`

func TestMigrateV1FileToCurrentSchema(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "trainer.db")

	// Build a populated v1 database by hand (no train_more columns, no reached,
	// user_version=1).
	raw, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatalf("open v1: %v", err)
	}
	for _, stmt := range []string{
		v1Schema,
		`INSERT INTO jobs(key,kind,state,priority,epochs,arch,created_at,finished_at,epoch,s_per_epoch)
		 VALUES('old','train','succeeded',1,120,'standard',10,100,119,3.5)`,
		`INSERT INTO results(job_key,nam,train_json) VALUES('old',x'cafe','{"esr":0.02}')`,
		`INSERT INTO audio_blobs(job_key,content) VALUES('old',x'0011')`,
		`INSERT INTO job_log(job_key,line) VALUES('old','trained')`,
		`PRAGMA user_version=1`,
	} {
		if _, err := raw.ExecContext(ctx, stmt); err != nil {
			t.Fatalf("seed v1 (%.40s): %v", stmt, err)
		}
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close v1: %v", err)
	}

	// Reopen with the current code: migrate() must carry the file all the way to v3.
	st, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open (migrate): %v", err)
	}
	t.Cleanup(func() { st.Close() })

	var uv int
	if err := st.db.QueryRowContext(ctx, "PRAGMA user_version").Scan(&uv); err != nil {
		t.Fatalf("user_version: %v", err)
	}
	if uv != 3 {
		t.Errorf("user_version = %d, want 3 after migration", uv)
	}

	// The v1 data survived intact, and its new columns default to NULL.
	old, ok, err := st.GetJob(ctx, "old")
	if err != nil || !ok {
		t.Fatalf("GetJob(old): ok=%v err=%v", ok, err)
	}
	if old.Epochs != 120 || old.State != jobs.StateSucceeded || !old.HasModel {
		t.Errorf("migrated job mismatch: %+v", old)
	}
	if old.WavSHA != nil || old.BaseKey != nil || old.StartEpoch != nil {
		t.Errorf("pre-v2 row must have NULL train_more columns: %+v", old)
	}
	if old.Reached != nil {
		t.Errorf("pre-v3 row must have NULL reached (finished by a daemon predating stop): %v", old.Reached)
	}
	if nam, ok, _ := st.ModelBytes(ctx, "old"); !ok || !bytes.Equal(nam, []byte{0xca, 0xfe}) {
		t.Errorf("model blob = %x (ok=%v), want cafe", nam, ok)
	}
	if lines, _ := st.JobLog(ctx, "old"); len(lines) != 1 || lines[0] != "trained" {
		t.Errorf("job log = %v, want [trained]", lines)
	}

	// The v2 machinery is fully usable on the migrated file: results.ckpt is
	// writable and resume_ckpts exists (a train_more inserts against a fresh parent).
	wav := []byte("migrated-capture")
	makeSucceededParent(t, st, "p", 200, "standard", wav, []byte("ckpt-p"))
	child := trainMoreChild("cm", "p", 400, "standard")
	if err := st.InsertJob(ctx, child, wav); err != nil {
		t.Fatalf("train_more on migrated DB: %v", err)
	}
	if snap, ok, _ := st.ResumeCkpt(ctx, "cm"); !ok || !bytes.Equal(snap, []byte("ckpt-p")) {
		t.Errorf("resume_ckpts snapshot = %q (ok=%v), want ckpt-p", snap, ok)
	}
}

func TestMigrateResumesAfterCrashMidV2Step(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "trainer.db")

	// A v1 file where a previous migration attempt crashed PARTWAY through the v2
	// step: two of the four columns already added, resume_ckpts already created,
	// user_version still 1. SQLite DDL has no transactional rollback, so this
	// half-state is reachable; reopening must complete the step (guarded ALTERs skip
	// what exists) AND carry on through v3, not fail on a duplicate column.
	raw, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatalf("open v1: %v", err)
	}
	for _, stmt := range []string{
		v1Schema,
		`ALTER TABLE jobs ADD COLUMN wav_sha TEXT`,
		`ALTER TABLE results ADD COLUMN ckpt BLOB`,
		`CREATE TABLE resume_ckpts (job_key TEXT PRIMARY KEY REFERENCES jobs(key) ON DELETE CASCADE, content BLOB NOT NULL)`,
		`PRAGMA user_version=1`,
	} {
		if _, err := raw.ExecContext(ctx, stmt); err != nil {
			t.Fatalf("seed half-migrated v1 (%.40s): %v", stmt, err)
		}
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	st, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open after crash-mid-migration: %v", err)
	}
	t.Cleanup(func() { st.Close() })

	var uv int
	if err := st.db.QueryRowContext(ctx, "PRAGMA user_version").Scan(&uv); err != nil {
		t.Fatalf("user_version: %v", err)
	}
	if uv != 3 {
		t.Errorf("user_version = %d, want 3", uv)
	}
	// The columns the crashed attempt had NOT reached exist now (train_more works,
	// and reached stamps on a natural finish).
	wav := []byte("post-crash-capture")
	makeSucceededParent(t, st, "p", 100, "standard", wav, []byte("ck"))
	if got, _, _ := st.GetJob(ctx, "p"); got.Reached == nil || *got.Reached != 100 {
		t.Errorf("reached = %v, want 100 (column present after repair)", got.Reached)
	}
	if err := st.InsertJob(ctx, trainMoreChild("c", "p", 200, "standard"), wav); err != nil {
		t.Fatalf("train_more on repaired DB: %v", err)
	}
}

func TestMigrateV2FileToV3(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "trainer.db")

	// Build a populated database at the exact v2 shape: the train_more columns +
	// resume_ckpts, a succeeded train row, user_version=2, and NO reached column.
	raw, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatalf("open v2: %v", err)
	}
	for _, stmt := range []string{
		v1Schema,
		`ALTER TABLE jobs ADD COLUMN wav_sha TEXT`,
		`ALTER TABLE jobs ADD COLUMN base_key TEXT`,
		`ALTER TABLE jobs ADD COLUMN start_epoch INTEGER`,
		`ALTER TABLE results ADD COLUMN ckpt BLOB`,
		`CREATE TABLE resume_ckpts (job_key TEXT PRIMARY KEY REFERENCES jobs(key) ON DELETE CASCADE, content BLOB NOT NULL)`,
		`INSERT INTO jobs(key,kind,state,priority,epochs,arch,created_at,finished_at,epoch,s_per_epoch,wav_sha)
		 VALUES('v2row','train','succeeded',1,150,'standard',10,200,149,3.0,'abc')`,
		`INSERT INTO results(job_key,nam,train_json,ckpt) VALUES('v2row',x'cafe','{}',x'beef')`,
		`PRAGMA user_version=2`,
	} {
		if _, err := raw.ExecContext(ctx, stmt); err != nil {
			t.Fatalf("seed v2 (%.40s): %v", stmt, err)
		}
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close v2: %v", err)
	}

	st, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open (migrate v2→v3): %v", err)
	}
	t.Cleanup(func() { st.Close() })

	var uv int
	if err := st.db.QueryRowContext(ctx, "PRAGMA user_version").Scan(&uv); err != nil {
		t.Fatalf("user_version: %v", err)
	}
	if uv != 3 {
		t.Errorf("user_version = %d, want 3 after v2→v3 migration", uv)
	}
	// The pre-v3 row's reached is NULL — a row finished by a daemon predating stop.
	got, ok, err := st.GetJob(ctx, "v2row")
	if err != nil || !ok {
		t.Fatalf("GetJob(v2row): ok=%v err=%v", ok, err)
	}
	if got.Reached != nil {
		t.Errorf("pre-v3 v2 row reached = %v, want NULL", got.Reached)
	}
	// reached is writable on the migrated file: a fresh natural finish stamps it.
	makeSucceededParent(t, st, "fresh", 80, "standard", []byte("w"), []byte("ck"))
	if fresh, _, _ := st.GetJob(ctx, "fresh"); fresh.Reached == nil || *fresh.Reached != 80 {
		t.Errorf("fresh reached = %v, want 80", fresh.Reached)
	}
}

func TestMigrateResumesAfterCrashMidV3Step(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "trainer.db")

	// A v2 file where the v3 ALTER already ran but the user_version bump did not
	// (crash between ADD COLUMN reached and PRAGMA user_version=3). Reopening must
	// skip the already-present column (guarded ALTER) and reach v3, not fail.
	raw, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatalf("open v2: %v", err)
	}
	for _, stmt := range []string{
		v1Schema,
		`ALTER TABLE jobs ADD COLUMN wav_sha TEXT`,
		`ALTER TABLE jobs ADD COLUMN base_key TEXT`,
		`ALTER TABLE jobs ADD COLUMN start_epoch INTEGER`,
		`ALTER TABLE results ADD COLUMN ckpt BLOB`,
		`CREATE TABLE resume_ckpts (job_key TEXT PRIMARY KEY REFERENCES jobs(key) ON DELETE CASCADE, content BLOB NOT NULL)`,
		`ALTER TABLE jobs ADD COLUMN reached INTEGER`, // the v3 step already ran
		`PRAGMA user_version=2`,                       // but the version bump did not
	} {
		if _, err := raw.ExecContext(ctx, stmt); err != nil {
			t.Fatalf("seed half-migrated v2 (%.40s): %v", stmt, err)
		}
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	st, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open after crash-mid-v3-migration: %v", err)
	}
	t.Cleanup(func() { st.Close() })

	var uv int
	if err := st.db.QueryRowContext(ctx, "PRAGMA user_version").Scan(&uv); err != nil {
		t.Fatalf("user_version: %v", err)
	}
	if uv != 3 {
		t.Errorf("user_version = %d, want 3", uv)
	}
	// The guarded ALTER did not double-add and error; reached is usable.
	makeSucceededParent(t, st, "p", 40, "standard", []byte("w"), []byte("ck"))
	if got, _, _ := st.GetJob(ctx, "p"); got.Reached == nil || *got.Reached != 40 {
		t.Errorf("reached = %v, want 40", got.Reached)
	}
}

func TestForeignKeyCascadeDeletesChildren(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	db := st.db

	_, err := db.ExecContext(ctx,
		"INSERT INTO jobs(key,kind,state,epochs,created_at) VALUES('k','train','queued',10,1)")
	if err != nil {
		t.Fatalf("insert job: %v", err)
	}
	if _, err := db.ExecContext(ctx,
		"INSERT INTO audio_blobs(job_key,content) VALUES('k',x'00')"); err != nil {
		t.Fatalf("insert blob: %v", err)
	}
	if _, err := db.ExecContext(ctx,
		"INSERT INTO job_log(job_key,line) VALUES('k','hello')"); err != nil {
		t.Fatalf("insert log: %v", err)
	}

	if _, err := db.ExecContext(ctx, "DELETE FROM jobs WHERE key='k'"); err != nil {
		t.Fatalf("delete job: %v", err)
	}

	for _, q := range []string{
		"SELECT COUNT(*) FROM audio_blobs WHERE job_key='k'",
		"SELECT COUNT(*) FROM job_log WHERE job_key='k'",
	} {
		var n int
		if err := db.QueryRowContext(ctx, q).Scan(&n); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
		if n != 0 {
			t.Errorf("%s = %d, want 0 (cascade)", q, n)
		}
	}
}
