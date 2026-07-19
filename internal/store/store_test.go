// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
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
	for _, tbl := range []string{"jobs", "audio_blobs", "results", "job_log"} {
		var name string
		err := st.db.QueryRowContext(ctx,
			"SELECT name FROM sqlite_master WHERE type='table' AND name=?", tbl).Scan(&name)
		if err != nil {
			t.Errorf("table %q missing: %v", tbl, err)
		}
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
