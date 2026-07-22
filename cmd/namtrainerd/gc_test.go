// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package main

import (
	"context"
	"os"
	"strings"
	"testing"

	"orbit-capture-nam-trainer/internal/applog"
	"orbit-capture-nam-trainer/internal/config"
	"orbit-capture-nam-trainer/internal/store"
)

// gcTestRig builds a config + story log + store over a throwaway base dir and seeds
// one old succeeded train with a stored model, finished long ago (finished_at=100).
func gcTestRig(t *testing.T, retentionDays int) (*config.Config, *applog.Logger, *store.Store) {
	t.Helper()
	ctx := context.Background()
	base := t.TempDir()
	cfg, err := config.Load(base)
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	cfg.RetentionDays = retentionDays

	lg, err := applog.Open(cfg.LogPath())
	if err != nil {
		t.Fatalf("applog.Open: %v", err)
	}
	t.Cleanup(func() { lg.Close() })

	st, err := store.Open(ctx, cfg.DBPath())
	if err != nil {
		t.Fatalf("store.Open: %v", err)
	}
	t.Cleanup(func() { st.Close() })

	for _, stmt := range []string{
		`INSERT INTO jobs(key,kind,state,epochs,arch,created_at,finished_at) VALUES('old','train','succeeded',100,'standard',1,100)`,
		`INSERT INTO results(job_key,nam) VALUES('old',x'cafe')`,
	} {
		if _, err := st.DB().ExecContext(ctx, stmt); err != nil {
			t.Fatalf("seed (%.40s): %v", stmt, err)
		}
	}
	return cfg, lg, st
}

func logBody(t *testing.T, cfg *config.Config) string {
	t.Helper()
	b, err := os.ReadFile(cfg.LogPath())
	if err != nil {
		t.Fatalf("read log: %v", err)
	}
	return string(b)
}

func TestGCPassRetentionZeroSkipsExpiryVacuumsAlways(t *testing.T) {
	ctx := context.Background()
	cfg, lg, st := gcTestRig(t, 0)

	gcPass(ctx, cfg, lg, st)

	// retention_days == 0 (keep forever): the ancient model must survive — the GC
	// step is skipped entirely.
	if _, ok, _ := st.ModelBytes(ctx, "old"); !ok {
		t.Error("retention 0 must keep the model forever (GCExpiredModels skipped)")
	}
	// The vacuum ALWAYS runs and logs, so the behavior is observable; no
	// GC/retention line may appear.
	body := logBody(t, cfg)
	if !strings.Contains(body, "incremental vacuum done") {
		t.Errorf("vacuum must always run and log — missing from story log:\n%s", body)
	}
	if strings.Contains(body, "retention") {
		t.Errorf("no GC/retention line may appear at retention 0:\n%s", body)
	}
}

func TestGCPassPositiveRetentionExpiresAndVacuums(t *testing.T) {
	ctx := context.Background()
	cfg, lg, st := gcTestRig(t, 30)

	gcPass(ctx, cfg, lg, st)

	// The ancient model (finished_at=100) is far past a 30-day window → freed.
	if _, ok, _ := st.ModelBytes(ctx, "old"); ok {
		t.Error("retention 30 must expire the ancient model")
	}
	body := logBody(t, cfg)
	if !strings.Contains(body, "expired 1 model blob") {
		t.Errorf("expected the expiry line at retention 30:\n%s", body)
	}
	if !strings.Contains(body, "incremental vacuum done") {
		t.Errorf("vacuum must also run at retention 30:\n%s", body)
	}
}
