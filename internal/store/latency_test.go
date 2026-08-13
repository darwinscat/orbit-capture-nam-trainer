// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"database/sql"
	"errors"
	"path/filepath"
	"testing"

	"orbit-capture-nam-trainer/internal/jobs"
)

func lat(n int64) *int64 { return &n }

// withLatency returns j with its latency set (nil leaves it absent).
func withLatency(j jobs.Job, latency *int64) jobs.Job {
	j.Latency = latency
	return j
}

// TestInsertJobRoundTripsLatency pins the column: a supplied latency comes back
// verbatim, an absent one stays NULL, and 0 is a stored value — not absence.
func TestInsertJobRoundTripsLatency(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	cases := map[string]*int64{"with": lat(1101), "zero": lat(0), "absent": nil}
	for key, want := range cases {
		j := jobs.Job{Key: key, Kind: jobs.KindTrain, State: jobs.StateQueued,
			Priority: 1, Epochs: 20, Arch: "standard", CreatedAt: 1, Latency: want}
		if err := st.InsertJob(ctx, j, []byte("capture-"+key)); err != nil {
			t.Fatalf("InsertJob(%s): %v", key, err)
		}
		got, ok, err := st.GetJob(ctx, key)
		if err != nil || !ok {
			t.Fatalf("GetJob(%s): ok=%v err=%v", key, ok, err)
		}
		if !jobs.SameLatency(got.Latency, want) {
			t.Errorf("%s: latency = %v, want %v", key, deref(got.Latency), deref(want))
		}
	}
}

// TestClaimCarriesLatency: the worker learns the latency from the CLAIM, not from
// the PUT, so a job picked up after a daemon restart re-spawns with the same
// --latency it was submitted with.
func TestClaimCarriesLatency(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	must(st.InsertJob(ctx, jobs.Job{Key: "k", Kind: jobs.KindTrain, State: jobs.StateQueued,
		Priority: 1, Epochs: 20, Arch: "standard", CreatedAt: 1, Latency: lat(1101)}, []byte("w")))

	j, ok, err := st.ClaimNextQueued(ctx, 1, jobs.KindTrain)
	if err != nil || !ok {
		t.Fatalf("ClaimNextQueued: ok=%v err=%v", ok, err)
	}
	if j.Latency == nil || *j.Latency != 1101 {
		t.Errorf("claimed latency = %v, want 1101", j.Latency)
	}
}

// TestTrainMoreRequiresMatchingLatency is the eligibility rule: a continuation may
// only ride a parent trained on the SAME alignment, and absence counts as a value
// (an auto-detected parent has an unknowable trim baked into its weights).
func TestTrainMoreRequiresMatchingLatency(t *testing.T) {
	ctx := context.Background()
	wav := []byte("shared-capture-bytes")
	ckpt := []byte("parent-checkpoint-blob")

	cases := []struct {
		name          string
		parent, child *int64
		wantErr       bool
	}{
		{"both absent", nil, nil, false},
		{"equal", lat(1101), lat(1101), false},
		{"equal zero", lat(0), lat(0), false},
		{"different", lat(1101), lat(1102), true},
		{"parent absent, child set", nil, lat(1101), true},
		{"parent set, child absent", lat(1101), nil, true},
		{"zero vs absent", lat(0), nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := openTest(t)
			// The parent is seeded through the normal insert+finish path so its latency
			// lands in the same column the eligibility check reads.
			p := jobs.Job{Key: "parent", Kind: jobs.KindTrain, State: jobs.StateQueued,
				Priority: 1, Epochs: 200, Arch: "standard", CreatedAt: 1, Latency: tc.parent}
			if err := st.InsertJob(ctx, p, wav); err != nil {
				t.Fatalf("insert parent: %v", err)
			}
			setRunning(t, st, "parent", 1)
			if ok, err := st.FinishTrainSuccess(ctx, "parent", 100, []byte("nam"), "{}", nil, ckpt); err != nil || !ok {
				t.Fatalf("finish parent: ok=%v err=%v", ok, err)
			}

			child := withLatency(trainMoreChild("child", "parent", 400, "standard"), tc.child)
			err := st.InsertJob(ctx, child, wav)
			if tc.wantErr {
				if !errors.Is(err, ErrBaseUnavailable) {
					t.Fatalf("InsertJob = %v, want ErrBaseUnavailable", err)
				}
				var bu *BaseUnavailableError
				if !errors.As(err, &bu) || bu.Reason != "latency does not match the parent" {
					t.Errorf("reason = %v, want latency does not match the parent", err)
				}
				// The rollback left nothing behind — no child row, no snapshot.
				if _, ok, _ := st.GetJob(ctx, "child"); ok {
					t.Error("child row survived an ineligible parent")
				}
				if _, ok, _ := st.ResumeCkpt(ctx, "child"); ok {
					t.Error("resume snapshot survived an ineligible parent")
				}
				return
			}
			if err != nil {
				t.Fatalf("InsertJob = %v, want success", err)
			}
			if _, ok, _ := st.ResumeCkpt(ctx, "child"); !ok {
				t.Error("eligible parent left no resume snapshot")
			}
		})
	}
}

// TestMigrateV3FileToV4 carries a populated v3 file (no latency column) forward: the
// pre-v4 row reads back NULL — it was trained on the trainer's own auto-detect — and
// the column is writable on the migrated file.
func TestMigrateV3FileToV4(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "trainer.db")

	raw, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatalf("open v3: %v", err)
	}
	for _, stmt := range []string{
		v1Schema,
		`ALTER TABLE jobs ADD COLUMN wav_sha TEXT`,
		`ALTER TABLE jobs ADD COLUMN base_key TEXT`,
		`ALTER TABLE jobs ADD COLUMN start_epoch INTEGER`,
		`ALTER TABLE jobs ADD COLUMN reached INTEGER`,
		`ALTER TABLE results ADD COLUMN ckpt BLOB`,
		`CREATE TABLE resume_ckpts (job_key TEXT PRIMARY KEY REFERENCES jobs(key) ON DELETE CASCADE, content BLOB NOT NULL)`,
		`INSERT INTO jobs(key,kind,state,priority,epochs,arch,created_at,finished_at,epoch,s_per_epoch,wav_sha,reached)
		 VALUES('v3row','train','succeeded',1,150,'standard',10,200,149,3.0,'abc',150)`,
		`INSERT INTO results(job_key,nam,train_json,ckpt) VALUES('v3row',x'cafe','{}',x'beef')`,
		`PRAGMA user_version=3`,
	} {
		if _, err := raw.ExecContext(ctx, stmt); err != nil {
			t.Fatalf("seed v3 (%.40s): %v", stmt, err)
		}
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close v3: %v", err)
	}

	st, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open (migrate v3→v4): %v", err)
	}
	t.Cleanup(func() { st.Close() })

	var uv int
	if err := st.db.QueryRowContext(ctx, "PRAGMA user_version").Scan(&uv); err != nil {
		t.Fatalf("user_version: %v", err)
	}
	if uv != schemaVersion {
		t.Errorf("user_version = %d, want %d after v3→v4 migration", uv, schemaVersion)
	}
	got, ok, err := st.GetJob(ctx, "v3row")
	if err != nil || !ok {
		t.Fatalf("GetJob(v3row): ok=%v err=%v", ok, err)
	}
	if got.Latency != nil {
		t.Errorf("pre-v4 row latency = %v, want NULL (the trainer auto-detected)", *got.Latency)
	}
	if got.Reached == nil || *got.Reached != 150 {
		t.Errorf("migrated row lost reached: %v", got.Reached)
	}
	// The new column is usable on the migrated file.
	fresh := jobs.Job{Key: "fresh", Kind: jobs.KindTrain, State: jobs.StateQueued,
		Priority: 1, Epochs: 20, Arch: "standard", CreatedAt: 3, Latency: lat(1101)}
	if err := st.InsertJob(ctx, fresh, []byte("w")); err != nil {
		t.Fatalf("insert on migrated file: %v", err)
	}
	if j, _, _ := st.GetJob(ctx, "fresh"); j.Latency == nil || *j.Latency != 1101 {
		t.Errorf("fresh latency = %v, want 1101", j.Latency)
	}
}

// TestMigrateResumesAfterCrashMidV4Step reopens a v3 file whose v4 ALTER already ran
// but whose version bump did not — the guarded ALTER must skip, not error.
func TestMigrateResumesAfterCrashMidV4Step(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "trainer.db")

	raw, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatalf("open v3: %v", err)
	}
	for _, stmt := range []string{
		v1Schema,
		`ALTER TABLE jobs ADD COLUMN wav_sha TEXT`,
		`ALTER TABLE jobs ADD COLUMN base_key TEXT`,
		`ALTER TABLE jobs ADD COLUMN start_epoch INTEGER`,
		`ALTER TABLE jobs ADD COLUMN reached INTEGER`,
		`ALTER TABLE results ADD COLUMN ckpt BLOB`,
		`CREATE TABLE resume_ckpts (job_key TEXT PRIMARY KEY REFERENCES jobs(key) ON DELETE CASCADE, content BLOB NOT NULL)`,
		`ALTER TABLE jobs ADD COLUMN latency INTEGER`, // the v4 step already ran
		`PRAGMA user_version=3`,                       // but the version bump did not
	} {
		if _, err := raw.ExecContext(ctx, stmt); err != nil {
			t.Fatalf("seed half-migrated v3 (%.40s): %v", stmt, err)
		}
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	st, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open after crash-mid-v4-migration: %v", err)
	}
	t.Cleanup(func() { st.Close() })

	var uv int
	if err := st.db.QueryRowContext(ctx, "PRAGMA user_version").Scan(&uv); err != nil {
		t.Fatalf("user_version: %v", err)
	}
	if uv != schemaVersion {
		t.Errorf("user_version = %d, want %d", uv, schemaVersion)
	}
	j := jobs.Job{Key: "k", Kind: jobs.KindTrain, State: jobs.StateQueued,
		Priority: 1, Epochs: 20, Arch: "standard", CreatedAt: 1, Latency: lat(7)}
	if err := st.InsertJob(ctx, j, []byte("w")); err != nil {
		t.Fatalf("insert after repair: %v", err)
	}
	if got, _, _ := st.GetJob(ctx, "k"); got.Latency == nil || *got.Latency != 7 {
		t.Errorf("latency = %v, want 7", got.Latency)
	}
}

func deref(n *int64) any {
	if n == nil {
		return "absent"
	}
	return *n
}
