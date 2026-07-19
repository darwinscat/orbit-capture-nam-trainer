// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"errors"
	"testing"

	"orbit-capture-nam-trainer/internal/jobs"
)

func mkJob(key string, priority int, createdAt int64) jobs.Job {
	return jobs.Job{
		Key: key, Kind: jobs.KindTrain, State: jobs.StateQueued,
		Priority: priority, Epochs: 100, Arch: "standard", CreatedAt: createdAt,
	}
}

func TestInsertAndGetJobRoundTrip(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	if err := st.InsertJob(ctx, mkJob("k1", 1, 1000), []byte("wavdata")); err != nil {
		t.Fatalf("InsertJob: %v", err)
	}
	got, ok, err := st.GetJob(ctx, "k1")
	if err != nil || !ok {
		t.Fatalf("GetJob: ok=%v err=%v", ok, err)
	}
	if got.Kind != jobs.KindTrain || got.State != jobs.StateQueued || got.Epochs != 100 {
		t.Errorf("job mismatch: %+v", got)
	}
	if got.HasModel {
		t.Errorf("has_model should be false with no result row")
	}
	// The blob rode along in the same transaction.
	var n int
	if err := st.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM audio_blobs WHERE job_key='k1'").Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("blob rows = %d, want 1", n)
	}
}

func TestInsertJobDuplicateKeyIsErrExists(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	if err := st.InsertJob(ctx, mkJob("dup", 1, 1), []byte("a")); err != nil {
		t.Fatalf("first insert: %v", err)
	}
	err := st.InsertJob(ctx, mkJob("dup", 1, 2), []byte("b"))
	if !errors.Is(err, ErrExists) {
		t.Errorf("second insert err = %v, want ErrExists", err)
	}
	// The original blob must be untouched (no partial overwrite).
	var content []byte
	if err := st.db.QueryRowContext(ctx, "SELECT content FROM audio_blobs WHERE job_key='dup'").Scan(&content); err != nil {
		t.Fatal(err)
	}
	if string(content) != "a" {
		t.Errorf("blob = %q, want the original 'a'", content)
	}
}

func TestGetJobUnknownKey(t *testing.T) {
	st := openTest(t)
	_, ok, err := st.GetJob(context.Background(), "missing")
	if err != nil {
		t.Fatalf("GetJob: %v", err)
	}
	if ok {
		t.Error("ok = true for unknown key, want false")
	}
}

func TestDeleteJobCascadesAndReportsExistence(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	_ = st.InsertJob(ctx, mkJob("d", 1, 1), []byte("x"))
	_, _ = st.db.ExecContext(ctx, "INSERT INTO job_log(job_key,line) VALUES('d','line')")

	existed, err := st.DeleteJob(ctx, "d")
	if err != nil || !existed {
		t.Fatalf("DeleteJob: existed=%v err=%v", existed, err)
	}
	// Row and children gone.
	for _, q := range []string{
		"SELECT COUNT(*) FROM jobs WHERE key='d'",
		"SELECT COUNT(*) FROM audio_blobs WHERE job_key='d'",
		"SELECT COUNT(*) FROM job_log WHERE job_key='d'",
	} {
		var n int
		_ = st.db.QueryRowContext(ctx, q).Scan(&n)
		if n != 0 {
			t.Errorf("%s = %d, want 0", q, n)
		}
	}
	// Deleting again reports not-existed (frees the key).
	existed, err = st.DeleteJob(ctx, "d")
	if err != nil || existed {
		t.Errorf("second DeleteJob: existed=%v err=%v, want false/nil", existed, err)
	}
}

func TestQueuedPositionOrdering(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	// Insert out of natural order: same priority, different created_at, plus one
	// high-priority latecomer that must jump to the front.
	_ = st.InsertJob(ctx, mkJob("b", 1, 200), []byte("b"))
	_ = st.InsertJob(ctx, mkJob("a", 1, 100), []byte("a"))
	_ = st.InsertJob(ctx, mkJob("c", 1, 300), []byte("c"))
	_ = st.InsertJob(ctx, mkJob("hi", 0, 999), []byte("h")) // priority 0 = high

	wantPos := map[string]int{"hi": 1, "a": 2, "b": 3, "c": 4}
	for key, want := range wantPos {
		pos, ok, err := st.QueuedPosition(ctx, key)
		if err != nil || !ok {
			t.Fatalf("QueuedPosition(%s): ok=%v err=%v", key, ok, err)
		}
		if pos != want {
			t.Errorf("position(%s) = %d, want %d", key, pos, want)
		}
	}
}

func TestQueuedPositionLaneScoped(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	// A probe queued AHEAD (earlier created_at) of a train. Lanes drain
	// concurrently, so the train's position must ignore the probe: 1, not 2.
	_ = st.InsertJob(ctx, jobs.Job{Key: "p", Kind: jobs.KindProbeE10, State: jobs.StateQueued,
		Priority: 1, Epochs: 10, Arch: "standard", CreatedAt: 100}, []byte("p"))
	_ = st.InsertJob(ctx, mkJob("t", 1, 200), []byte("t"))

	if pos, ok, err := st.QueuedPosition(ctx, "t"); err != nil || !ok || pos != 1 {
		t.Errorf("train position = %d (ok=%v err=%v), want 1 — a probe ahead must not count", pos, ok, err)
	}
	if pos, ok, _ := st.QueuedPosition(ctx, "p"); !ok || pos != 1 {
		t.Errorf("probe position = %d (ok=%v), want 1 in its own lane", pos, ok)
	}
}

func TestQueuedPositionNilForNonQueued(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	_ = st.InsertJob(ctx, mkJob("r", 1, 1), []byte("r"))
	_, _ = st.db.ExecContext(ctx, "UPDATE jobs SET state='running' WHERE key='r'")

	_, ok, err := st.QueuedPosition(ctx, "r")
	if err != nil {
		t.Fatalf("QueuedPosition: %v", err)
	}
	if ok {
		t.Error("running job should have no position (ok=false)")
	}
	// Unknown key too.
	_, ok, _ = st.QueuedPosition(ctx, "nope")
	if ok {
		t.Error("unknown key should have no position")
	}
}

func TestSetPriorityIfQueued(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	_ = st.InsertJob(ctx, mkJob("q", 1, 1), []byte("q"))

	existed, err := st.SetPriorityIfQueued(ctx, "q", 0)
	if err != nil || !existed {
		t.Fatalf("SetPriorityIfQueued: existed=%v err=%v", existed, err)
	}
	got, _, _ := st.GetJob(ctx, "q")
	if got.Priority != 0 {
		t.Errorf("priority = %d, want 0", got.Priority)
	}

	// Running job: existed=true but priority unchanged (no-op).
	_, _ = st.db.ExecContext(ctx, "UPDATE jobs SET state='running', priority=1 WHERE key='q'")
	existed, err = st.SetPriorityIfQueued(ctx, "q", 2)
	if err != nil || !existed {
		t.Fatalf("running SetPriority: existed=%v err=%v", existed, err)
	}
	got, _, _ = st.GetJob(ctx, "q")
	if got.Priority != 1 {
		t.Errorf("running priority = %d, want unchanged 1", got.Priority)
	}

	// Unknown key: existed=false.
	existed, _ = st.SetPriorityIfQueued(ctx, "nope", 0)
	if existed {
		t.Error("unknown key should report existed=false")
	}
}

func TestModelBytesAndHasModel(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	_ = st.InsertJob(ctx, mkJob("m", 1, 1), []byte("m"))

	// No result yet.
	_, ok, err := st.ModelBytes(ctx, "m")
	if err != nil || ok {
		t.Fatalf("ModelBytes before result: ok=%v err=%v", ok, err)
	}
	// Add a result with a nam blob.
	_, _ = st.db.ExecContext(ctx,
		"INSERT INTO results(job_key,nam,train_json) VALUES('m',x'0102',?)", `{"esr":0.01}`)

	nam, ok, err := st.ModelBytes(ctx, "m")
	if err != nil || !ok {
		t.Fatalf("ModelBytes after result: ok=%v err=%v", ok, err)
	}
	if len(nam) != 2 || nam[0] != 1 || nam[1] != 2 {
		t.Errorf("nam bytes = %v, want [1 2]", nam)
	}
	got, _, _ := st.GetJob(ctx, "m")
	if !got.HasModel {
		t.Error("has_model should be true once nam is stored")
	}
}

func TestJobLog(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	_ = st.InsertJob(ctx, mkJob("l", 1, 1), []byte("l"))
	for _, line := range []string{"first", "second", "third"} {
		_, _ = st.db.ExecContext(ctx, "INSERT INTO job_log(job_key,line) VALUES('l',?)", line)
	}
	lines, err := st.JobLog(ctx, "l")
	if err != nil {
		t.Fatalf("JobLog: %v", err)
	}
	if len(lines) != 3 || lines[0] != "first" || lines[2] != "third" {
		t.Errorf("lines = %v, want [first second third] in order", lines)
	}
}
