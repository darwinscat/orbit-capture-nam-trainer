// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"testing"

	"orbit-capture-nam-trainer/internal/jobs"
)

func setRunning(t *testing.T, st *Store, key string, startedAt int64) {
	t.Helper()
	if _, err := st.db.ExecContext(context.Background(),
		"UPDATE jobs SET state='running', started_at=? WHERE key=?", startedAt, key); err != nil {
		t.Fatalf("setRunning: %v", err)
	}
}

func TestClaimNextQueuedClearsStaleProgress(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	_ = st.InsertJob(ctx, mkJob("k", 1, 100), []byte("x"))
	// Simulate a recovered row that still carries old progress numbers.
	_, _ = st.db.ExecContext(ctx, "UPDATE jobs SET epoch=37, s_per_epoch=4.2 WHERE key='k'")

	j, ok, err := st.ClaimNextQueued(ctx, 500)
	if err != nil || !ok {
		t.Fatalf("ClaimNextQueued: ok=%v err=%v", ok, err)
	}
	if j.Epoch != nil || j.SPerEpoch != nil {
		t.Errorf("claimed job still carries progress: epoch=%v s=%v", j.Epoch, j.SPerEpoch)
	}
	got, _, _ := st.GetJob(ctx, "k")
	if got.Epoch != nil || got.SPerEpoch != nil {
		t.Errorf("row still has stale progress after claim: epoch=%v s=%v", got.Epoch, got.SPerEpoch)
	}
	if got.State != jobs.StateRunning {
		t.Errorf("state = %q, want running", got.State)
	}
}

func TestClaimNextQueuedEmpty(t *testing.T) {
	st := openTest(t)
	_, ok, err := st.ClaimNextQueued(context.Background(), 1)
	if err != nil || ok {
		t.Errorf("empty claim: ok=%v err=%v, want false/nil", ok, err)
	}
}

func TestUpdateProgressFencedByStartedAt(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	_ = st.InsertJob(ctx, mkJob("k", 1, 1), []byte("x"))
	setRunning(t, st, "k", 100)

	if err := st.UpdateProgress(ctx, "k", 5, 1.5, 100); err != nil {
		t.Fatalf("UpdateProgress (matching): %v", err)
	}
	// A lagging worker from a prior run (different started_at) must not overwrite.
	if err := st.UpdateProgress(ctx, "k", 9, 9.9, 999); err != nil {
		t.Fatalf("UpdateProgress (stale): %v", err)
	}
	got, _, _ := st.GetJob(ctx, "k")
	if got.Epoch == nil || *got.Epoch != 5 {
		t.Errorf("epoch = %v, want 5 (stale write must be fenced out)", got.Epoch)
	}
}

func TestAppendLogFencedByStartedAt(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	_ = st.InsertJob(ctx, mkJob("k", 1, 1), []byte("x"))
	setRunning(t, st, "k", 100)

	if err := st.AppendLog(ctx, "k", "kept", 100); err != nil {
		t.Fatalf("AppendLog (matching): %v", err)
	}
	if err := st.AppendLog(ctx, "k", "stale", 999); err != nil {
		t.Fatalf("AppendLog (stale): %v", err)
	}
	// A deleted row simply drops the line — no FK error.
	_, _ = st.db.ExecContext(ctx, "UPDATE jobs SET state='succeeded' WHERE key='k'")
	if err := st.AppendLog(ctx, "k", "after-terminal", 100); err != nil {
		t.Fatalf("AppendLog (terminal): %v", err)
	}
	lines, _ := st.JobLog(ctx, "k")
	if len(lines) != 1 || lines[0] != "kept" {
		t.Errorf("log = %v, want only [kept]", lines)
	}
}

func TestRecoverRunningReturnsPidsRequeuesAndClears(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	_ = st.InsertJob(ctx, mkJob("k", 1, 1), []byte("x"))
	_, _ = st.db.ExecContext(ctx,
		"UPDATE jobs SET state='running', pid=4242, started_at=10, epoch=10, s_per_epoch=3.0 WHERE key='k'")

	pids, err := st.RecoverRunning(ctx)
	if err != nil {
		t.Fatalf("RecoverRunning: %v", err)
	}
	if len(pids) != 1 || pids[0] != 4242 {
		t.Errorf("pids = %v, want [4242]", pids)
	}
	got, _, _ := st.GetJob(ctx, "k")
	if got.State != jobs.StateQueued {
		t.Errorf("state = %q, want queued", got.State)
	}
	if got.PID != nil || got.Epoch != nil || got.SPerEpoch != nil || got.StartedAt != nil {
		t.Errorf("recovered row not cleared: pid=%v epoch=%v s=%v started=%v",
			got.PID, got.Epoch, got.SPerEpoch, got.StartedAt)
	}
}

func TestFinishGatedByRunningState(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	// Running job → finish succeeds, blob dropped, result + esr stored.
	_ = st.InsertJob(ctx, mkJob("run", 1, 1), []byte("x"))
	setRunning(t, st, "run", 1)
	esr := 0.0123
	ok, err := st.FinishTrainSuccess(ctx, "run", 99, []byte{0xca, 0xfe}, `{"esr":0.0123}`, &esr, nil)
	if err != nil || !ok {
		t.Fatalf("FinishTrainSuccess: ok=%v err=%v", ok, err)
	}
	if _, present, _ := st.AudioBlob(ctx, "run"); present {
		t.Error("blob should be dropped at terminal state")
	}
	if _, present, _ := st.ModelBytes(ctx, "run"); !present {
		t.Error("model should be stored")
	}
	if got, _, _ := st.GetJob(ctx, "run"); got.ESR == nil || *got.ESR != 0.0123 {
		t.Errorf("train esr = %v, want 0.0123", got.ESR)
	}

	// A job that is NOT running (deleted mid-run analog) → ok=false, nothing written.
	_ = st.InsertJob(ctx, mkJob("q", 1, 2), []byte("y"))
	ok, err = st.FinishTrainSuccess(ctx, "q", 99, []byte{0x01}, "{}", nil, nil)
	if err != nil {
		t.Fatalf("FinishTrainSuccess (non-running): %v", err)
	}
	if ok {
		t.Error("finishing a non-running job should report ok=false")
	}
	if _, present, _ := st.ModelBytes(ctx, "q"); present {
		t.Error("no model should be written for a non-running job")
	}
}

func TestAvgSPerEpoch(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	// No train history → nil.
	if avg, err := st.AvgSPerEpoch(ctx); err != nil || avg != nil {
		t.Fatalf("empty: avg=%v err=%v, want nil/nil", avg, err)
	}

	addTrain := func(key string, spe float64, epoch int, finished int64) {
		_ = st.InsertJob(ctx, mkJob(key, 1, 1), []byte("x"))
		_, _ = st.db.ExecContext(ctx,
			"UPDATE jobs SET state='succeeded', s_per_epoch=?, epoch=?, finished_at=? WHERE key=?",
			spe, epoch, finished, key)
	}

	// One long train covers the whole 30-epoch window by itself.
	addTrain("a", 5.0, 399, 1000) // 400 computed epochs @ 5.0 s/ep
	if avg, _ := st.AvgSPerEpoch(ctx); avg == nil || *avg != 5.0 {
		t.Fatalf("single job avg = %v, want 5.0", avg)
	}

	// A newer, short run: 10 epochs @ 8.0. The strict 30-epoch window is filled by b
	// (its 10 epochs) plus the newest 20 of a's epochs, weighted by epoch count:
	// (10*8 + 20*5)/30 = 6.0 — a's contribution is clipped to the window edge.
	addTrain("b", 8.0, 9, 2000)
	avg, _ := st.AvgSPerEpoch(ctx)
	if avg == nil || *avg < 5.99 || *avg > 6.01 {
		t.Errorf("epoch-weighted windowed avg = %v, want 6.0", avg)
	}

	// Excluded from the average: probes, and a train with no s_per_epoch.
	_ = st.InsertJob(ctx, jobs.Job{Key: "p", Kind: jobs.KindProbeE10, State: jobs.StateQueued, Priority: 1, Epochs: 10, Arch: "standard", CreatedAt: 1}, []byte("p"))
	_, _ = st.db.ExecContext(ctx, "UPDATE jobs SET state='succeeded', s_per_epoch=99.0, epoch=9, finished_at=3000 WHERE key='p'")
	_ = st.InsertJob(ctx, mkJob("noesp", 1, 1), []byte("n"))
	_, _ = st.db.ExecContext(ctx, "UPDATE jobs SET state='failed', s_per_epoch=NULL, epoch=0, finished_at=3000 WHERE key='noesp'")
	avg2, _ := st.AvgSPerEpoch(ctx)
	if avg2 == nil || *avg2 < 5.99 || *avg2 > 6.01 {
		t.Errorf("avg after adding a probe + a no-s_per_epoch train = %v, want unchanged 6.0", avg2)
	}

	// Truncation: three newer 10-epoch runs @ 2.0 fill the whole 30-epoch window on
	// their own, so the older 8.0 and 5.0 runs drop out entirely and the average is
	// exactly 2.0. This proves the window actually clips, not just accumulates.
	addTrain("c", 2.0, 9, 4000)
	addTrain("d", 2.0, 9, 5000)
	addTrain("e", 2.0, 9, 6000)
	avg3, _ := st.AvgSPerEpoch(ctx)
	if avg3 == nil || *avg3 < 1.99 || *avg3 > 2.01 {
		t.Errorf("windowed avg after 3 newer 10-epoch runs = %v, want 2.0 (old runs truncated)", avg3)
	}
}

func TestGCExpiredModelsKeepsHistory(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	// An old succeeded job with a model, a log line, and a fresh one.
	_ = st.InsertJob(ctx, mkJob("old", 1, 1), []byte("x"))
	_, _ = st.db.ExecContext(ctx, "UPDATE jobs SET state='succeeded', finished_at=100 WHERE key='old'")
	_, _ = st.db.ExecContext(ctx, "INSERT INTO results(job_key,nam,train_json) VALUES('old',x'cafe','{}')")
	_, _ = st.db.ExecContext(ctx, "INSERT INTO job_log(job_key,line) VALUES('old','a line')")

	_ = st.InsertJob(ctx, mkJob("new", 1, 2), []byte("y"))
	_, _ = st.db.ExecContext(ctx, "UPDATE jobs SET state='succeeded', finished_at=1000 WHERE key='new'")
	_, _ = st.db.ExecContext(ctx, "INSERT INTO results(job_key,nam) VALUES('new',x'beef')")

	n, err := st.GCExpiredModels(ctx, 500) // cutoff between the two
	if err != nil || n != 1 {
		t.Fatalf("GCExpiredModels: n=%d err=%v, want 1", n, err)
	}
	// The old model blob is gone, but the row and its log survive as history.
	if _, ok, _ := st.ModelBytes(ctx, "old"); ok {
		t.Error("expired model should be freed")
	}
	if ex, _ := st.JobExists(ctx, "old"); !ex {
		t.Error("job row must remain as history")
	}
	if lines, _ := st.JobLog(ctx, "old"); len(lines) != 1 {
		t.Error("job log must remain as history")
	}
	// The recent model is kept.
	if _, ok, _ := st.ModelBytes(ctx, "new"); !ok {
		t.Error("in-window model should be kept")
	}
	if err := st.IncrementalVacuum(ctx); err != nil {
		t.Errorf("IncrementalVacuum: %v", err)
	}
}

func TestFinishFailedKeepsRowDropsBlob(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	_ = st.InsertJob(ctx, mkJob("f", 1, 1), []byte("x"))
	setRunning(t, st, "f", 1)

	ok, err := st.FinishFailed(ctx, "f", 99, "stalled", "no output")
	if err != nil || !ok {
		t.Fatalf("FinishFailed: ok=%v err=%v", ok, err)
	}
	got, present, _ := st.GetJob(ctx, "f")
	if !present || got.State != jobs.StateFailed {
		t.Errorf("job should remain as failed history, got present=%v state=%q", present, got.State)
	}
	if got.ErrorCode == nil || *got.ErrorCode != "stalled" {
		t.Errorf("error_code = %v, want stalled", got.ErrorCode)
	}
	if _, present, _ := st.AudioBlob(ctx, "f"); present {
		t.Error("blob should be dropped even on failure")
	}
}
