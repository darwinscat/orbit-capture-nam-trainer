// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"orbit-capture-nam-trainer/internal/jobkey"
	"orbit-capture-nam-trainer/internal/jobs"
)

// makeSucceededParent inserts a job (kind), runs it, and finishes it succeeded with
// the given ckpt — the shape a train_more parent must have. A child must re-PUT the
// same wav bytes to match wav_sha, so callers share one wav slice with the child.
func makeSucceededParent(t *testing.T, st *Store, key string, epochs int, arch string, wav, ckpt []byte) {
	t.Helper()
	ctx := context.Background()
	j := jobs.Job{Key: key, Kind: jobs.KindTrain, State: jobs.StateQueued,
		Priority: 1, Epochs: epochs, Arch: arch, CreatedAt: 1}
	if err := st.InsertJob(ctx, j, wav); err != nil {
		t.Fatalf("insert parent %s: %v", key, err)
	}
	setRunning(t, st, key, 1)
	if ok, err := st.FinishTrainSuccess(ctx, key, 100, []byte("nam-"+key), "{}", nil, ckpt); err != nil || !ok {
		t.Fatalf("finish parent %s: ok=%v err=%v", key, ok, err)
	}
}

// trainMoreChild builds a queued train_more job whose parent is baseKey.
func trainMoreChild(key, baseKey string, epochs int, arch string) jobs.Job {
	b := baseKey
	return jobs.Job{Key: key, Kind: jobs.KindTrainMore, State: jobs.StateQueued,
		Priority: 1, Epochs: epochs, Arch: arch, CreatedAt: 2, BaseKey: &b}
}

func TestInsertTrainMoreHappyPath(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	wav := []byte("shared-capture-bytes")
	ckpt := []byte("parent-checkpoint-blob")
	makeSucceededParent(t, st, "parent", 200, "standard", wav, ckpt)

	if err := st.InsertJob(ctx, trainMoreChild("child", "parent", 400, "standard"), wav); err != nil {
		t.Fatalf("InsertJob train_more: %v", err)
	}

	// The child row committed with its provenance and numbering origin.
	child, ok, err := st.GetJob(ctx, "child")
	if err != nil || !ok {
		t.Fatalf("GetJob(child): ok=%v err=%v", ok, err)
	}
	if child.Kind != jobs.KindTrainMore || child.State != jobs.StateQueued {
		t.Errorf("child = %+v, want queued train_more", child)
	}
	if child.BaseKey == nil || *child.BaseKey != "parent" {
		t.Errorf("child base_key = %v, want parent", child.BaseKey)
	}
	if child.StartEpoch == nil || *child.StartEpoch != 200 {
		t.Errorf("child start_epoch = %v, want 200 (parent's epochs)", child.StartEpoch)
	}
	if child.WavSHA == nil || *child.WavSHA != jobkey.SHA256Hex(wav) {
		t.Errorf("child wav_sha = %v, want %s", child.WavSHA, jobkey.SHA256Hex(wav))
	}

	// The parent's checkpoint was snapshotted into the child's private row, so the
	// child is self-contained: deleting the parent must not touch the snapshot.
	snap, ok, err := st.ResumeCkpt(ctx, "child")
	if err != nil || !ok {
		t.Fatalf("ResumeCkpt(child): ok=%v err=%v", ok, err)
	}
	if !bytes.Equal(snap, ckpt) {
		t.Errorf("snapshot = %q, want the parent ckpt %q", snap, ckpt)
	}
	if _, err := st.DeleteJob(ctx, "parent"); err != nil {
		t.Fatalf("DeleteJob(parent): %v", err)
	}
	if snap, ok, _ := st.ResumeCkpt(ctx, "child"); !ok || !bytes.Equal(snap, ckpt) {
		t.Errorf("snapshot after parent delete = %q (ok=%v), want it to survive", snap, ok)
	}
}

func TestInsertTrainMoreEligibilityMatrix(t *testing.T) {
	wav := []byte("shared-capture-bytes")
	other := []byte("a-different-take")
	ckpt := []byte("parent-checkpoint-blob")

	// setup prepares the store for one ineligible case and returns the child to try.
	cases := []struct {
		name  string
		setup func(t *testing.T, st *Store) (child jobs.Job, childWav []byte)
	}{
		{"unknown parent", func(t *testing.T, st *Store) (jobs.Job, []byte) {
			return trainMoreChild("c", "ghost", 400, "standard"), wav
		}},
		{"failed parent", func(t *testing.T, st *Store) (jobs.Job, []byte) {
			makeSucceededParent(t, st, "p", 200, "standard", wav, ckpt)
			// A ckpt is present, but the state gate must reject a non-succeeded parent.
			if _, err := st.db.ExecContext(context.Background(),
				"UPDATE jobs SET state='failed' WHERE key='p'"); err != nil {
				t.Fatal(err)
			}
			return trainMoreChild("c", "p", 400, "standard"), wav
		}},
		{"no-ckpt parent", func(t *testing.T, st *Store) (jobs.Job, []byte) {
			makeSucceededParent(t, st, "p", 200, "standard", wav, nil) // succeeded but ckpt NULL
			return trainMoreChild("c", "p", 400, "standard"), wav
		}},
		{"wav mismatch", func(t *testing.T, st *Store) (jobs.Job, []byte) {
			makeSucceededParent(t, st, "p", 200, "standard", wav, ckpt)
			return trainMoreChild("c", "p", 400, "standard"), other // different capture
		}},
		{"arch mismatch", func(t *testing.T, st *Store) (jobs.Job, []byte) {
			makeSucceededParent(t, st, "p", 200, "standard", wav, ckpt)
			return trainMoreChild("c", "p", 400, "lite"), wav
		}},
		{"epochs not greater", func(t *testing.T, st *Store) (jobs.Job, []byte) {
			makeSucceededParent(t, st, "p", 200, "standard", wav, ckpt)
			return trainMoreChild("c", "p", 200, "standard"), wav // == parent, must be >
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := openTest(t)
			ctx := context.Background()
			child, childWav := tc.setup(t, st)

			err := st.InsertJob(ctx, child, childWav)
			if !errors.Is(err, ErrBaseUnavailable) {
				t.Fatalf("err = %v, want ErrBaseUnavailable", err)
			}
			var bu *BaseUnavailableError
			if !errors.As(err, &bu) || bu.Reason == "" {
				t.Errorf("err = %v, want a *BaseUnavailableError with a Reason", err)
			}
			// No dangling child row and no dangling snapshot — the whole tx rolled back.
			if _, ok, _ := st.GetJob(ctx, child.Key); ok {
				t.Error("child row must not exist after an ineligible insert")
			}
			if _, ok, _ := st.ResumeCkpt(ctx, child.Key); ok {
				t.Error("no resume_ckpts snapshot may survive an ineligible insert")
			}
		})
	}
}

func TestInsertTrainMoreParentDeletedBeforeInsert(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	wav := []byte("shared-capture-bytes")
	makeSucceededParent(t, st, "parent", 200, "standard", wav, []byte("ckpt"))
	// The parent is gone by the time the child is submitted (the DELETE=forget race).
	if _, err := st.DeleteJob(ctx, "parent"); err != nil {
		t.Fatalf("DeleteJob(parent): %v", err)
	}

	err := st.InsertJob(ctx, trainMoreChild("child", "parent", 400, "standard"), wav)
	if !errors.Is(err, ErrBaseUnavailable) {
		t.Fatalf("err = %v, want ErrBaseUnavailable", err)
	}
	if _, ok, _ := st.GetJob(ctx, "child"); ok {
		t.Error("child row must not exist when the parent vanished")
	}
	if _, ok, _ := st.ResumeCkpt(ctx, "child"); ok {
		t.Error("no dangling snapshot when the parent vanished")
	}
}

func TestInsertTrainMoreDuplicateKeyIsErrExists(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	wav := []byte("shared-capture-bytes")
	makeSucceededParent(t, st, "parent", 200, "standard", wav, []byte("ckpt"))
	if err := st.InsertJob(ctx, trainMoreChild("child", "parent", 400, "standard"), wav); err != nil {
		t.Fatalf("first insert: %v", err)
	}
	err := st.InsertJob(ctx, trainMoreChild("child", "parent", 400, "standard"), wav)
	if !errors.Is(err, ErrExists) {
		t.Errorf("duplicate train_more err = %v, want ErrExists", err)
	}
}

func TestFinishRemovesResumeCkpt(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	wav := []byte("shared-capture-bytes")
	makeSucceededParent(t, st, "parent", 200, "standard", wav, []byte("ckpt"))
	if err := st.InsertJob(ctx, trainMoreChild("child", "parent", 400, "standard"), wav); err != nil {
		t.Fatalf("insert child: %v", err)
	}
	if _, ok, _ := st.ResumeCkpt(ctx, "child"); !ok {
		t.Fatal("snapshot must exist before the child runs")
	}

	// Run the child to a terminal state; its resume snapshot is run-input and dies.
	setRunning(t, st, "child", 5)
	if ok, err := st.FinishTrainSuccess(ctx, "child", 200, []byte("nam-child"), "{}", nil, []byte("child-ckpt")); err != nil || !ok {
		t.Fatalf("FinishTrainSuccess(child): ok=%v err=%v", ok, err)
	}
	if _, ok, _ := st.ResumeCkpt(ctx, "child"); ok {
		t.Error("resume_ckpts snapshot must be dropped at terminal state")
	}
	// The child's OWN result ckpt lives on (it is chain-ready, not run-input).
	var n int
	if err := st.db.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM results WHERE job_key='child' AND ckpt IS NOT NULL").Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("child result ckpt rows = %d, want 1 (kept)", n)
	}
}

func TestGCExpiredModelsNullsNamAndCkpt(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	// A succeeded train with both a model and a checkpoint, finished long ago.
	makeSucceededParent(t, st, "old", 200, "standard", []byte("wav"), []byte("ckpt-old"))
	if _, err := st.db.ExecContext(ctx, "UPDATE jobs SET finished_at=100 WHERE key='old'"); err != nil {
		t.Fatal(err)
	}
	// A probe_e10 of the same age: its results row has nam=NULL from birth, so a
	// nam-only GC gate would never match it and its seed ckpt would outlive
	// retention forever (a train_more off it would still get 201 past the window).
	probe := jobs.Job{Key: "oldprobe", Kind: jobs.KindProbeE10, State: jobs.StateQueued,
		Priority: 1, Epochs: jobs.ProbeE10Epochs, Arch: "standard", CreatedAt: 1}
	if err := st.InsertJob(ctx, probe, []byte("probe-wav")); err != nil {
		t.Fatal(err)
	}
	setRunning(t, st, "oldprobe", 1)
	if ok, err := st.FinishProbeE10(ctx, "oldprobe", 100, 0.05, []byte("ckpt-probe")); err != nil || !ok {
		t.Fatalf("FinishProbeE10: ok=%v err=%v", ok, err)
	}

	n, err := st.GCExpiredModels(ctx, 500)
	if err != nil || n != 2 {
		t.Fatalf("GCExpiredModels: n=%d err=%v, want 2 (train row AND probe ckpt row)", n, err)
	}
	for _, key := range []string{"old", "oldprobe"} {
		var nam, ckpt []byte
		if err := st.db.QueryRowContext(ctx,
			"SELECT nam, ckpt FROM results WHERE job_key=?", key).Scan(&nam, &ckpt); err != nil {
			t.Fatal(err)
		}
		if nam != nil || ckpt != nil {
			t.Errorf("after GC %s: nam=%v ckpt=%v, want both NULL", key, nam, ckpt)
		}
	}
}

func TestFinishProbeE10StoresCkptSeedsTrainMore(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	// A probe_e10 that ran to completion leaves a checkpoint but never a model.
	wav := []byte("probe-capture-bytes")
	probe := jobs.Job{Key: "probe", Kind: jobs.KindProbeE10, State: jobs.StateQueued,
		Priority: 1, Epochs: jobs.ProbeE10Epochs, Arch: "standard", CreatedAt: 1}
	if err := st.InsertJob(ctx, probe, wav); err != nil {
		t.Fatalf("insert probe: %v", err)
	}
	setRunning(t, st, "probe", 1)
	if ok, err := st.FinishProbeE10(ctx, "probe", 100, 0.031, []byte("probe-ckpt")); err != nil || !ok {
		t.Fatalf("FinishProbeE10: ok=%v err=%v", ok, err)
	}

	// has_model stays false (nam IS NULL) even though a results row now exists.
	got, _, _ := st.GetJob(ctx, "probe")
	if got.HasModel {
		t.Error("probe_e10 must never report has_model=true")
	}
	if _, ok, _ := st.ModelBytes(ctx, "probe"); ok {
		t.Error("probe_e10 must have no downloadable model")
	}
	var ckpt []byte
	if err := st.db.QueryRowContext(ctx, "SELECT ckpt FROM results WHERE job_key='probe'").Scan(&ckpt); err != nil {
		t.Fatalf("read probe ckpt: %v", err)
	}
	if !bytes.Equal(ckpt, []byte("probe-ckpt")) {
		t.Errorf("probe ckpt = %q, want probe-ckpt", ckpt)
	}

	// Kind-agnostic eligibility: the probe seeds a train_more (start_epoch=10).
	if err := st.InsertJob(ctx, trainMoreChild("cont", "probe", 40, "standard"), wav); err != nil {
		t.Fatalf("train_more off a probe_e10 parent: %v", err)
	}
	child, _, _ := st.GetJob(ctx, "cont")
	if child.StartEpoch == nil || *child.StartEpoch != jobs.ProbeE10Epochs {
		t.Errorf("start_epoch = %v, want %d (the probe's epochs)", child.StartEpoch, jobs.ProbeE10Epochs)
	}
	if snap, ok, _ := st.ResumeCkpt(ctx, "cont"); !ok || !bytes.Equal(snap, []byte("probe-ckpt")) {
		t.Errorf("snapshot = %q (ok=%v), want the probe's ckpt", snap, ok)
	}
}

func TestResumeCkptAbsent(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	// A plain job never has a resume snapshot; an unknown key likewise.
	_ = st.InsertJob(ctx, mkJob("plain", 1, 1), []byte("x"))
	if _, ok, err := st.ResumeCkpt(ctx, "plain"); err != nil || ok {
		t.Errorf("ResumeCkpt(plain): ok=%v err=%v, want false/nil", ok, err)
	}
	if _, ok, err := st.ResumeCkpt(ctx, "ghost"); err != nil || ok {
		t.Errorf("ResumeCkpt(ghost): ok=%v err=%v, want false/nil", ok, err)
	}
}

// insertQueuedMore inserts a QUEUED train_more row directly (bypassing InsertJob's
// parent-eligibility check), so the lane/queue-math tests can pin the arithmetic
// without staging a full parent for every row.
func insertQueuedMore(t *testing.T, st *Store, key string, priority int, createdAt, epochs, startEpoch int) {
	t.Helper()
	if _, err := st.db.ExecContext(context.Background(),
		`INSERT INTO jobs(key,kind,state,priority,epochs,arch,created_at,start_epoch)
		 VALUES(?,?,'queued',?,?,'standard',?,?)`,
		key, jobs.KindTrainMore, priority, epochs, createdAt, startEpoch); err != nil {
		t.Fatalf("insert train_more %s: %v", key, err)
	}
}

func TestQueuedPositionTrainAndTrainMoreShareLane(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	// One drain order across train + train_more; a probe in its own lane must not count.
	insertQueued(t, st, "t1", jobs.KindTrain, 1, 100, 100)
	insertQueuedMore(t, st, "tm2", 1, 200, 400, 200)
	insertQueued(t, st, "t3", jobs.KindTrain, 1, 300, 100)
	insertQueued(t, st, "pe", jobs.KindProbeE10, 1, 150, 10)

	want := map[string]int{"t1": 1, "tm2": 2, "t3": 3}
	for key, wp := range want {
		if pos, ok, err := st.QueuedPosition(ctx, key); err != nil || !ok || pos != wp {
			t.Errorf("position(%s) = %d (ok=%v err=%v), want %d", key, pos, ok, err, wp)
		}
	}
	// The probe is alone in its lane.
	if pos, ok, _ := st.QueuedPosition(ctx, "pe"); !ok || pos != 1 {
		t.Errorf("probe position = %d (ok=%v), want 1", pos, ok)
	}
}

func TestQueueViewTrainMoreEpochsMinusStartEpoch(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	// A running train, then a queued train_more that only owes epochs−start_epoch.
	insertQueued(t, st, "t1", jobs.KindTrain, 1, 100, 100)
	insertQueuedMore(t, st, "tm", 1, 200, 400, 200) // remaining contribution = 200, not 400
	insertQueued(t, st, "t3", jobs.KindTrain, 1, 300, 100)
	runAtEpoch(t, st, "t1", 29) // 0-based epoch 29 → remaining 100-(29+1) = 70

	got, err := st.QueueView(ctx, []string{"t1", "tm", "t3"})
	if err != nil {
		t.Fatalf("QueueView: %v", err)
	}
	assertEntry(t, got, "t1", nil, i64(0))     // running, nothing ahead
	assertEntry(t, got, "tm", ip(1), i64(70))  // queued #1: ahead = t1 remainder
	assertEntry(t, got, "t3", ip(2), i64(270)) // queued #2: 70 + tm's 200 (not 400)
}

func TestQueueViewRunningTrainMoreNullEpochUsesStartEpoch(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	// A running train_more silent during torch import (epoch NULL): its remainder is
	// epochs−start_epoch = 200, NOT the full 400 and NOT zero.
	insertQueuedMore(t, st, "tm", 1, 100, 400, 200)
	insertQueued(t, st, "q", jobs.KindTrain, 1, 200, 100)
	if _, err := st.db.ExecContext(ctx,
		"UPDATE jobs SET state='running', started_at=1, epoch=NULL WHERE key='tm'"); err != nil {
		t.Fatal(err)
	}
	got, err := st.QueueView(ctx, []string{"q"})
	if err != nil {
		t.Fatalf("QueueView: %v", err)
	}
	assertEntry(t, got, "q", ip(1), i64(200))
}

func TestQueueTotalsTrainMoreFoldsIntoTrainLane(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	insertQueued(t, st, "t1", jobs.KindTrain, 1, 100, 100)
	insertQueuedMore(t, st, "tm", 1, 200, 400, 200) // contributes 200 to the train lane
	insertQueued(t, st, "pe", jobs.KindProbeE10, 1, 150, 10)

	running, queued, remaining, err := st.QueueTotals(ctx)
	if err != nil {
		t.Fatalf("QueueTotals: %v", err)
	}
	if running != 0 || queued != 3 {
		t.Errorf("counts = %d/%d, want 0/3", running, queued)
	}
	if remaining[jobs.KindTrain] != 300 { // 100 (train) + 200 (train_more) in one lane
		t.Errorf("train-lane remaining = %d, want 300", remaining[jobs.KindTrain])
	}
	if remaining[jobs.KindProbeE10] != 10 {
		t.Errorf("probe_e10 remaining = %d, want 10", remaining[jobs.KindProbeE10])
	}
}

func TestAvgSPerEpochWeightsByComputedEpochs(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	// Newest: a train_more that resumed at epoch 20 and ran to absolute epoch 29 —
	// it computed only 29+1−20 = 10 epochs, weighted as 10 (not 30). Older: a full
	// train of 100 epochs @ 2.0. The 30-epoch window is 10 of tm @ 10.0 + 20 of t @
	// 2.0 → (10*10 + 20*2)/30 = 4.6667. A weight bug (using epoch+1=30 for tm) would
	// fill the window with tm alone and yield 10.0.
	if _, err := st.db.ExecContext(ctx,
		`INSERT INTO jobs(key,kind,state,epochs,arch,created_at,finished_at,epoch,s_per_epoch,start_epoch)
		 VALUES('tm','train_more','succeeded',400,'standard',1,2000,29,10.0,20)`); err != nil {
		t.Fatal(err)
	}
	if _, err := st.db.ExecContext(ctx,
		`INSERT INTO jobs(key,kind,state,epochs,arch,created_at,finished_at,epoch,s_per_epoch)
		 VALUES('t','train','succeeded',100,'standard',1,1000,99,2.0)`); err != nil {
		t.Fatal(err)
	}

	avg, err := st.AvgSPerEpoch(ctx)
	if err != nil || avg == nil {
		t.Fatalf("AvgSPerEpoch: avg=%v err=%v", avg, err)
	}
	if *avg < 4.66 || *avg > 4.67 {
		t.Errorf("avg = %v, want ~4.6667 (train_more weighted by 10 computed epochs)", *avg)
	}
}

func TestAvgSPerEpochClampsNonPositiveWeight(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	// A pathological row whose recorded epoch sits BELOW its start_epoch (nothing
	// writes one today) must not poison the average with a negative weight: the
	// MAX(1,…) clamp pins it to 1 epoch. Window: 1 of bad @ 100.0 + 29 of t @ 2.0.
	if _, err := st.db.ExecContext(ctx,
		`INSERT INTO jobs(key,kind,state,epochs,arch,created_at,finished_at,epoch,s_per_epoch,start_epoch)
		 VALUES('bad','train_more','succeeded',400,'standard',1,2000,5,100.0,200)`); err != nil {
		t.Fatal(err)
	}
	if _, err := st.db.ExecContext(ctx,
		`INSERT INTO jobs(key,kind,state,epochs,arch,created_at,finished_at,epoch,s_per_epoch)
		 VALUES('t','train','succeeded',100,'standard',1,1000,99,2.0)`); err != nil {
		t.Fatal(err)
	}

	avg, err := st.AvgSPerEpoch(ctx)
	if err != nil || avg == nil {
		t.Fatalf("AvgSPerEpoch: avg=%v err=%v", avg, err)
	}
	want := (1*100.0 + 29*2.0) / 30.0 // ≈ 5.2667
	if *avg < want-0.01 || *avg > want+0.01 {
		t.Errorf("avg = %v, want ~%.4f (bad row clamped to weight 1)", *avg, want)
	}
}
