// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"testing"

	"orbit-capture-nam-trainer/internal/jobs"
)

func insertQueued(t *testing.T, st *Store, key, kind string, priority int, createdAt int64, epochs int) {
	t.Helper()
	if err := st.InsertJob(context.Background(), jobs.Job{
		Key: key, Kind: kind, State: jobs.StateQueued,
		Priority: priority, Epochs: epochs, Arch: "standard", CreatedAt: createdAt,
	}, []byte("wav-"+key)); err != nil {
		t.Fatalf("insert %s: %v", key, err)
	}
}

func runAtEpoch(t *testing.T, st *Store, key string, epoch int) {
	t.Helper()
	if _, err := st.db.ExecContext(context.Background(),
		"UPDATE jobs SET state='running', started_at=1, epoch=? WHERE key=?", epoch, key); err != nil {
		t.Fatalf("run %s: %v", key, err)
	}
}

func eqIP(a, b *int) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

func eqI64(a, b *int64) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

func assertEntry(t *testing.T, m map[string]QueueEntry, key string, wantPos *int, wantAhead *int64) {
	t.Helper()
	e, ok := m[key]
	if !ok || !e.Found {
		t.Fatalf("%s: missing or not found (%+v)", key, e)
	}
	if !eqIP(e.Position, wantPos) {
		t.Errorf("%s position = %v, want %v", key, e.Position, wantPos)
	}
	if !eqI64(e.EpochsAhead, wantAhead) {
		t.Errorf("%s epochs_ahead = %v, want %v", key, e.EpochsAhead, wantAhead)
	}
}

func ip(v int) *int      { return &v }
func i64(v int64) *int64 { return &v }

func TestQueueViewLaneScopedPositionsAndAhead(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	insertQueued(t, st, "t1", jobs.KindTrain, 1, 100, 100)
	insertQueued(t, st, "t2", jobs.KindTrain, 1, 200, 100)
	insertQueued(t, st, "t3", jobs.KindTrain, 1, 300, 100)
	insertQueued(t, st, "p1", jobs.KindProbeE10, 1, 150, 10)
	runAtEpoch(t, st, "t1", 29) // 0-based epoch 29 → remaining 100-(29+1) = 70

	got, err := st.QueueView(ctx, []string{"t1", "t2", "t3", "p1", "nope"})
	if err != nil {
		t.Fatalf("QueueView: %v", err)
	}

	assertEntry(t, got, "t1", nil, i64(0))    // running: no position, 0 ahead
	assertEntry(t, got, "t2", ip(1), i64(70)) // queued #1, ahead = t1 remainder
	assertEntry(t, got, "t3", ip(2), i64(170)) // queued #2, ahead = 70 + t2.epochs
	assertEntry(t, got, "p1", ip(1), i64(0))  // its own lane — train jobs do not count
	if e := got["nope"]; e.Found {
		t.Errorf("nope: found = true, want false")
	}
}

func TestQueueViewCountsJobsNotInRequest(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	insertQueued(t, st, "t1", jobs.KindTrain, 1, 100, 100)
	insertQueued(t, st, "t2", jobs.KindTrain, 1, 200, 100)
	insertQueued(t, st, "mine", jobs.KindTrain, 1, 300, 100)
	runAtEpoch(t, st, "t1", 29) // remainder 70

	// Ask ONLY about "mine": its epochs_ahead must still count t1 (70) + t2 (100),
	// which are not in the request — the sum is over the whole lane, not the list.
	got, err := st.QueueView(ctx, []string{"mine"})
	if err != nil {
		t.Fatalf("QueueView: %v", err)
	}
	assertEntry(t, got, "mine", ip(2), i64(170))
}

func TestQueueViewRunningNullEpochCountsFull(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	insertQueued(t, st, "r", jobs.KindTrain, 1, 100, 400)
	insertQueued(t, st, "q", jobs.KindTrain, 1, 200, 100)
	// Running but no progress reported yet (epoch NULL, as for minutes during torch
	// import). It must count as its FULL epochs, not 0 (the SUM-skips-NULL trap).
	if _, err := st.db.ExecContext(ctx,
		"UPDATE jobs SET state='running', started_at=1, epoch=NULL WHERE key='r'"); err != nil {
		t.Fatal(err)
	}
	got, err := st.QueueView(ctx, []string{"q"})
	if err != nil {
		t.Fatalf("QueueView: %v", err)
	}
	assertEntry(t, got, "q", ip(1), i64(400))
}

func TestQueueViewTerminalHasNoScheduling(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	insertQueued(t, st, "done", jobs.KindTrain, 1, 100, 100)
	if _, err := st.db.ExecContext(ctx,
		"UPDATE jobs SET state='succeeded', finished_at=1 WHERE key='done'"); err != nil {
		t.Fatal(err)
	}
	got, err := st.QueueView(ctx, []string{"done"})
	if err != nil {
		t.Fatalf("QueueView: %v", err)
	}
	e := got["done"]
	if !e.Found {
		t.Fatal("done: not found")
	}
	if e.Position != nil || e.EpochsAhead != nil {
		t.Errorf("terminal: position=%v epochs_ahead=%v, want nil/nil", e.Position, e.EpochsAhead)
	}
}

func TestQueueViewDedupAndUnknown(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()
	insertQueued(t, st, "a", jobs.KindTrain, 1, 100, 100)

	got, err := st.QueueView(ctx, []string{"a", "a", "ghost"})
	if err != nil {
		t.Fatalf("QueueView: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("entries = %d, want 2 (a deduped, ghost)", len(got))
	}
	if e := got["a"]; !e.Found {
		t.Error("a: not found")
	}
	if e := got["ghost"]; e.Found {
		t.Errorf("ghost: found = true, want false")
	}
}
