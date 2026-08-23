// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.
// Migration 0006: a running job's weights live in the library, so losing the trainer costs one
// unfinished epoch instead of the whole run.

package worker

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/storetest"
)

// The trainer writes a pair every epoch; the daemon carries it into the library as the epochs land.
func TestEachFinishedEpochIsKeptInTheLibrary(t *testing.T) {
	h := newHarness(t, "train-hang-with-ckpts", time.Minute)
	id := h.seed(t, jobs.KindTrain, 20)
	h.start(t)

	waitFor(t, 20*time.Second, func() bool {
		c, ok, err := h.store.Checkpoint(context.Background(), id)
		return err == nil && ok && c.Reached == 6
	}, "the library never received the finished epoch")

	c, _, _ := h.store.Checkpoint(context.Background(), id)
	if len(c.Ckpt) == 0 || len(c.Nam) == 0 {
		t.Fatalf("both halves of the pair must be kept: nam=%d ckpt=%d bytes", len(c.Nam), len(c.Ckpt))
	}
	if c.Reached != 6 {
		t.Errorf("reached = %d, want 6 (epoch 5 is the sixth)", c.Reached)
	}
}

// THE POINT OF ALL OF IT. A trainer dies mid-run — its row goes back in the queue, another claim
// picks it up, and it CONTINUES. Before this the requeue was a restart at zero, and an eight-hundred
// epoch run caught at seven hundred cost an hour and twenty minutes of somebody's GPU.
func TestARequeuedRunContinuesFromTheKeptEpoch(t *testing.T) {
	h := newHarness(t, "train-hang-with-ckpts", time.Minute)
	cr := &capturingRunner{inner: h.pool.runner}
	h.pool.runner = cr
	id := h.seed(t, jobs.KindTrain, 20)
	h.start(t)

	waitFor(t, 20*time.Second, func() bool {
		_, ok, err := h.store.Checkpoint(context.Background(), id)
		return err == nil && ok
	}, "nothing was kept before the trainer was lost")

	// The trainer is gone: kill the child and put the row back exactly as a recovery does.
	j := h.get(t, id)
	if err := h.store.RequeueJob(context.Background(), id, j.ClaimToken); err != nil {
		t.Fatalf("requeue: %v", err)
	}
	// The kept epoch must SURVIVE the requeue — a requeue is not a terminal state.
	if _, ok, _ := h.store.Checkpoint(context.Background(), id); !ok {
		t.Fatal("a requeue threw the kept epoch away — that is the whole regression")
	}

	waitFor(t, 20*time.Second, func() bool {
		_, _, saw := cr.resume()
		return saw
	}, "the re-claim started from scratch instead of continuing")

	resume, exists, _ := cr.resume()
	if filepath.Base(resume) != "resume.ckpt" || !exists {
		t.Errorf("resume ckpt = %q (exists=%v), want a present <scratch>/resume.ckpt", resume, exists)
	}
}

// A run that is OVER holds nothing: its weights are in job_result, or they were not wanted. The
// schema does this by trigger, because there are several writers and one rule — and the case that
// would rot quietly is `cancelled`, where "kept nothing" used to be true for free.
func TestAFinishedRunKeepsNoCheckpoint(t *testing.T) {
	for _, tc := range []struct{ name, mode, want string }{
		{"a natural finish", "train-ok", jobs.StateSucceeded},
		{"a failure", "train-fail", jobs.StateFailed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, tc.mode, time.Minute)
			id := h.seed(t, jobs.KindTrain, 3)
			h.start(t)
			h.waitState(t, id, tc.want, 20*time.Second)
			if _, ok, err := h.store.Checkpoint(context.Background(), id); err != nil || ok {
				t.Fatalf("a %s job still holds a running checkpoint", tc.want)
			}
		})
	}
}

// …and a cancel, which is the one the trigger is really there for.
func TestACancelledRunKeepsNoCheckpoint(t *testing.T) {
	h := newHarness(t, "train-hang-with-ckpts", time.Minute)
	id := h.seed(t, jobs.KindTrain, 20)
	h.start(t)
	waitFor(t, 20*time.Second, func() bool {
		_, ok, err := h.store.Checkpoint(context.Background(), id)
		return err == nil && ok
	}, "nothing was kept to cancel")

	storetest.Exec(t, h.store, `UPDATE jobs SET cancel_requested_at = now() WHERE id = $1`, id)
	h.waitState(t, id, jobs.StateCancelled, 20*time.Second)
	if _, ok, _ := h.store.Checkpoint(context.Background(), id); ok {
		t.Fatal("a cancelled run kept its weights — cancel means keep nothing, in both places")
	}
}

// A straggler of an attempt that has lost the row writes nothing at all, and nothing goes backwards.
func TestAStaleAttemptCannotTouchTheKeptEpoch(t *testing.T) {
	h := newHarness(t, "train-hang-with-ckpts", time.Minute)
	id := h.seed(t, jobs.KindTrain, 20)
	h.start(t)
	waitFor(t, 20*time.Second, func() bool {
		c, ok, err := h.store.Checkpoint(context.Background(), id)
		return err == nil && ok && c.Reached == 6
	}, "nothing was kept")

	ctx := context.Background()
	const stale = "00000000-0000-0000-0000-000000000001"
	if err := h.store.PutCheckpoint(ctx, id, stale, 99, nil, []byte("x"), []byte("y"), sha256Hex([]byte("x"))); err != nil {
		t.Fatalf("a fenced-out write must be a no-op, not an error: %v", err)
	}
	if c, _, _ := h.store.Checkpoint(ctx, id); c.Reached != 6 {
		t.Fatalf("a stale claim overwrote the kept epoch: reached = %d", c.Reached)
	}

	// …and the live claim cannot go backwards either.
	j := h.get(t, id)
	if err := h.store.PutCheckpoint(ctx, id, j.ClaimToken, 2, nil, []byte("x"), []byte("y"), sha256Hex([]byte("x"))); err != nil {
		t.Fatalf("PutCheckpoint: %v", err)
	}
	if c, _, _ := h.store.Checkpoint(ctx, id); c.Reached != 6 {
		t.Fatalf("epoch 2 replaced epoch 6: reached = %d", c.Reached)
	}
}
