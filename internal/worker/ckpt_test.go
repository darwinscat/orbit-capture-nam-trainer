// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.
// Migration 0007: a take's weights live in the library, one row, written epoch by epoch — so losing
// the trainer costs the unfinished epoch instead of the run, and the run can change machines.

package worker

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/storetest"
)

// The trainer writes a pair every epoch; the daemon carries it into the library as the epochs land,
// and it lands on the TAKE — which is what makes it survive the run that produced it.
func TestEachFinishedEpochIsKeptOnTheTake(t *testing.T) {
	h := newHarness(t, "train-hang-with-ckpts", time.Minute)
	id := h.seed(t, jobs.KindTrain, 20)
	h.start(t)
	take := h.get(t, id).TakeID

	waitFor(t, 20*time.Second, func() bool {
		c, ok, err := h.store.Checkpoint(context.Background(), take)
		return err == nil && ok && c.Reached == 6
	}, "the library never received the finished epoch")

	c, _, _ := h.store.Checkpoint(context.Background(), take)
	if len(c.Ckpt) == 0 || len(c.Nam) == 0 {
		t.Fatalf("both halves of the pair must be kept: nam=%d ckpt=%d bytes", len(c.Nam), len(c.Ckpt))
	}
	// …and the BEST pair beside the last one, for whoever wants to listen rather than continue.
	if n := storetest.Count(t, h.store,
		`SELECT count(*) FROM take_checkpoint WHERE take_id = $1 AND best_nam IS NOT NULL`, take); n != 1 {
		t.Errorf("the best pair was not kept beside the last one")
	}
}

// THE POINT OF ALL OF IT. A trainer loses the run — its row goes back in the queue, another claim
// picks it up, and it CONTINUES. Before this the requeue was a restart at zero.
func TestARequeuedRunContinuesFromTheTakesWeights(t *testing.T) {
	h := newHarness(t, "train-hang-with-ckpts", time.Minute)
	cr := &capturingRunner{inner: h.pool.runner}
	h.pool.runner = cr
	id := h.seed(t, jobs.KindTrain, 20)
	h.start(t)
	take := h.get(t, id).TakeID

	waitFor(t, 20*time.Second, func() bool {
		_, ok, err := h.store.Checkpoint(context.Background(), take)
		return err == nil && ok
	}, "nothing was kept before the trainer was lost")

	j := h.get(t, id)
	if err := h.store.RequeueJob(context.Background(), id, j.ClaimToken); err != nil {
		t.Fatalf("requeue: %v", err)
	}
	if _, ok, _ := h.store.Checkpoint(context.Background(), take); !ok {
		t.Fatal("a requeue threw the take's weights away — that is the whole regression")
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

// SUCCESS IS A PAUSE. A run reaching its epoch count is not the end of anything — the next gesture
// may be "now take it to sixteen hundred" — so the weights stay. Same for a failure: a run that died
// at epoch seven hundred trained seven hundred epochs.
func TestAFinishedRunLeavesTheTakeItsWeights(t *testing.T) {
	for _, tc := range []struct{ name, mode, want string }{
		{"a natural finish", "train-ok", jobs.StateSucceeded},
		{"a failure", "train-fail", jobs.StateFailed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, tc.mode, time.Minute)
			id := h.seed(t, jobs.KindTrain, 3)
			take := h.get(t, id).TakeID
			storetest.SeedTakeCheckpoint(t, h.store, take, 3, []byte("nam"), []byte("ckpt"))
			h.start(t)
			h.waitState(t, id, tc.want, 20*time.Second)
			if _, ok, err := h.store.Checkpoint(context.Background(), take); err != nil || !ok {
				t.Fatalf("a %s run took the take's weights with it", tc.want)
			}
		})
	}
}

// …and a CANCEL is the one thing that discards — but even it only moves them aside, bytes and all.
func TestACancelMovesTheWeightsAsideRatherThanDroppingThem(t *testing.T) {
	h := newHarness(t, "train-hang-with-ckpts", time.Minute)
	id := h.seed(t, jobs.KindTrain, 20)
	h.start(t)
	take := h.get(t, id).TakeID
	waitFor(t, 20*time.Second, func() bool {
		_, ok, err := h.store.Checkpoint(context.Background(), take)
		return err == nil && ok
	}, "nothing was kept to cancel")

	storetest.Exec(t, h.store, `UPDATE jobs SET cancel_requested_at = now() WHERE id = $1`, id)
	h.waitState(t, id, jobs.StateCancelled, 20*time.Second)

	if _, ok, _ := h.store.Checkpoint(context.Background(), take); ok {
		t.Fatal("a cancelled run kept the take's weights live — cancel means keep nothing")
	}
	if n := storetest.Count(t, h.store,
		`SELECT count(*) FROM take_checkpoint_discarded WHERE take_id = $1 AND reason = 'cancelled'
		   AND octet_length(nam) > 0 AND octet_length(ckpt) > 0`, take); n != 1 {
		t.Fatalf("the discarded weights must be readable afterwards, bytes and all (rows = %d)", n)
	}
}

// THE WRITE IS ALSO THE OWNERSHIP CHECK. A run handed to another machine — or marked claimable
// because its task went silent — writes nothing and is told so in the same breath.
func TestTheWriteTellsARunItIsNoLongerItsOwn(t *testing.T) {
	h := newHarness(t, "train-hang-with-ckpts", time.Minute)
	id := h.seed(t, jobs.KindTrain, 20)
	h.start(t)
	take := h.get(t, id).TakeID
	waitFor(t, 20*time.Second, func() bool {
		_, ok, err := h.store.Checkpoint(context.Background(), take)
		return err == nil && ok
	}, "nothing was kept")

	ctx := context.Background()
	j := h.get(t, id)
	// a stale claim writes nothing, and says so
	v, err := h.store.PutCheckpoint(ctx, id, "00000000-0000-0000-0000-000000000001", take,
		mkStorePair(99), nil)
	if err != nil {
		t.Fatalf("a fenced-out write must be a no-op, not an error: %v", err)
	}
	if v.Mine {
		t.Fatal("a stale claim was told the run is still its own")
	}

	// …and so does a run whose task cron has marked claimable
	storetest.Exec(t, h.store, `UPDATE jobs SET paused_at = now() WHERE id = $1`, id)
	if v, err = h.store.PutCheckpoint(ctx, id, j.ClaimToken, take, mkStorePair(7), nil); err != nil {
		t.Fatalf("PutCheckpoint: %v", err)
	}
	if v.Mine {
		t.Fatal("a paused run was told to carry on — it is up for grabs and must stop")
	}

	// the live, unpaused claim writes normally, and the write is also how it hears about a stop or a
	// cancel: one channel, in the round trip it was making anyway.
	storetest.Exec(t, h.store, `UPDATE jobs SET paused_at = NULL WHERE id = $1`, id)
	if v, err = h.store.PutCheckpoint(ctx, id, j.ClaimToken, take, mkStorePair(7), nil); err != nil || !v.Mine {
		t.Fatalf("the owner must be able to write: %+v err=%v", v, err)
	}
	if v.Stop || v.Cancel {
		t.Fatalf("nothing was asked of this run, yet it was told %+v", v)
	}
	storetest.Exec(t, h.store, `UPDATE jobs SET stop_requested_at = now() WHERE id = $1`, id)
	if v, err = h.store.PutCheckpoint(ctx, id, j.ClaimToken, take, mkStorePair(8), nil); err != nil || !v.Stop {
		t.Fatalf("a stop must come back with the write: %+v err=%v", v, err)
	}
	storetest.Exec(t, h.store, `UPDATE jobs SET cancel_requested_at = now() WHERE id = $1`, id)
	if v, err = h.store.PutCheckpoint(ctx, id, j.ClaimToken, take, mkStorePair(9), nil); err != nil || !v.Cancel {
		t.Fatalf("a cancel must come back with the write: %+v err=%v", v, err)
	}
}

// stop() IS A BARRIER. The last look happens on the way out, and the two things right after it — the
// requeue that takes the claim away and the removal of the scratch directory that takes the file
// away — both beat a database round trip.
func TestStoppingTheSaverWaitsForItsLastWrite(t *testing.T) {
	// No pool: this take has exactly one saver, the one below. A running job would have its own, and
	// two savers writing one take's row is a thing only a test can arrange.
	h := newHarness(t, "", time.Minute)
	j := h.seedAndClaim(t, jobs.KindTrain, 200)
	take := j.TakeID

	scratch := filepath.Join(h.base, "scratch", "barrier")
	mkPair(t, scratch, ".", "checkpoint_last_epoch=0040_step=2480.ckpt", `{"e40":true}`, false)

	saver := h.pool.startCkptSaver(context.Background(), j, scratch)
	saver.nudge(-1, nil)
	saver.stop() // must not return until the write has landed
	os.RemoveAll(scratch)

	c, ok, err := h.store.Checkpoint(context.Background(), take)
	if err != nil || !ok {
		t.Fatalf("the checkpoint vanished: ok=%v err=%v", ok, err)
	}
	if c.Reached != 41 {
		t.Fatalf("reached = %d, want 41 — stop() returned before the last write landed", c.Reached)
	}
}

// mkStorePair is a whole-looking pair for the fence tests, where the bytes do not matter.
func mkStorePair(reached int) store.Pair {
	nam := []byte(`{"x":1}`)
	return store.Pair{Reached: reached, Nam: nam, Ckpt: []byte("ckpt"), NamSHA: sha256Hex(nam)}
}
