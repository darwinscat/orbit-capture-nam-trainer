// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"testing"

	"orbit-capture-nam-trainer/internal/jobs"
)

func f64p(v float64) *float64 { return &v }

// insertQueuedTrain inserts a plain queued train row for the reached tests.
func insertQueuedTrain(t *testing.T, st *Store, key string, epochs int) {
	t.Helper()
	j := jobs.Job{Key: key, Kind: jobs.KindTrain, State: jobs.StateQueued,
		Priority: 1, Epochs: epochs, Arch: "standard", CreatedAt: 1}
	if err := st.InsertJob(context.Background(), j, []byte("wav-"+key)); err != nil {
		t.Fatalf("insert train %s: %v", key, err)
	}
}

func TestFinishTrainSuccessStampsReachedEqualsEpochs(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	insertQueuedTrain(t, st, "t", 250)
	setRunning(t, st, "t", 1)
	if ok, err := st.FinishTrainSuccess(ctx, "t", 99, []byte("nam"), "{}", nil, []byte("ck")); err != nil || !ok {
		t.Fatalf("FinishTrainSuccess: ok=%v err=%v", ok, err)
	}
	got, _, _ := st.GetJob(ctx, "t")
	if got.Reached == nil || *got.Reached != 250 {
		t.Errorf("reached = %v, want 250 (== epochs on a natural finish)", got.Reached)
	}
}

func TestProbesLeaveReachedNull(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	// probe_self: killed on verdict, never carries reached.
	ps := jobs.Job{Key: "ps", Kind: jobs.KindProbeSelf, State: jobs.StateQueued,
		Priority: 1, Epochs: jobs.ProbeSelfEpochs, Arch: "standard", CreatedAt: 1}
	if err := st.InsertJob(ctx, ps, []byte("w1")); err != nil {
		t.Fatalf("insert probe_self: %v", err)
	}
	setRunning(t, st, "ps", 1)
	if ok, err := st.FinishProbeSelf(ctx, "ps", 99, jobs.VerdictPass, nil); err != nil || !ok {
		t.Fatalf("FinishProbeSelf: ok=%v err=%v", ok, err)
	}

	// probe_e10: succeeds with an E@10 ESR and a seed ckpt, still no reached.
	pe := jobs.Job{Key: "pe", Kind: jobs.KindProbeE10, State: jobs.StateQueued,
		Priority: 1, Epochs: jobs.ProbeE10Epochs, Arch: "standard", CreatedAt: 2}
	if err := st.InsertJob(ctx, pe, []byte("w2")); err != nil {
		t.Fatalf("insert probe_e10: %v", err)
	}
	setRunning(t, st, "pe", 1)
	if ok, err := st.FinishProbeE10(ctx, "pe", 99, 0.05, []byte("ck")); err != nil || !ok {
		t.Fatalf("FinishProbeE10: ok=%v err=%v", ok, err)
	}

	for _, key := range []string{"ps", "pe"} {
		if got, _, _ := st.GetJob(ctx, key); got.Reached != nil {
			t.Errorf("%s reached = %v, want NULL (probes never carry reached)", key, got.Reached)
		}
	}
}

func TestFinishStoppedHappyPath(t *testing.T) {
	cases := []struct {
		name string
		esr  *float64
	}{
		{"esr nil (log line unavailable)", nil},
		{"esr set", f64p(0.0321)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := openTest(t)
			ctx := context.Background()

			// A train_more child so a resume snapshot exists — the stop must drop it,
			// like every other terminal transition (finish() invariants).
			wav := []byte("shared-capture-bytes")
			makeSucceededParent(t, st, "parent", 200, "standard", wav, []byte("parent-ckpt"))
			if err := st.InsertJob(ctx, trainMoreChild("child", "parent", 400, "standard"), wav); err != nil {
				t.Fatalf("insert child: %v", err)
			}
			setRunning(t, st, "child", 5)
			if _, ok, _ := st.ResumeCkpt(ctx, "child"); !ok {
				t.Fatal("resume snapshot must exist before the stop")
			}

			nam := []byte("last-nam")
			ckpt := []byte("last-ckpt")
			ok, err := st.FinishStopped(ctx, "child", 123, nam, ckpt, tc.esr, 251)
			if err != nil || !ok {
				t.Fatalf("FinishStopped: ok=%v err=%v", ok, err)
			}

			got, present, _ := st.GetJob(ctx, "child")
			if !present || got.State != jobs.StateSucceeded {
				t.Fatalf("child = present:%v state:%q, want a normal succeeded run", present, got.State)
			}
			if got.Reached == nil || *got.Reached != 251 {
				t.Errorf("reached = %v, want 251 (the harvested count, not epochs 400)", got.Reached)
			}
			if got.Epochs != 400 {
				t.Errorf("epochs = %d, want 400 (unchanged — it is in the key)", got.Epochs)
			}
			if tc.esr == nil {
				if got.ESR != nil {
					t.Errorf("esr = %v, want NULL", got.ESR)
				}
			} else if got.ESR == nil || *got.ESR != *tc.esr {
				t.Errorf("esr = %v, want %v", got.ESR, *tc.esr)
			}
			if got.ErrorCode != nil || got.ErrorMsg != nil {
				t.Errorf("error fields = %v/%v, want both NULL on a succeeded stop", got.ErrorCode, got.ErrorMsg)
			}

			// The last pair is stored: the .nam is downloadable, the ckpt kept, and
			// NO train.json is written (a stop keeps no metrics file).
			if !got.HasModel {
				t.Error("has_model must be true (the last .nam is stored)")
			}
			if b, ok, _ := st.ModelBytes(ctx, "child"); !ok || !bytes.Equal(b, nam) {
				t.Errorf("model = %q (ok=%v), want %q", b, ok, nam)
			}
			var storedCkpt []byte
			var trainJSON sql.NullString
			if err := st.db.QueryRowContext(ctx,
				"SELECT ckpt, train_json FROM results WHERE job_key='child'").Scan(&storedCkpt, &trainJSON); err != nil {
				t.Fatalf("read result: %v", err)
			}
			if !bytes.Equal(storedCkpt, ckpt) {
				t.Errorf("stored ckpt = %q, want %q", storedCkpt, ckpt)
			}
			if trainJSON.Valid {
				t.Errorf("train_json = %q, want NULL (a stop keeps no metrics file)", trainJSON.String)
			}

			// finish() invariants: the capture blob and the resume snapshot are gone.
			if _, present, _ := st.AudioBlob(ctx, "child"); present {
				t.Error("capture blob must be dropped at terminal state")
			}
			if _, ok, _ := st.ResumeCkpt(ctx, "child"); ok {
				t.Error("resume snapshot must be dropped at terminal state")
			}
		})
	}
}

func TestFinishStoppedGatedByRunningState(t *testing.T) {
	st := openTest(t)
	ctx := context.Background()

	// A queued (never-running) job: the running gate makes FinishStopped a no-op,
	// exactly like its siblings — a DELETE mid-run must never resurrect the row.
	insertQueuedTrain(t, st, "q", 100)
	ok, err := st.FinishStopped(ctx, "q", 99, []byte("nam"), []byte("ck"), nil, 3)
	if err != nil {
		t.Fatalf("FinishStopped (non-running): %v", err)
	}
	if ok {
		t.Error("finishing a non-running job should report ok=false")
	}
	if _, present, _ := st.ModelBytes(ctx, "q"); present {
		t.Error("no model should be written for a non-running job")
	}
	if got, _, _ := st.GetJob(ctx, "q"); got.Reached != nil {
		t.Errorf("reached = %v, want NULL (never finished)", got.Reached)
	}
}

// makeStoppedParent inserts an early-stopped parent: epochs requested = reqEpochs,
// but its computed-epoch count is reached (reached < reqEpochs), with a stored ckpt.
func makeStoppedParent(t *testing.T, st *Store, key string, reqEpochs int, reached int64, wav, ckpt []byte) {
	t.Helper()
	ctx := context.Background()
	j := jobs.Job{Key: key, Kind: jobs.KindTrain, State: jobs.StateQueued,
		Priority: 1, Epochs: reqEpochs, Arch: "standard", CreatedAt: 1}
	if err := st.InsertJob(ctx, j, wav); err != nil {
		t.Fatalf("insert stopped parent %s: %v", key, err)
	}
	setRunning(t, st, key, 1)
	if ok, err := st.FinishStopped(ctx, key, 100, []byte("nam-"+key), ckpt, nil, reached); err != nil || !ok {
		t.Fatalf("FinishStopped parent %s: ok=%v err=%v", key, ok, err)
	}
}

func TestInsertTrainMoreEligibilityUsesReached(t *testing.T) {
	wav := []byte("shared-capture-bytes")
	ckpt := []byte("parent-checkpoint-blob")

	t.Run("stopped parent: child above reached seeds at reached", func(t *testing.T) {
		st := openTest(t)
		ctx := context.Background()
		makeStoppedParent(t, st, "p", 100, 5, wav, ckpt) // reached=5, epochs=100
		if err := st.InsertJob(ctx, trainMoreChild("c", "p", 50, "standard"), wav); err != nil {
			t.Fatalf("child epochs 50 must be eligible (50 > reached 5): %v", err)
		}
		if child, _, _ := st.GetJob(ctx, "c"); child.StartEpoch == nil || *child.StartEpoch != 5 {
			t.Errorf("start_epoch = %v, want 5 (the parent's reached count)", child.StartEpoch)
		}
	})

	t.Run("stopped parent: child at reached is rejected", func(t *testing.T) {
		st := openTest(t)
		ctx := context.Background()
		makeStoppedParent(t, st, "p", 100, 5, wav, ckpt)
		err := st.InsertJob(ctx, trainMoreChild("c", "p", 5, "standard"), wav)
		if !errors.Is(err, ErrBaseUnavailable) {
			t.Fatalf("child epochs 5 (== reached) err = %v, want ErrBaseUnavailable", err)
		}
		var bu *BaseUnavailableError
		if !errors.As(err, &bu) || bu.Reason == "" {
			t.Errorf("err = %v, want a *BaseUnavailableError with a Reason", err)
		}
	})

	t.Run("stopped parent: child at original epochs is eligible", func(t *testing.T) {
		st := openTest(t)
		ctx := context.Background()
		makeStoppedParent(t, st, "p", 100, 5, wav, ckpt)
		if err := st.InsertJob(ctx, trainMoreChild("c", "p", 100, "standard"), wav); err != nil {
			t.Fatalf("child epochs 100 must be eligible (100 > reached 5): %v", err)
		}
		if child, _, _ := st.GetJob(ctx, "c"); child.StartEpoch == nil || *child.StartEpoch != 5 {
			t.Errorf("start_epoch = %v, want 5", child.StartEpoch)
		}
	})

	t.Run("natural parent: reached == epochs, unchanged behavior", func(t *testing.T) {
		st := openTest(t)
		ctx := context.Background()
		makeSucceededParent(t, st, "p", 200, "standard", wav, ckpt) // reached stamped = 200
		// epochs must exceed reached (== epochs here): 200 rejected, 400 ok at 200.
		if err := st.InsertJob(ctx, trainMoreChild("eq", "p", 200, "standard"), wav); !errors.Is(err, ErrBaseUnavailable) {
			t.Fatalf("child epochs 200 (== reached) err = %v, want ErrBaseUnavailable", err)
		}
		if err := st.InsertJob(ctx, trainMoreChild("c", "p", 400, "standard"), wav); err != nil {
			t.Fatalf("child epochs 400 must be eligible: %v", err)
		}
		if child, _, _ := st.GetJob(ctx, "c"); child.StartEpoch == nil || *child.StartEpoch != 200 {
			t.Errorf("start_epoch = %v, want 200 (== parent reached == epochs)", child.StartEpoch)
		}
	})

	t.Run("pre-v3 parent: NULL reached coalesces to epochs", func(t *testing.T) {
		st := openTest(t)
		ctx := context.Background()
		makeSucceededParent(t, st, "p", 200, "standard", wav, ckpt)
		// Simulate a row finished by a daemon predating stop: reached NULL.
		if _, err := st.db.ExecContext(ctx, "UPDATE jobs SET reached=NULL WHERE key='p'"); err != nil {
			t.Fatal(err)
		}
		// Eligibility falls back to epochs (200): 150 rejected, 250 ok at 200.
		if err := st.InsertJob(ctx, trainMoreChild("lo", "p", 150, "standard"), wav); !errors.Is(err, ErrBaseUnavailable) {
			t.Fatalf("child epochs 150 (<= epochs 200) err = %v, want ErrBaseUnavailable", err)
		}
		if err := st.InsertJob(ctx, trainMoreChild("c", "p", 250, "standard"), wav); err != nil {
			t.Fatalf("child epochs 250 must be eligible: %v", err)
		}
		if child, _, _ := st.GetJob(ctx, "c"); child.StartEpoch == nil || *child.StartEpoch != 200 {
			t.Errorf("start_epoch = %v, want 200 (COALESCE to epochs)", child.StartEpoch)
		}
	})
}
