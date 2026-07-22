// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"orbit-capture-nam-trainer/internal/jobs"
)

// --- helpers for the early-stop harvest tests ---

// writeRealZip writes a minimal but REAL zip archive (torch checkpoints are zips), so
// the harvest's zip.NewReader accepts it. torn truncates the bytes to drop the
// end-of-central-directory record — a SIGKILL-frozen partial write zip.NewReader
// refuses to open.
func writeRealZip(t *testing.T, path string, torn bool) {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, err := zw.Create("archive/data.pkl")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte("weights")); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()
	if torn {
		b = b[:len(b)/2]
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// mkPair lays down a checkpoint pair under <scratch>/out/<sub>/checkpoints: a real (or
// torn) zip .ckpt plus, when nam != "", its same-stem .nam sibling.
func mkPair(t *testing.T, scratch, sub, ckptName, nam string, torn bool) {
	t.Helper()
	dir := filepath.Join(scratch, "out", sub, "checkpoints")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeRealZip(t, filepath.Join(dir, ckptName), torn)
	if nam != "" {
		stem := ckptName[:len(ckptName)-len(".ckpt")]
		if err := os.WriteFile(filepath.Join(dir, stem+".nam"), []byte(nam), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func assertValidZip(t *testing.T, b []byte, what string) {
	t.Helper()
	if len(b) == 0 {
		t.Fatalf("%s: empty, want a real zip checkpoint", what)
	}
	if _, err := zip.NewReader(bytes.NewReader(b), int64(len(b))); err != nil {
		t.Fatalf("%s: not a valid zip: %v", what, err)
	}
}

// seedAndClaim inserts a queued job and flips it to running via the store (started_at
// stamped), returning the running Job — the setup every direct-classify test needs (the
// finish* transitions are gated on state='running').
func (h *harness) seedAndClaim(t *testing.T, key, kind string, epochs int) jobs.Job {
	t.Helper()
	h.seed(t, key, kind, epochs)
	j, ok, err := h.store.ClaimNextQueued(context.Background(), 1, kind)
	if err != nil || !ok || j.Key != key {
		t.Fatalf("claim %s: ok=%v err=%v key=%q", key, ok, err, j.Key)
	}
	return j
}

func (h *harness) insertLog(t *testing.T, key, line string) {
	t.Helper()
	if _, err := h.store.DB().ExecContext(context.Background(),
		`INSERT INTO job_log(job_key, line) VALUES(?, ?)`, key, line); err != nil {
		t.Fatal(err)
	}
}

func (h *harness) resultRows(t *testing.T, key string) int {
	t.Helper()
	var n int
	if err := h.store.DB().QueryRowContext(context.Background(),
		`SELECT COUNT(*) FROM results WHERE job_key = ?`, key).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func (h *harness) modelNam(t *testing.T, key string) []byte {
	t.Helper()
	nam, ok, err := h.store.ModelBytes(context.Background(), key)
	if err != nil {
		t.Fatalf("ModelBytes: %v", err)
	}
	if !ok {
		return nil
	}
	return nam
}

// waitLog blocks until job_log for key contains sub (used to fence a stop until the
// stub has written its checkpoints — it writes them BEFORE any Epoch line).
func (h *harness) waitLog(t *testing.T, key, sub string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		lines, _ := h.store.JobLog(context.Background(), key)
		if logContains(lines, sub) {
			return
		}
		time.Sleep(15 * time.Millisecond)
	}
	t.Fatalf("job %s log never contained %q", key, sub)
}

// --- StopJob request-side sentinels ---

func TestStopJobUnknownKey(t *testing.T) {
	h := newHarness(t, "", 0)
	if err := h.pool.StopJob("nope"); !errors.Is(err, ErrNotRunning) {
		t.Fatalf("StopJob(unknown) = %v, want ErrNotRunning", err)
	}
}

func TestStopJobNoCheckpointDoesNotKill(t *testing.T) {
	h := newHarness(t, "", 0)
	s := t.TempDir()
	if err := os.MkdirAll(filepath.Join(s, "out"), 0o755); err != nil {
		t.Fatal(err)
	}
	e := &procEntry{pgid: 424242, scratch: s} // scratch has no checkpoints yet
	h.pool.register("k", e)

	if err := h.pool.StopJob("k"); !errors.Is(err, ErrNoCheckpoint) {
		t.Fatalf("StopJob(no ckpt) = %v, want ErrNoCheckpoint", err)
	}
	// No kill happened: the entry's reason is still unset.
	e.mu.Lock()
	reason := e.reason
	e.mu.Unlock()
	if reason != "" {
		t.Errorf("entry reason = %q after a no-checkpoint StopJob, want unset (no kill)", reason)
	}
}

// --- the stop happy path, end-to-end through the pool ---

func TestStopHappyPath(t *testing.T) {
	h := newHarness(t, "train-hang-with-ckpts", time.Minute)
	h.seed(t, "k", jobs.KindTrain, 100)
	h.start(t)

	// Running with a pid, and the full checkpoint set on disk (fenced by the epoch-5
	// esr line — the stub writes checkpoints before it prints anything).
	waitFor(t, 5*time.Second, func() bool {
		j := h.get(t, "k")
		return j.State == jobs.StateRunning && j.PID != nil
	}, "job never reached running with a pid")
	h.waitLog(t, "k", "DRIVER: epoch_esr=5=", 5*time.Second)

	if err := h.pool.StopJob("k"); err != nil {
		t.Fatalf("StopJob = %v, want nil", err)
	}

	j := h.waitState(t, "k", jobs.StateSucceeded, 10*time.Second)

	if j.Reached == nil || *j.Reached != 6 {
		t.Errorf("reached = %v, want 6 (last epoch 5 + 1)", j.Reached)
	}
	if j.ESR == nil || !approx(*j.ESR, 0.031) {
		t.Errorf("esr = %v, want 0.031 (epoch_esr=5= log line)", j.ESR)
	}
	if !j.HasModel {
		t.Error("a stopped run must store a model (the last pair's .nam)")
	}
	if nam := h.modelNam(t, "k"); string(nam) != `{}` {
		t.Errorf("stored nam = %q, want %q (the last pair's sibling)", nam, `{}`)
	}
	ckpt, ok := h.resultCkpt(t, "k")
	if !ok {
		t.Fatal("a stopped run must store the last checkpoint (continuation seed)")
	}
	assertValidZip(t, ckpt, "stored stop ckpt")
	// Run-input dropped, history kept.
	if _, ok, _ := h.store.AudioBlob(context.Background(), "k"); ok {
		t.Error("capture blob should be gone after a stop")
	}
	if h.hasResumeCkpt(t, "k") {
		t.Error("resume snapshot should be gone after a stop")
	}
	if lines, _ := h.store.JobLog(context.Background(), "k"); len(lines) == 0 {
		t.Error("job_log should be kept through a stop")
	}
}

// The newest checkpoint_last (epoch 5) is a torn zip mid-rotation; the harvest falls
// back to the intact previous pair (epoch 4), NOT to best.
func TestStopMidRotationTornNewest(t *testing.T) {
	h := newHarness(t, "train-hang-with-ckpts-torn", time.Minute)
	h.seed(t, "k", jobs.KindTrain, 100)
	h.start(t)

	waitFor(t, 5*time.Second, func() bool {
		j := h.get(t, "k")
		return j.State == jobs.StateRunning && j.PID != nil
	}, "job never reached running with a pid")
	h.waitLog(t, "k", "DRIVER: epoch_esr=5=", 5*time.Second)

	if err := h.pool.StopJob("k"); err != nil {
		t.Fatalf("StopJob = %v, want nil", err)
	}

	j := h.waitState(t, "k", jobs.StateSucceeded, 10*time.Second)
	if j.Reached == nil || *j.Reached != 5 {
		t.Errorf("reached = %v, want 5 (intact epoch 4 + 1)", j.Reached)
	}
	if j.ESR == nil || !approx(*j.ESR, 0.033) {
		t.Errorf("esr = %v, want 0.033 (epoch_esr=4=)", j.ESR)
	}
	if nam := h.modelNam(t, "k"); string(nam) != `{"e4":true}` {
		t.Errorf("stored nam = %q, want the intact epoch-4 sibling", nam)
	}
}

// --- direct-classify harvest fallbacks (deterministic, no child race) ---

// No last pair survives (the only last is torn), so the harvest falls back to the BEST
// pair (epoch 3).
func TestStopBestPairFallback(t *testing.T) {
	h := newHarness(t, "", time.Minute)
	job := h.seedAndClaim(t, "k", jobs.KindTrain, 50)

	scratch := t.TempDir()
	// torn newest last, no intact previous → no qualifying last pair.
	mkPair(t, scratch, "w", "checkpoint_last_epoch=0009_step=560.ckpt", `{}`, true)
	// the best pair (min-ESR), intact → the fallback winner.
	mkPair(t, scratch, "w", "checkpoint_best_epoch=0003_step=186_ESR=0.04173389_MSE=1.0e-03.ckpt", `{"best":true}`, false)
	h.insertLog(t, "k", "DRIVER: epoch_esr=3=0.03500000")
	outdir := filepath.Join(scratch, "out")

	h.pool.classify(job, outdir, reasonStop, outcome{}, fmt.Errorf("killed"))

	j := h.get(t, "k")
	if j.State != jobs.StateSucceeded {
		t.Fatalf("state = %q, want succeeded (best-pair fallback)", j.State)
	}
	if j.Reached == nil || *j.Reached != 4 {
		t.Errorf("reached = %v, want 4 (best epoch 3 + 1)", j.Reached)
	}
	if j.ESR == nil || !approx(*j.ESR, 0.035) {
		t.Errorf("esr = %v, want 0.035 (epoch_esr=3=)", j.ESR)
	}
	if nam := h.modelNam(t, "k"); string(nam) != `{"best":true}` {
		t.Errorf("stored nam = %q, want the best sibling", nam)
	}
	ckpt, ok := h.resultCkpt(t, "k")
	if !ok {
		t.Fatal("best-pair fallback must store a ckpt")
	}
	assertValidZip(t, ckpt, "best-pair ckpt")
}

// The deterministic direct-classify test the plan demands: reasonStop with a COMPLETED
// outdir (the driver exported model.nam + rmtree'd its work dir before the kill landed)
// and only torn checkpoints — the natural result is kept, reached = epochs (D3).
func TestStopCompletedOutdirIsNaturalSuccess(t *testing.T) {
	h := newHarness(t, "", time.Minute)

	t.Run("waitErr nil → natural-success rule (branch 1)", func(t *testing.T) {
		job := h.seedAndClaim(t, "a", jobs.KindTrain, 42)
		scratch := t.TempDir()
		mkPair(t, scratch, "w", "checkpoint_last_epoch=0009_step=560.ckpt", `{}`, true) // torn
		outdir := filepath.Join(scratch, "out")
		if err := os.WriteFile(filepath.Join(outdir, "model.nam"), []byte(`{"final":true}`), 0o644); err != nil {
			t.Fatal(err)
		}
		h.pool.classify(job, outdir, reasonStop, outcome{}, nil)

		j := h.get(t, "a")
		if j.State != jobs.StateSucceeded {
			t.Fatalf("state = %q, want succeeded", j.State)
		}
		if j.Reached == nil || *j.Reached != 42 {
			t.Errorf("reached = %v, want 42 (natural finish stamps epochs)", j.Reached)
		}
		if nam := h.modelNam(t, "a"); string(nam) != `{"final":true}` {
			t.Errorf("stored nam = %q, want the exported model", nam)
		}
	})

	t.Run("waitErr set → stop-branch outdir fallback (branch 2)", func(t *testing.T) {
		job := h.seedAndClaim(t, "b", jobs.KindTrain, 30)
		scratch := t.TempDir()
		mkPair(t, scratch, "w", "checkpoint_last_epoch=0009_step=560.ckpt", `{}`, true) // torn
		outdir := filepath.Join(scratch, "out")
		if err := os.WriteFile(filepath.Join(outdir, "model.nam"), []byte(`{"final":true}`), 0o644); err != nil {
			t.Fatal(err)
		}
		h.pool.classify(job, outdir, reasonStop, outcome{}, fmt.Errorf("killed after export"))

		j := h.get(t, "b")
		if j.State != jobs.StateSucceeded {
			t.Fatalf("state = %q, want succeeded (outdir fallback)", j.State)
		}
		if j.Reached == nil || *j.Reached != 30 {
			t.Errorf("reached = %v, want 30", j.Reached)
		}
	})
}

// All checkpoints torn AND no exported model → failed/stop_failed.
func TestStopAllTornNothingFails(t *testing.T) {
	h := newHarness(t, "", time.Minute)
	job := h.seedAndClaim(t, "k", jobs.KindTrain, 50)

	scratch := t.TempDir()
	mkPair(t, scratch, "w", "checkpoint_last_epoch=0009_step=560.ckpt", `{}`, true)                          // torn last
	mkPair(t, scratch, "w", "checkpoint_best_epoch=0003_step=186_ESR=0.04173389_MSE=1.0e-03.ckpt", ``, true) // torn best
	outdir := filepath.Join(scratch, "out")

	h.pool.classify(job, outdir, reasonStop, outcome{}, fmt.Errorf("killed"))

	j := h.get(t, "k")
	if j.State != jobs.StateFailed {
		t.Fatalf("state = %q, want failed", j.State)
	}
	if j.ErrorCode == nil || *j.ErrorCode != "stop_failed" {
		t.Errorf("error_code = %v, want stop_failed", j.ErrorCode)
	}
	if h.resultRows(t, "k") != 0 {
		t.Error("a stop_failed must store no results row")
	}
}

// A stall-kill already decided the entry (reason=stall, first-sticks); a StopJob during
// teardown must NOT flip the outcome to stop_failed or succeeded — it ends 'stalled'.
func TestStallFirstThenStop(t *testing.T) {
	h := newHarness(t, "silent-hang", 300*time.Millisecond)
	h.seed(t, "k", jobs.KindTrain, 100)
	h.start(t)

	waitFor(t, 5*time.Second, func() bool {
		return h.get(t, "k").State == jobs.StateRunning
	}, "job never reached running")

	// silent-hang writes no checkpoints, so StopJob can only report ErrNoCheckpoint (or
	// ErrNotRunning once the stall already finalized) — never a successful stop.
	if err := h.pool.StopJob("k"); err == nil {
		t.Fatal("StopJob returned nil for a checkpoint-less run, want an error")
	}

	j := h.waitState(t, "k", jobs.StateFailed, 5*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != "stalled" {
		t.Errorf("error_code = %v, want stalled (stall reason outranks a later stop)", j.ErrorCode)
	}
}

// A DELETE landing during the stop's finalize wins: FinishStopped's running gate finds
// no row, done() takes its "row already gone" path, and nothing is resurrected.
func TestStopDeleteDuringFinalizeRowGone(t *testing.T) {
	h := newHarness(t, "", time.Minute)
	job := h.seedAndClaim(t, "k", jobs.KindTrain, 50)

	scratch := t.TempDir()
	mkPair(t, scratch, "w", "checkpoint_last_epoch=0005_step=310.ckpt", `{}`, false) // a valid, harvestable pair
	h.insertLog(t, "k", "DRIVER: epoch_esr=5=0.03100000")
	outdir := filepath.Join(scratch, "out")

	// The DELETE lands before classify writes the terminal state.
	if _, err := h.store.DeleteJob(context.Background(), "k"); err != nil {
		t.Fatalf("DeleteJob: %v", err)
	}

	h.pool.classify(job, outdir, reasonStop, outcome{}, fmt.Errorf("killed"))

	if _, ok, _ := h.store.GetJob(context.Background(), "k"); ok {
		t.Error("row should stay gone — no resurrection by the stop finalize")
	}
	if h.resultRows(t, "k") != 0 {
		t.Error("no results row should exist for a deleted job")
	}
}

// A double StopJob must not double-finalize: the second call is a no-op kill (or
// ErrNotRunning once unregistered), and the observable is exactly ONE terminal write —
// succeeded, a single results row.
func TestStopIsIdempotent(t *testing.T) {
	h := newHarness(t, "train-hang-with-ckpts", time.Minute)
	h.seed(t, "k", jobs.KindTrain, 100)
	h.start(t)

	waitFor(t, 5*time.Second, func() bool {
		j := h.get(t, "k")
		return j.State == jobs.StateRunning && j.PID != nil
	}, "job never reached running with a pid")
	h.waitLog(t, "k", "DRIVER: epoch_esr=5=", 5*time.Second)

	if err := h.pool.StopJob("k"); err != nil {
		t.Fatalf("first StopJob = %v, want nil", err)
	}
	// The second call is nil (entry still registered), ErrNotRunning (finalized +
	// unregistered), or — in a hair-thin window — ErrNoCheckpoint (entry captured
	// pre-unregister, scratch swept mid-walk). All are fine; what must hold is a
	// single finalize.
	if err := h.pool.StopJob("k"); err != nil &&
		!errors.Is(err, ErrNotRunning) && !errors.Is(err, ErrNoCheckpoint) {
		t.Fatalf("second StopJob = %v, want nil/ErrNotRunning/ErrNoCheckpoint", err)
	}

	j := h.waitState(t, "k", jobs.StateSucceeded, 10*time.Second)
	if j.Reached == nil || *j.Reached != 6 {
		t.Errorf("reached = %v, want 6", j.Reached)
	}
	if n := h.resultRows(t, "k"); n != 1 {
		t.Errorf("results rows = %d, want exactly 1 (single finalize)", n)
	}
}

// Regression through the POOL path: a natural finish still stamps reached == epochs
// (the step-1 change must hold via classify's branch (1), not only the store test).
func TestNaturalFinishStampsReachedThroughPool(t *testing.T) {
	h := newHarness(t, "train-ok", time.Minute)
	h.seed(t, "k", jobs.KindTrain, 5)
	h.start(t)

	j := h.waitState(t, "k", jobs.StateSucceeded, 10*time.Second)
	if j.Reached == nil || *j.Reached != 5 {
		t.Errorf("reached = %v, want 5 (natural finish stamps epochs)", j.Reached)
	}
}

// The full stop→continue chain THROUGH THE POOL (crew F-1): a pool-stopped
// parent's stored zip checkpoint seeds a train_more via the real materialize +
// --resume-from path, and the continuation resumes numbering at the parent's
// reached. Mode "auto": epochs==6 → train-hang-with-ckpts for the parent;
// --resume-from present → resume_ok for the child (it reads the zip ckpt's
// resume_at entry).
func TestStopThenTrainMoreChainThroughPool(t *testing.T) {
	h := newHarness(t, "auto", time.Minute)
	h.seed(t, "parent", jobs.KindTrain, 6)
	h.start(t)

	waitFor(t, 5*time.Second, func() bool {
		j := h.get(t, "parent")
		return j.State == jobs.StateRunning && j.PID != nil
	}, "parent never reached running with a pid")
	h.waitLog(t, "parent", "DRIVER: epoch_esr=5=", 5*time.Second)
	if err := h.pool.StopJob("parent"); err != nil {
		t.Fatalf("StopJob = %v, want nil", err)
	}
	j := h.waitState(t, "parent", jobs.StateSucceeded, 10*time.Second)
	if j.Reached == nil || *j.Reached != 6 {
		t.Fatalf("parent reached = %v, want 6 (stopped after epoch 5)", j.Reached)
	}

	// Continue 6→12: eligibility and start_epoch key off reached (6), and the
	// child resumes exactly there.
	h.seedTrainMore(t, "child", "parent", 12, []byte("capture-bytes")) // seed()'s wav
	h.pool.Notify()
	cj := h.waitState(t, "child", jobs.StateSucceeded, 15*time.Second)
	if cj.StartEpoch == nil || *cj.StartEpoch != 6 {
		t.Errorf("child start_epoch = %v, want 6 (parent's reached)", cj.StartEpoch)
	}
	lines, _ := h.store.JobLog(context.Background(), "child")
	if first := firstEpochLine(lines); first != 6 {
		t.Errorf("child first Epoch line = %d, want 6 (resumed at reached)", first)
	}
	if cj.Reached == nil || *cj.Reached != 12 {
		t.Errorf("child reached = %v, want 12 (natural finish stamps epochs)", cj.Reached)
	}
}
