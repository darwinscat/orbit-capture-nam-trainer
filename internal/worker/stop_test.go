// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/storetest"
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

// seedAndClaim queues a job and claims it through the store (a real claim_token),
// returning the running Job — the setup every direct-classify test needs (the
// finish* transitions are fenced on the token and state='running').
func (h *harness) seedAndClaim(t *testing.T, kind string, epochs int) jobs.Job {
	t.Helper()
	id := h.seed(t, kind, epochs)
	j, ok, err := h.store.ClaimNext(context.Background(), jobs.Lane(kind), testWorker, testInstance, storetest.NamVersion)
	if err != nil || !ok || j.ID != id {
		t.Fatalf("claim %d: ok=%v err=%v id=%d", id, ok, err, j.ID)
	}
	return j
}

func (h *harness) insertLog(t *testing.T, j jobs.Job, line string) {
	t.Helper()
	storetest.Exec(t, h.store, `INSERT INTO job_log (job_id, claim_token, line) VALUES ($1, $2::uuid, $3)`, j.ID, j.ClaimToken, line)
}

func (h *harness) resultRows(t *testing.T, id int64) int {
	t.Helper()
	return storetest.Count(t, h.store, `SELECT count(*) FROM job_result WHERE job_id = $1`, id)
}

// waitLog blocks until job_log for id contains sub (used to fence a stop until the
// stub has written its checkpoints — it writes them BEFORE any Epoch line).
func (h *harness) waitLog(t *testing.T, id int64, sub string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		lines, _ := h.store.JobLog(context.Background(), id)
		if logContains(lines, sub) {
			return
		}
		time.Sleep(15 * time.Millisecond)
	}
	t.Fatalf("job %d log never contained %q", id, sub)
}

// --- the stop happy path, end-to-end through the pool ---

func TestStopHappyPath(t *testing.T) {
	// A run that GOES ON finishing epochs: the stop comes back with the write an epoch makes, so a
	// stub that stopped producing them could never be told anything.
	h := newHarness(t, "train-keeps-going", time.Minute)
	id := h.seed(t, jobs.KindTrain, 100)
	h.start(t)

	h.waitRunningWithPGID(t, id)
	h.waitLog(t, id, "DRIVER: epoch_esr=2=", 5*time.Second)

	h.requestStop(t, id)

	j := h.waitState(t, id, jobs.StateSucceeded, 15*time.Second)
	if j.Reached == nil || *j.Reached < 3 {
		t.Fatalf("reached = %v, want the epoch it had kept by then", j.Reached)
	}
	reached := *j.Reached
	if j.ESR == nil {
		t.Error("the stopped run kept no ESR")
	}
	// A stop needs no state of its own: this row was ASKED to stop and it FINISHED, which is the
	// whole of the answer.
	if j.StopRequestedAt == nil || j.FinishedAt == nil {
		t.Errorf("stop answer: requested=%v finished=%v", j.StopRequestedAt, j.FinishedAt)
	}
	if nam := h.modelNam(t, id); len(nam) == 0 {
		t.Error("a stopped run must store the model of the epoch it kept")
	}
	ckpt, ok := h.takeCkpt(t, id)
	if !ok {
		t.Fatal("a stopped run must leave the take its weights (a continuation picks them up)")
	}
	assertValidZip(t, ckpt, "stored stop ckpt")
	if n := storetest.Count(t, h.store,
		`SELECT count(*) FROM job_result WHERE job_id = $1 AND epochs = $2`, id, reached); n != 1 {
		t.Error("job_result.epochs must be the epoch the run actually kept")
	}
	if lines, _ := h.store.JobLog(context.Background(), id); len(lines) == 0 {
		t.Error("job_log should be kept through a stop")
	}
}

// The newest checkpoint_last (epoch 5) is a torn zip mid-rotation; the harvest falls
// back to the intact previous pair (epoch 4), NOT to best.
func TestStopMidRotationTornNewest(t *testing.T) {
	h := newHarness(t, "", time.Minute)
	job := h.seedAndClaim(t, jobs.KindTrain, 100)
	storetest.Exec(t, h.store, `UPDATE jobs SET stop_requested_at = now() WHERE id = $1`, job.ID)

	scratch := t.TempDir()
	mkPair(t, scratch, "w", "checkpoint_last_epoch=0005_step=310.ckpt", `{}`, true)          // torn newest
	mkPair(t, scratch, "w", "checkpoint_last_epoch=0004_step=248.ckpt", `{"e4":true}`, false) // intact previous
	h.insertLog(t, job, "DRIVER: epoch_esr=4=0.03300000")
	h.recordEpoch(t, job, 4, 0.033)
	outdir := filepath.Join(scratch, "out")
	h.storeEpoch(t, job, scratch, nil)

	h.pool.classify(job, outdir, reasonStop, outcome{}, fmt.Errorf("killed"), 0)

	j := h.get(t, job.ID)
	if j.Reached == nil || *j.Reached != 5 {
		t.Errorf("reached = %v, want 5 (intact epoch 4 + 1)", j.Reached)
	}
	if j.ESR == nil || !approxF32(*j.ESR, 0.033) {
		t.Errorf("esr = %v, want 0.033 (epoch_esr=4=)", j.ESR)
	}
	if nam := h.modelNam(t, job.ID); string(nam) != `{"e4":true}` {
		t.Errorf("stored nam = %q, want the intact epoch-4 sibling", nam)
	}
}

// --- direct-classify harvest fallbacks (deterministic, no child race) ---

// No last pair survives (the only last is torn), so the harvest falls back to the BEST
// pair (epoch 3).
func TestStopBestPairFallback(t *testing.T) {
	h := newHarness(t, "", time.Minute)
	job := h.seedAndClaim(t, jobs.KindTrain, 50)
	storetest.Exec(t, h.store, `UPDATE jobs SET stop_requested_at = now() WHERE id = $1`, job.ID)

	scratch := t.TempDir()
	mkPair(t, scratch, "w", "checkpoint_last_epoch=0009_step=560.ckpt", `{}`, true)
	mkPair(t, scratch, "w", "checkpoint_best_epoch=0003_step=186_ESR=0.04173389_MSE=1.0e-03.ckpt", `{"best":true}`, false)
	h.insertLog(t, job, "DRIVER: epoch_esr=3=0.03500000")
	h.recordEpoch(t, job, 3, 0.035)
	outdir := filepath.Join(scratch, "out")
	h.storeEpoch(t, job, scratch, nil)

	h.pool.classify(job, outdir, reasonStop, outcome{}, fmt.Errorf("killed"), 0)

	j := h.get(t, job.ID)
	if j.State != jobs.StateSucceeded {
		t.Fatalf("state = %q, want succeeded (best-pair fallback)", j.State)
	}
	if j.Reached == nil || *j.Reached != 4 {
		t.Errorf("reached = %v, want 4 (best epoch 3 + 1)", j.Reached)
	}
	if j.ESR == nil || !approxF32(*j.ESR, 0.035) {
		t.Errorf("esr = %v, want 0.035 (epoch_esr=3=)", j.ESR)
	}
	if nam := h.modelNam(t, job.ID); string(nam) != `{"best":true}` {
		t.Errorf("stored nam = %q, want the best sibling", nam)
	}
	ckpt, ok := h.takeCkpt(t, job.ID)
	if !ok {
		t.Fatal("best-pair fallback must store a ckpt")
	}
	assertValidZip(t, ckpt, "best-pair ckpt")
}

// reasonStop with a COMPLETED outdir (the driver exported model.nam + rmtree'd its
// work dir before the kill landed) and only torn checkpoints — the natural result is
// kept, reached = epochs.
func TestStopCompletedOutdirIsNaturalSuccess(t *testing.T) {
	h := newHarness(t, "", time.Minute)

	t.Run("waitErr nil → natural-success rule (branch 1)", func(t *testing.T) {
		job := h.seedAndClaim(t, jobs.KindTrain, 42)
		scratch := t.TempDir()
		mkPair(t, scratch, "w", "checkpoint_last_epoch=0009_step=560.ckpt", `{}`, true) // torn
		outdir := filepath.Join(scratch, "out")
		if err := os.WriteFile(filepath.Join(outdir, "model.nam"), []byte(`{"final":true}`), 0o644); err != nil {
			t.Fatal(err)
		}
		h.pool.classify(job, outdir, reasonStop, outcome{}, nil, 0)

		j := h.get(t, job.ID)
		if j.State != jobs.StateSucceeded {
			t.Fatalf("state = %q, want succeeded", j.State)
		}
		if j.Reached == nil || *j.Reached != 42 {
			t.Errorf("reached = %v, want 42 (natural finish stamps epochs)", j.Reached)
		}
		if nam := h.modelNam(t, job.ID); string(nam) != `{"final":true}` {
			t.Errorf("stored nam = %q, want the exported model", nam)
		}
	})

	t.Run("waitErr set → stop-branch outdir fallback (branch 2)", func(t *testing.T) {
		job := h.seedAndClaim(t, jobs.KindTrain, 30)
		scratch := t.TempDir()
		mkPair(t, scratch, "w", "checkpoint_last_epoch=0009_step=560.ckpt", `{}`, true) // torn
		outdir := filepath.Join(scratch, "out")
		if err := os.WriteFile(filepath.Join(outdir, "model.nam"), []byte(`{"final":true}`), 0o644); err != nil {
			t.Fatal(err)
		}
		h.pool.classify(job, outdir, reasonStop, outcome{}, fmt.Errorf("killed after export"), 0)

		j := h.get(t, job.ID)
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
	job := h.seedAndClaim(t, jobs.KindTrain, 50)
	storetest.Exec(t, h.store, `UPDATE jobs SET stop_requested_at = now() WHERE id = $1`, job.ID)

	scratch := t.TempDir()
	mkPair(t, scratch, "w", "checkpoint_last_epoch=0009_step=560.ckpt", `{}`, true)                          // torn last
	mkPair(t, scratch, "w", "checkpoint_best_epoch=0003_step=186_ESR=0.04173389_MSE=1.0e-03.ckpt", ``, true) // torn best
	outdir := filepath.Join(scratch, "out")
	h.storeEpoch(t, job, scratch, nil)

	h.pool.classify(job, outdir, reasonStop, outcome{}, fmt.Errorf("killed"), 0)

	j := h.get(t, job.ID)
	if j.State != jobs.StateFailed {
		t.Fatalf("state = %q, want failed", j.State)
	}
	if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrStopFailed {
		t.Errorf("error_code = %v, want stop_failed", j.ErrorCode)
	}
	if h.resultRows(t, job.ID) != 0 {
		t.Error("a stop_failed must store no result row")
	}
}

// A stall-kill already decided the entry (reason=stall, first-sticks); a stop request
// during teardown must NOT flip the outcome — it ends 'stalled', the stop stays pending.
func TestStallFirstThenStop(t *testing.T) {
	h := newHarness(t, "silent-hang", 400*time.Millisecond)
	id := h.seed(t, jobs.KindTrain, 100)
	h.start(t)
	h.waitRunningWithPGID(t, id)

	// silent-hang writes no checkpoints, so the stop can only be acknowledged pending.
	h.requestStop(t, id)

	j := h.waitState(t, id, jobs.StateFailed, 10*time.Second)
	if j.ErrorCode == nil || *j.ErrorCode != jobs.ErrStalled {
		t.Errorf("error_code = %v, want stalled (stall reason outranks a later stop)", j.ErrorCode)
	}
}

// The app takes the row away during the stop's finalize (it cancelled a "stale"
// running row): the fenced FinishSucceeded finds no row, nothing is written, and the
// app's cancelled state stands.
func TestStopLostRowDuringFinalizeWritesNothing(t *testing.T) {
	h := newHarness(t, "", time.Minute)
	job := h.seedAndClaim(t, jobs.KindTrain, 50)

	scratch := t.TempDir()
	mkPair(t, scratch, "w", "checkpoint_last_epoch=0005_step=310.ckpt", `{}`, false) // a valid, harvestable pair
	h.insertLog(t, job, "DRIVER: epoch_esr=5=0.03100000")
	h.recordEpoch(t, job, 5, 0.031)
	outdir := filepath.Join(scratch, "out")

	storetest.Exec(t, h.store,
		`UPDATE jobs SET state = 'cancelled', finished_at = now(), error_code = 'cancelled', claim_token = NULL WHERE id = $1`, job.ID)

	h.pool.classify(job, outdir, reasonStop, outcome{}, fmt.Errorf("killed"), 0)

	if j := h.get(t, job.ID); j.State != jobs.StateCancelled {
		t.Errorf("state = %q, want the app's cancelled to stand", j.State)
	}
	if h.resultRows(t, job.ID) != 0 {
		t.Error("no job_result may land on a row that is no longer ours")
	}
}

// Regression through the POOL path: a natural finish stamps reached == epochs.
func TestNaturalFinishStampsReachedThroughPool(t *testing.T) {
	h := newHarness(t, "train-ok", time.Minute)
	id := h.seed(t, jobs.KindTrain, 5)
	h.start(t)

	j := h.waitState(t, id, jobs.StateSucceeded, 15*time.Second)
	if j.Reached == nil || *j.Reached != 5 {
		t.Errorf("reached = %v, want 5 (natural finish stamps epochs)", j.Reached)
	}
}

// The full stop→continue chain THROUGH THE POOL: a pool-stopped parent's stored zip
// checkpoint seeds a train_more (job_resume snapshotted from job_result, as the app
// does), and the continuation resumes numbering at the parent's reached.
func TestStopThenTrainMoreChainThroughPool(t *testing.T) {
	h := newHarness(t, "auto", time.Minute)
	tk := storetest.SeedTake(t, h.store, validWav)
	parent := storetest.InsertJob(t, h.store, storetest.JobSpec{Take: tk, Kind: jobs.KindTrain, Epochs: 6})
	h.start(t)

	h.waitRunningWithPGID(t, parent)
	h.waitLog(t, parent, "DRIVER: epoch_esr=2=", 5*time.Second)
	h.requestStop(t, parent)
	j := h.waitState(t, parent, jobs.StateSucceeded, 15*time.Second)
	if j.Reached == nil || *j.Reached < 3 {
		t.Fatalf("parent reached = %v, want the epoch it had kept", j.Reached)
	}
	reached := *j.Reached

	child := h.seedTrainMore(t, parent, tk, 12)
	h.pool.Notify()
	cj := h.waitState(t, child, jobs.StateSucceeded, 20*time.Second)
	if cj.StartEpoch == nil || *cj.StartEpoch != reached {
		t.Errorf("child start_epoch = %v, want %d (the take's weights)", cj.StartEpoch, reached)
	}
	lines, _ := h.store.JobLog(context.Background(), child)
	if first := firstEpochLine(lines); int64(first) != reached {
		t.Errorf("child first Epoch line = %d, want %d (resumed where the take was)", first, reached)
	}
	if cj.Reached == nil || *cj.Reached != 12 {
		t.Errorf("child reached = %v, want 12 (natural finish stamps epochs)", cj.Reached)
	}
}

// A PAUSE IS AN EARLY STOP, NOT A DISCARD. "Pause now" is what somebody presses when they need the
// GPU of the machine they are sitting at, and until this it threw the run away: the reason shared a
// branch with a daemon shutdown and requeued, while the pair holding the last COMPLETED epoch sat on
// disk, unread, waiting for the scratch wipe. An hour and a half of GPU, for a click that means "give
// me my machine back", not "lose my work".
func TestPauseKeepsTheLastCompletedEpoch(t *testing.T) {
	h := newHarness(t, "", time.Minute)
	job := h.seedAndClaim(t, jobs.KindTrain, 800)

	scratch := t.TempDir()
	mkPair(t, scratch, "w", "checkpoint_last_epoch=0006_step=420.ckpt", `{}`, false)
	h.insertLog(t, job, "DRIVER: epoch_esr=6=0.02100000")
	h.recordEpoch(t, job, 6, 0.021)
	outdir := filepath.Join(scratch, "out")
	h.storeEpoch(t, job, scratch, nil)

	h.pool.classify(job, outdir, reasonPause, outcome{}, fmt.Errorf("killed"), 0)

	j := h.get(t, job.ID)
	if j.State != jobs.StateSucceeded {
		t.Fatalf("state = %q, want succeeded — a pause keeps what was trained", j.State)
	}
	if j.Reached == nil || *j.Reached != 7 {
		t.Errorf("reached = %v, want 7 (last epoch 6 + 1) out of 800", j.Reached)
	}
	if _, ok := h.takeCkpt(t, job.ID); !ok {
		t.Error("a paused run must store its checkpoint, or Continue has nothing to resume from")
	}
}

// …but before the first epoch there is nothing to keep — torch import, dataset build, epoch 0 — and
// that is the ordinary case in the first minutes of a run. It is not a failure: the job goes back to
// the queue and is trained when the machine is free again.
func TestPauseBeforeTheFirstEpochRequeues(t *testing.T) {
	h := newHarness(t, "", time.Minute)
	job := h.seedAndClaim(t, jobs.KindTrain, 800)

	outdir := filepath.Join(t.TempDir(), "out") // no pair, no exported model
	h.pool.classify(job, outdir, reasonPause, outcome{}, fmt.Errorf("killed"), 0)

	j := h.get(t, job.ID)
	if j.State != jobs.StateQueued {
		t.Fatalf("state = %q, want queued — nothing was trained yet, so nothing is lost", j.State)
	}
	if j.Worker != nil || j.ClaimToken != "" {
		t.Errorf("a requeued attempt keeps no claim: worker=%v token=%q", j.Worker, j.ClaimToken)
	}
	// …and it is CLAIMABLE. A requeue that left the old attempt's stop ask on the row produced a
	// queued job nothing could ever pick up (ClaimNext skips stop_requested_at) and nothing could
	// re-queue either — a take stuck in the lane until somebody terminated it by hand.
	if j.StopRequestedAt != nil {
		t.Error("the stop asked of the attempt that is gone must not follow it into the queue")
	}
}

// A DAEMON SHUTDOWN IS NOT A PAUSE. Nobody asked the training to stop — the daemon is going away —
// so the run goes back to the queue whole, to be finished by whoever picks it up. Harvesting here
// would mean restarting the daemon silently truncated every run it was holding.
func TestShutdownStillRequeues(t *testing.T) {
	h := newHarness(t, "", time.Minute)
	job := h.seedAndClaim(t, jobs.KindTrain, 800)

	scratch := t.TempDir()
	mkPair(t, scratch, "w", "checkpoint_last_epoch=0006_step=420.ckpt", `{}`, false)
	outdir := filepath.Join(scratch, "out")

	h.pool.classify(job, outdir, reasonShutdown, outcome{}, fmt.Errorf("killed"), 0)

	if j := h.get(t, job.ID); j.State != jobs.StateQueued {
		t.Fatalf("state = %q, want queued — a shutdown is not an instruction to stop training", j.State)
	}
}

// A PAUSE MUST OUTLIVE THE PROCESS. It lived in memory, so every restart resumed — and a restart is
// what an upgrade, a config re-read from the tray, and a crash all are. The person who paused this
// trainer was sitting at the machine wanting their GPU; a relaunch nobody asked for gave the machine
// back to the queue. Only a hand lifts a pause.
func TestAPauseIsRememberedAcrossARestart(t *testing.T) {
	h := newHarness(t, "", 0)
	file := filepath.Join(h.base, "paused")

	if PauseWasRemembered(file) {
		t.Fatal("a fresh trainer must not start paused")
	}
	h.pool.Pause(false)
	if !PauseWasRemembered(file) {
		t.Fatal("a paused trainer must come up paused after a restart")
	}

	// …and a hand lifting it forgets it, so the NEXT start claims again.
	h.pool.Resume()
	if PauseWasRemembered(file) {
		t.Fatal("Resume must forget the pause, not leave the trainer paused for ever")
	}

	// The escalation remembers it too: "pause now" is the same gate, asked harder.
	h.pool.Pause(true)
	if !PauseWasRemembered(file) {
		t.Fatal("pause now must be remembered as well")
	}
}

// …and switching the memory off is what a test pool wants: no path, no file, no surprise.
func TestAPoolWithNoPauseFileRemembersNothing(t *testing.T) {
	if PauseWasRemembered("") {
		t.Fatal("an empty path is not a remembered pause")
	}
	if PauseWasRemembered(filepath.Join(t.TempDir(), "never-written")) {
		t.Fatal("a missing file is not a remembered pause")
	}
}

// storeEpoch runs the saver once over `scratch`, exactly as a finished epoch does, so a test that
// drives classify directly starts from the state a real run would be in: the take's weights in the
// library. Since 0007 a stop READS that row rather than going hunting on disk, so the disk-selection
// subtleties (skip the torn newest, fall back to the best) belong to the saver — which is what this
// puts under test.
func (h *harness) storeEpoch(t *testing.T, job jobs.Job, scratch string, esr *float64) {
	t.Helper()
	s := h.pool.startCkptSaver(context.Background(), job, scratch)
	s.nudge(-1, esr)
	s.stop()
}

// recordEpoch writes the job_epochs row the daemon writes for every epoch line it parses. A test that
// fabricates the line has to fabricate the row: since 0007 the figure attached to a stored checkpoint
// is read from there, not re-parsed out of the log.
func (h *harness) recordEpoch(t *testing.T, job jobs.Job, epoch int, esr float64) {
	t.Helper()
	if err := h.store.RecordEpoch(context.Background(), job.ID, job.ClaimToken, epoch, &esr, 1); err != nil {
		t.Fatalf("record epoch: %v", err)
	}
}

// approxF32 compares at the precision job_epochs.esr actually keeps: the column is a `real`, so a
// figure that went through it comes back with about seven significant digits, not with the exactness
// of the text it was parsed from.
func approxF32(a, b float64) bool { return math.Abs(a-b) <= math.Abs(b)*1e-6+1e-12 }

// (Two tests lived here and are gone with what they described.
//
// "pending then armed at the first checkpoint" was the stop's own state machine: a stop that arrived
// before any checkpoint existed was acknowledged `pending`, then `armed` a moment before the kill.
// A stop is only ever HEARD at a checkpoint write now, so there is always one by then, and nothing
// reads a value that exists for a millisecond between two round trips.
//
// "a stop on a probe is refused" answered a stop nobody can send to a three-second job: there is no
// channel to a probe, because a probe writes no checkpoints. It runs to its verdict, which is what
// `refused` meant.)
