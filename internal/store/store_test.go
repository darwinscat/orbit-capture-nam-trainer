// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/storetest"
)

const (
	workerA = "studio.local"
	workerB = "laptop.local"
)

var prov = store.Provenance{
	NamVersion:   storetest.NamVersion,
	DriverSHA256: "1111111111111111111111111111111111111111111111111111111111111111",
	SignalSHA256: storetest.SignalSHA,
}

// openWithWorker opens a private schema and registers the worker rows the jobs.worker
// FK needs (a claim stamps worker = name).
func openWithWorker(t *testing.T) *store.Store {
	t.Helper()
	st := storetest.Open(t)
	for _, name := range []string{workerA, workerB} {
		if _, _, err := st.Heartbeat(context.Background(), store.WorkerInfo{
			Name: name, Instance: "inst-" + name, SchemaVersion: store.SupportedQueueContract,
			TrainCap: 1, ProbeCap: 1}); err != nil {
			t.Fatalf("heartbeat %s: %v", name, err)
		}
	}
	return st
}

// queue inserts a queued job on a fresh take. at orders it (queued_at). A
// train_more gets start_epoch = epochs/2 (the schema requires one).
func queue(t *testing.T, st *store.Store, kind string, epochs int, priority string, at time.Time) int64 {
	t.Helper()
	tk := storetest.SeedTake(t, st, []byte("wav-"+time.Now().Format(time.RFC3339Nano)+"-"+kind))
	var start *int64
	if kind == jobs.KindTrainMore {
		s := int64(epochs / 2)
		start = &s
	}
	return storetest.InsertJob(t, st, storetest.JobSpec{Take: tk, Kind: kind, Epochs: epochs, Priority: priority, QueuedAt: &at, StartEpoch: start})
}

// claim claims the next job of lane for workerA.
func claim(t *testing.T, st *store.Store, lane string) (jobs.Job, bool) {
	t.Helper()
	j, ok, err := st.ClaimNext(context.Background(), lane, workerA, "inst-a", storetest.NamVersion)
	if err != nil {
		t.Fatalf("ClaimNext(%s): %v", lane, err)
	}
	return j, ok
}

func mustClaim(t *testing.T, st *store.Store, lane string) jobs.Job {
	t.Helper()
	j, ok := claim(t, st, lane)
	if !ok {
		t.Fatalf("ClaimNext(%s): nothing claimed", lane)
	}
	return j
}

func get(t *testing.T, st *store.Store, id int64) jobs.Job {
	t.Helper()
	j, ok, err := st.GetJob(context.Background(), id)
	if err != nil || !ok {
		t.Fatalf("GetJob(%d): ok=%v err=%v", id, ok, err)
	}
	return j
}

func TestClaimNextDrainOrderAndStamps(t *testing.T) {
	st := openWithWorker(t)
	base := time.Date(2026, 8, 21, 10, 0, 0, 0, time.UTC)
	b := queue(t, st, jobs.KindTrain, 100, "normal", base.Add(2*time.Minute))
	a := queue(t, st, jobs.KindTrain, 100, "normal", base.Add(1*time.Minute))
	c := queue(t, st, jobs.KindTrainMore, 100, "normal", base.Add(3*time.Minute)) // shares the train lane
	hi := queue(t, st, jobs.KindTrain, 100, "high", base.Add(9*time.Minute))      // priority beats age
	lo := queue(t, st, jobs.KindTrain, 100, "low", base)                          // low drains last

	// A recovered row may still carry stale progress; the claim must clear it.
	storetest.Exec(t, st, `UPDATE jobs SET epoch = 37, s_per_epoch = 4.2 WHERE id = $1`, a)

	want := []int64{hi, a, b, c, lo}
	for i, id := range want {
		j := mustClaim(t, st, jobs.LaneTrain)
		if j.ID != id {
			t.Fatalf("claim #%d = job %d, want %d (drain order priority, queued_at, id)", i+1, j.ID, id)
		}
		if j.State != jobs.StateRunning || j.Worker == nil || *j.Worker != workerA ||
			j.WorkerInstance == nil || *j.WorkerInstance != "inst-a" || len(j.ClaimToken) != 36 {
			t.Errorf("claimed row not stamped: state=%s worker=%v instance=%v token=%q", j.State, j.Worker, j.WorkerInstance, j.ClaimToken)
		}
		if j.Epoch != nil || j.SPerEpoch != nil || j.PGID != nil {
			t.Errorf("claimed row carries stale progress: epoch=%v s=%v pgid=%v", j.Epoch, j.SPerEpoch, j.PGID)
		}
		if j.ClaimedAt == nil || j.StartedAt == nil {
			t.Errorf("claimed_at/started_at not stamped: %v %v", j.ClaimedAt, j.StartedAt)
		}
	}
	if _, ok := claim(t, st, jobs.LaneTrain); ok {
		t.Error("a sixth claim must find nothing")
	}
}

func TestClaimNextSkipsFlaggedAndForeignRows(t *testing.T) {
	st := openWithWorker(t)
	now := time.Now()
	stopped := queue(t, st, jobs.KindTrain, 100, "normal", now)
	cancelled := queue(t, st, jobs.KindTrain, 100, "normal", now.Add(time.Second))
	otherNam := queue(t, st, jobs.KindTrain, 100, "normal", now.Add(2*time.Second))
	probe := queue(t, st, jobs.KindProbeSelf, 1, "normal", now.Add(3*time.Second))
	plain := queue(t, st, jobs.KindTrain, 100, "normal", now.Add(4*time.Second))
	storetest.Exec(t, st, `UPDATE jobs SET stop_requested_at = now() WHERE id = $1`, stopped)
	storetest.Exec(t, st, `UPDATE jobs SET cancel_requested_at = now() WHERE id = $1`, cancelled)
	storetest.Exec(t, st, `UPDATE jobs SET required_nam_version = '9.9.9' WHERE id = $1`, otherNam)
	// A job whose take was wiped: the journal row survives with take_id NULL and is nobody's to run.
	orphan := queue(t, st, jobs.KindTrain, 100, "high", now) // 'high' would otherwise drain first
	storetest.Exec(t, st, `UPDATE jobs SET take_id = NULL WHERE id = $1`, orphan)

	if j := mustClaim(t, st, jobs.LaneTrain); j.ID != plain {
		t.Errorf("train lane claimed %d, want %d (stop/cancel/other-nam/probe rows skipped)", j.ID, plain)
	}
	if _, ok := claim(t, st, jobs.LaneTrain); ok {
		t.Error("flagged rows must never be claimed")
	}
	if j := mustClaim(t, st, jobs.LaneProbe); j.ID != probe || j.Lane != jobs.LaneProbe {
		t.Errorf("probe lane claimed %+v, want job %d", j.ID, probe)
	}
	// …and the orphan is still readable (take_id reads as 0), so the app can close it.
	j, ok, err := st.GetJob(context.Background(), orphan)
	if err != nil || !ok {
		t.Fatalf("orphaned job unreadable: ok=%v err=%v", ok, err)
	}
	{
		if j.TakeID != 0 {
			t.Errorf("orphaned job take_id = %d, want 0 (NULL)", j.TakeID)
		}
	}
}

func TestClaimNextConcurrentClaimersNeverShareARow(t *testing.T) {
	st := openWithWorker(t)
	const n = 12
	for i := 0; i < n; i++ {
		queue(t, st, jobs.KindTrain, 10, "normal", time.Now())
	}
	var (
		mu   sync.Mutex
		seen = map[int64]int{}
		wg   sync.WaitGroup
	)
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for {
				j, ok, err := st.ClaimNext(context.Background(), jobs.LaneTrain, workerA, "inst-a", storetest.NamVersion)
				if err != nil {
					t.Errorf("claimer %d: %v", w, err)
					return
				}
				if !ok {
					return
				}
				mu.Lock()
				seen[j.ID]++
				mu.Unlock()
			}
		}(w)
	}
	wg.Wait()
	if len(seen) != n {
		t.Fatalf("claimed %d distinct jobs, want %d", len(seen), n)
	}
	for id, c := range seen {
		if c != 1 {
			t.Errorf("job %d claimed %d times", id, c)
		}
	}
}

func TestWritesAreFencedOnClaimToken(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()
	id := queue(t, st, jobs.KindTrain, 100, "normal", time.Now())
	j := mustClaim(t, st, jobs.LaneTrain)
	stale := "00000000-0000-0000-0000-000000000000"

	if ok, err := st.SetJobPGID(ctx, id, j.ClaimToken, 4242); err != nil || !ok {
		t.Fatalf("SetJobPGID (ours): ok=%v err=%v", ok, err)
	}
	if ok, err := st.SetJobPGID(ctx, id, stale, 9999); err != nil || ok {
		t.Fatalf("SetJobPGID (stale token): ok=%v err=%v, want false/nil", ok, err)
	}
	if err := st.UpdateProgress(ctx, id, j.ClaimToken, 5, 1.5, nil); err != nil {
		t.Fatal(err)
	}
	if err := st.UpdateProgress(ctx, id, stale, 9, 9.9, nil); err != nil {
		t.Fatal(err)
	}
	if err := st.AppendLog(ctx, id, j.ClaimToken, "kept"); err != nil {
		t.Fatal(err)
	}
	if err := st.AppendLog(ctx, id, stale, "straggler"); err != nil {
		t.Fatal(err)
	}
	got := get(t, st, id)
	if got.PGID == nil || *got.PGID != 4242 || got.Epoch == nil || *got.Epoch != 5 || got.SPerEpoch == nil || *got.SPerEpoch != 1.5 {
		t.Errorf("row = pgid %v epoch %v s %v, want 4242/5/1.5 (stale writes fenced out)", got.PGID, got.Epoch, got.SPerEpoch)
	}
	lines, _ := st.JobLog(ctx, id)
	if len(lines) != 1 || lines[0] != "kept" {
		t.Errorf("log = %v, want only [kept]", lines)
	}
	// A non-positive s/epoch is stored NULL (the column CHECKs > 0), not rejected.
	if err := st.UpdateProgress(ctx, id, j.ClaimToken, 6, 0, nil); err != nil {
		t.Fatalf("UpdateProgress s=0: %v", err)
	}
	if got := get(t, st, id); got.SPerEpoch != nil || *got.Epoch != 6 {
		t.Errorf("after s=0: s_per_epoch=%v epoch=%v, want NULL/6", got.SPerEpoch, got.Epoch)
	}
}

func TestRequeueReleasesClaim(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()
	id := queue(t, st, jobs.KindTrain, 100, "normal", time.Now())
	j := mustClaim(t, st, jobs.LaneTrain)
	_, _ = st.SetJobPGID(ctx, id, j.ClaimToken, 77)
	_ = st.UpdateProgress(ctx, id, j.ClaimToken, 3, 2.0, nil)

	if err := st.RequeueJob(ctx, id, j.ClaimToken); err != nil {
		t.Fatal(err)
	}
	got := get(t, st, id)
	if got.State != jobs.StateQueued || got.Worker != nil || got.WorkerInstance != nil || got.ClaimToken != "" ||
		got.PGID != nil || got.ClaimedAt != nil || got.StartedAt != nil || got.Epoch != nil || got.SPerEpoch != nil {
		t.Errorf("requeued row not clean: %+v", got)
	}
	// It is claimable again, under a NEW token.
	j2 := mustClaim(t, st, jobs.LaneTrain)
	if j2.ID != id || j2.ClaimToken == j.ClaimToken {
		t.Errorf("reclaim = job %d token %q, want job %d with a fresh token", j2.ID, j2.ClaimToken, id)
	}
}

func TestFinishSucceededWritesRowAndResultTogether(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()
	id := queue(t, st, jobs.KindTrain, 250, "normal", time.Now())
	j := mustClaim(t, st, jobs.LaneTrain)
	storetest.Exec(t, st, `UPDATE jobs SET stop_requested_at = now() WHERE id = $1`, id) // a stop that raced the finish
	esr := 0.0123
	ok, err := st.FinishSucceeded(ctx, id, j.ClaimToken, store.Result{Reached: 250, ESR: &esr, Nam: []byte("nam-bytes")}, prov)
	if err != nil || !ok {
		t.Fatalf("FinishSucceeded: ok=%v err=%v", ok, err)
	}
	got := get(t, st, id)
	if got.State != jobs.StateSucceeded || got.FinishedAt == nil || got.PGID != nil {
		t.Errorf("row = state %s finished %v pgid %v", got.State, got.FinishedAt, got.PGID)
	}
	if got.Reached == nil || *got.Reached != 250 || got.ESR == nil || *got.ESR != esr {
		t.Errorf("reached=%v esr=%v, want 250/%v", got.Reached, got.ESR, esr)
	}
	if got.NamVersion == nil || *got.NamVersion != prov.NamVersion || got.DriverSHA256 == nil || *got.DriverSHA256 != prov.DriverSHA256 || got.SignalSHA256 == nil || *got.SignalSHA256 != prov.SignalSHA256 {
		t.Errorf("provenance not stamped: %v %v %v", got.NamVersion, got.DriverSHA256, got.SignalSHA256)
	}
	// job_result keeps the MODEL and the outcome; the checkpoint is not here any more — since 0007
	// the weights live once, on the take.
	nam, present := storetest.Result(t, st, id)
	if !present || string(nam) != "nam-bytes" {
		t.Errorf("job_result = %q (present=%v)", nam, present)
	}
	if n := storetest.Count(t, st, `SELECT count(*) FROM job_result WHERE job_id = $1 AND epochs = 250 AND claim_token = $2::uuid AND size = 9`, id, j.ClaimToken); n != 1 {
		t.Error("job_result epochs/claim_token/size not as written")
	}

	// A second finish (a straggler, or the same token after the row left running) writes nothing.
	if ok, err := st.FinishSucceeded(ctx, id, j.ClaimToken, store.Result{Reached: 1, Nam: []byte("x")}, prov); err != nil || ok {
		t.Errorf("second finish: ok=%v err=%v, want false/nil", ok, err)
	}

	// A second success on its own row.
	id2 := queue(t, st, jobs.KindTrain, 10, "normal", time.Now())
	j2 := mustClaim(t, st, jobs.LaneTrain)
	if ok, err := st.FinishSucceeded(ctx, id2, j2.ClaimToken, store.Result{Reached: 10, Nam: []byte("n")}, prov); err != nil || !ok {
		t.Fatalf("FinishSucceeded (no ckpt): ok=%v err=%v", ok, err)
	}
	if n := storetest.Count(t, st, `SELECT count(*) FROM job_result WHERE job_id = $1`, id2); n != 1 {
		t.Error("a success must store its outcome row")
	}
}

func TestFinishProbeFailedCancelled(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()

	pid := queue(t, st, jobs.KindProbeSelf, 1, "normal", time.Now())
	pj := mustClaim(t, st, jobs.LaneProbe)
	esr := 0.0012
	if ok, err := st.FinishProbeSelf(ctx, pid, pj.ClaimToken, jobs.VerdictPass, &esr, prov); err != nil || !ok {
		t.Fatalf("FinishProbeSelf: ok=%v err=%v", ok, err)
	}
	if got := get(t, st, pid); got.State != jobs.StateSucceeded || got.Verdict == nil || *got.Verdict != "pass" || got.Reached != nil {
		t.Errorf("probe row = %+v", got)
	}
	if _, present := storetest.Result(t, st, pid); present {
		t.Error("a probe must store no result row")
	}

	fid := queue(t, st, jobs.KindTrain, 100, "normal", time.Now())
	fj := mustClaim(t, st, jobs.LaneTrain)
	storetest.Exec(t, st, `UPDATE jobs SET stop_requested_at = now() WHERE id = $1`, fid)
	if ok, err := st.FinishFailed(ctx, fid, fj.ClaimToken, jobs.ErrStopFailed, "no usable pair", prov); err != nil || !ok {
		t.Fatalf("FinishFailed: ok=%v err=%v", ok, err)
	}
	// A code outside the job_error domain is refused by the database, not silently written.
	xid := queue(t, st, jobs.KindTrain, 100, "normal", time.Now())
	xj := mustClaim(t, st, jobs.LaneTrain)
	if _, err := st.FinishFailed(ctx, xid, xj.ClaimToken, "made_up_code", "", prov); err == nil {
		t.Error("an error code outside the job_error domain must be rejected")
	}

	cid := queue(t, st, jobs.KindTrain, 100, "normal", time.Now())
	cj := mustClaim(t, st, jobs.LaneTrain)
	if ok, err := st.FinishCancelled(ctx, cid, cj.ClaimToken, prov); err != nil || !ok {
		t.Fatalf("FinishCancelled: ok=%v err=%v", ok, err)
	}
	if got := get(t, st, cid); got.State != jobs.StateCancelled || got.ErrorCode == nil || *got.ErrorCode != jobs.ErrCancelled || got.FinishedAt == nil {
		t.Errorf("cancelled row = %+v", got)
	}
}

func TestESRSanitizedForTheCheck(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()
	id := queue(t, st, jobs.KindProbeSelf, 1, "normal", time.Now())
	j := mustClaim(t, st, jobs.LaneProbe)
	neg := -1.0 // a driver "esr=-1" must become NULL, not a rejected transaction
	if ok, err := st.FinishProbeSelf(ctx, id, j.ClaimToken, jobs.VerdictFail, &neg, prov); err != nil || !ok {
		t.Fatalf("FinishProbeSelf(esr=-1): ok=%v err=%v", ok, err)
	}
	if got := get(t, st, id); got.ESR != nil {
		t.Errorf("esr = %v, want NULL for a negative value", *got.ESR)
	}
}

// finishedTrain writes a terminal train-lane row for worker with the given speed
// numbers directly (the avg reads only terminal rows).
func finishedTrain(t *testing.T, st *store.Store, worker string, kind string, epochs int, startEpoch *int64, epoch int64, spe float64, finishedAt time.Time) {
	t.Helper()
	tk := storetest.SeedTake(t, st, []byte("w-"+time.Now().Format(time.RFC3339Nano)))
	id := storetest.InsertJob(t, st, storetest.JobSpec{Take: tk, Kind: kind, Epochs: epochs, StartEpoch: startEpoch})
	storetest.Exec(t, st,
		`UPDATE jobs SET state = 'succeeded', worker = $2, finished_at = $3, epoch = $4, s_per_epoch = $5 WHERE id = $1`,
		id, worker, finishedAt, epoch, spe)
}

func TestAvgSPerEpochWindowedPerWorker(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()
	if avg, err := st.AvgSPerEpoch(ctx, workerA); err != nil || avg != nil {
		t.Fatalf("empty: avg=%v err=%v, want nil/nil", avg, err)
	}
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	// One long train covers the whole 30-epoch window by itself.
	finishedTrain(t, st, workerA, jobs.KindTrain, 400, nil, 399, 5.0, base.Add(1*time.Hour))
	if avg, _ := st.AvgSPerEpoch(ctx, workerA); avg == nil || *avg != 5.0 {
		t.Fatalf("single job avg = %v, want 5.0", avg)
	}
	// A newer 10-epoch run @ 8.0: (10*8 + 20*5)/30 = 6.0 — the older run is clipped at the edge.
	finishedTrain(t, st, workerA, jobs.KindTrain, 10, nil, 9, 8.0, base.Add(2*time.Hour))
	if avg, _ := st.AvgSPerEpoch(ctx, workerA); avg == nil || *avg < 5.99 || *avg > 6.01 {
		t.Errorf("windowed avg = %v, want 6.0", avg)
	}
	// Another worker's history does not leak in; a probe never counts.
	finishedTrain(t, st, workerB, jobs.KindTrain, 100, nil, 99, 99.0, base.Add(3*time.Hour))
	if avg, _ := st.AvgSPerEpoch(ctx, workerA); avg == nil || *avg < 5.99 || *avg > 6.01 {
		t.Errorf("avg after another worker's run = %v, want unchanged 6.0", avg)
	}
	// Three newer 10-epoch runs @ 2.0 fill the window → exactly 2.0 (the window clips).
	for i := 0; i < 3; i++ {
		finishedTrain(t, st, workerA, jobs.KindTrain, 10, nil, 9, 2.0, base.Add(time.Duration(4+i)*time.Hour))
	}
	if avg, _ := st.AvgSPerEpoch(ctx, workerA); avg == nil || *avg < 1.99 || *avg > 2.01 {
		t.Errorf("avg after 3 newer runs = %v, want 2.0", avg)
	}
}

// TestMyTally is the menu's one line about the machine itself: what it has computed, ever —
// counted from the epoch rows as they land, not from what a finished job says it reached.
func TestMyTally(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()
	if got, err := st.MyTally(ctx, workerA); err != nil || got != (store.Tally{}) {
		t.Fatalf("a box that has computed nothing: %+v err=%v, want a zero tally", got, err)
	}
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	// A run of this box, STILL GOING: its epochs count as they land, which is the whole reason
	// the tally is read off job_epochs rather than off what a finished job says it reached.
	queue(t, st, jobs.KindTrain, 800, "normal", base)
	j := mustClaim(t, st, jobs.LaneTrain)
	// The first epoch of an attempt has nothing to measure its duration from: it counts among
	// the epochs and not in the hours.
	if err := st.RecordEpoch(ctx, j.ID, j.ClaimToken, 0, nil, 0); err != nil {
		t.Fatalf("RecordEpoch(first): %v", err)
	}
	for i, sec := range []float64{600, 1200, 1800} { // an hour in three epochs, 1200 s each on average
		if err := st.RecordEpoch(ctx, j.ID, j.ClaimToken, i+1, nil, sec); err != nil {
			t.Fatalf("RecordEpoch(%d): %v", i+1, err)
		}
	}
	want := store.Tally{TrainEpochs: 4, TotalEpochs: 4, Hours: 1, SPerEpoch: 1200.0}
	if got, err := st.MyTally(ctx, workerA); err != nil || got != want {
		t.Fatalf("mid-run tally = %+v err=%v, want %+v", got, err, want)
	}

	// A SELF-CHECK WRITES NO EPOCH ROWS — it is killed on its verdict — and still counts as the
	// one epoch of GPU it is.
	pid := queue(t, st, jobs.KindProbeSelf, 1, "normal", base.Add(time.Minute))
	pj := mustClaim(t, st, jobs.LaneProbe)
	if ok, err := st.FinishProbeSelf(ctx, pid, pj.ClaimToken, jobs.VerdictPass, nil, prov); err != nil || !ok {
		t.Fatalf("FinishProbeSelf: ok=%v err=%v", ok, err)
	}
	want.Probes, want.TotalEpochs = 1, 5
	if got, err := st.MyTally(ctx, workerA); err != nil || got != want {
		t.Fatalf("after a probe = %+v err=%v, want %+v", got, err, want)
	}

	// Another machine's epochs are another machine's line.
	queue(t, st, jobs.KindTrain, 400, "normal", base.Add(2*time.Minute))
	other, ok, err := st.ClaimNext(ctx, jobs.LaneTrain, workerB, "inst-b", storetest.NamVersion)
	if err != nil || !ok {
		t.Fatalf("ClaimNext for %s: ok=%v err=%v", workerB, ok, err)
	}
	if err := st.RecordEpoch(ctx, other.ID, other.ClaimToken, 0, nil, 7200); err != nil {
		t.Fatalf("RecordEpoch(other): %v", err)
	}
	if got, err := st.MyTally(ctx, workerA); err != nil || got != want {
		t.Errorf("after another worker's epoch = %+v err=%v, want unchanged %+v", got, err, want)
	}
	theirs := store.Tally{TrainEpochs: 1, TotalEpochs: 1, Hours: 2, SPerEpoch: 7200.0}
	if got, err := st.MyTally(ctx, workerB); err != nil || got != theirs {
		t.Errorf("the other box = %+v err=%v, want %+v", got, err, theirs)
	}
}

func TestAvgSPerEpochWeightsContinuationsAndClamps(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	start := int64(20)
	// A train_more resumed at 20 and ran to absolute 29: 10 computed epochs @ 10.0; an
	// older full train of 100 @ 2.0 → (10*10 + 20*2)/30 = 4.6667.
	finishedTrain(t, st, workerA, jobs.KindTrainMore, 400, &start, 29, 10.0, base.Add(2*time.Hour))
	finishedTrain(t, st, workerA, jobs.KindTrain, 100, nil, 99, 2.0, base.Add(1*time.Hour))
	avg, err := st.AvgSPerEpoch(ctx, workerA)
	if err != nil || avg == nil || *avg < 4.66 || *avg > 4.67 {
		t.Fatalf("avg = %v err=%v, want ~4.6667 (continuation weighted by 10)", avg, err)
	}

	st2 := openWithWorker(t)
	bad := int64(200) // epoch below start_epoch: GREATEST clamps the weight to 1
	finishedTrain(t, st2, workerA, jobs.KindTrainMore, 400, &bad, 5, 100.0, base.Add(2*time.Hour))
	finishedTrain(t, st2, workerA, jobs.KindTrain, 100, nil, 99, 2.0, base.Add(1*time.Hour))
	want := (1*100.0 + 29*2.0) / 30.0
	if avg, _ := st2.AvgSPerEpoch(ctx, workerA); avg == nil || *avg < want-0.01 || *avg > want+0.01 {
		t.Errorf("clamped avg = %v, want ~%.4f", avg, want)
	}
}

func TestCountsAndMyRuns(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()
	now := time.Date(2026, 8, 21, 12, 0, 0, 0, time.UTC)
	_ = queue(t, st, jobs.KindTrain, 100, "normal", now)
	t2 := queue(t, st, jobs.KindTrain, 200, "high", now.Add(3*time.Minute)) // high, later arrival
	_ = queue(t, st, jobs.KindTrain, 300, "normal", now.Add(2*time.Minute))
	_ = queue(t, st, jobs.KindTrainMore, 400, "normal", now.Add(4*time.Minute)) // start_epoch 200
	ps := queue(t, st, jobs.KindProbeSelf, 1, "normal", now.Add(time.Minute))
	stopped := queue(t, st, jobs.KindTrain, 50, "normal", now.Add(5*time.Minute))
	storetest.Exec(t, st, `UPDATE jobs SET stop_requested_at = now() WHERE id = $1`, stopped)

	j := mustClaim(t, st, jobs.LaneTrain) // t2 (high)
	if j.ID != t2 {
		t.Fatalf("first claim = %d, want %d", j.ID, t2)
	}
	_ = st.UpdateProgress(ctx, t2, j.ClaimToken, 29, 1.0, nil) // remaining 200-(29+1) = 170
	pj := mustClaim(t, st, jobs.LaneProbe)
	if pj.ID != ps {
		t.Fatalf("probe claim = %d, want %d", pj.ID, ps)
	}

	running, queued, err := st.CountByState(ctx, workerA)
	if err != nil || running != 2 || queued != 3 {
		t.Errorf("CountByState = %d/%d err=%v, want 2 running / 3 claimable queued (the stopped row excluded)", running, queued, err)
	}
	// WHAT THIS WORKER IS RUNNING, and nothing else: the training row it claimed and
	// the probe. The other three queued rows are nobody's until somebody claims them,
	// and another worker's runs are not this menu's business.
	mine, err := st.MyRuns(ctx, workerA)
	if err != nil || len(mine) != 2 {
		t.Fatalf("MyRuns = %d runs err=%v, want 2 (the claimed train + the claimed probe)", len(mine), err)
	}
	if mine[0].Kind != jobs.KindTrain || mine[0].Lane != jobs.LaneTrain || mine[0].Remaining != 170 {
		t.Errorf("mine[0] = %+v, want the train run with 200-(29+1) = 170 epochs to go", mine[0])
	}
	if mine[0].Epoch == nil || *mine[0].Epoch != 29 || mine[0].Label == "" {
		t.Errorf("mine[0] = %+v, want epoch 29 and a label to draw", mine[0])
	}
	if mine[1].Kind != jobs.KindProbeSelf || mine[1].Lane != jobs.LaneProbe {
		t.Errorf("mine[1] = %+v, want the running probe", mine[1])
	}
	if other, err := st.MyRuns(ctx, "somebody-else"); err != nil || len(other) != 0 {
		t.Errorf("MyRuns for a worker running nothing = %v err=%v, want empty", other, err)
	}
}

func TestHeartbeatUpsertCapWantedAndContract(t *testing.T) {
	st := storetest.Open(t)
	ctx := context.Background()
	v, err := st.QueueContract(ctx)
	if err != nil || v != store.SupportedQueueContract {
		t.Fatalf("QueueContract = %d err=%v, want %d", v, err, store.SupportedQueueContract)
	}
	avg := 4.2
	disk := int64(123456789)
	info := store.WorkerInfo{
		Name: "studio.local", Instance: "inst-1", Version: "0.3.0", NamVersion: "0.13.0",
		DriverSHA256: prov.DriverSHA256, SignalSHA256: prov.SignalSHA256, GPU: "mps", Python: "3.12.13",
		SchemaVersion: store.SupportedQueueContract, TrainCap: 2, ProbeCap: 1, Running: 1, Paused: false, Ready: true,
		AvgSPerEpoch: &avg, DiskFreeBytes: &disk,
	}
	wanted, _, err := st.Heartbeat(ctx, info)
	if err != nil || wanted != nil {
		t.Fatalf("first Heartbeat: wanted=%v err=%v", wanted, err)
	}
	// The app asks for a wider lane.
	storetest.Exec(t, st, `UPDATE workers SET train_cap_wanted = 4 WHERE name = 'studio.local'`)
	info.Instance = "inst-2" // a restart: the instance changes, the row is the same
	info.Ready = false
	info.Note = "provisioning runtime"
	wanted, _, err = st.Heartbeat(ctx, info)
	if err != nil || wanted == nil || *wanted != 4 {
		t.Fatalf("second Heartbeat: wanted=%v err=%v, want 4", wanted, err)
	}
	var (
		instance, note string
		ready          bool
		cap            int
		n              int
	)
	if err := st.Pool().QueryRow(ctx, `SELECT instance, note, ready, train_cap, (SELECT count(*) FROM workers) FROM workers WHERE name = 'studio.local'`).Scan(&instance, &note, &ready, &cap, &n); err != nil {
		t.Fatal(err)
	}
	if instance != "inst-2" || note != "provisioning runtime" || ready || cap != 2 || n != 1 {
		t.Errorf("workers row = instance %s note %q ready %v cap %d rows %d", instance, note, ready, cap, n)
	}
	// Applied → consumed (guarded by the value).
	if err := st.ConsumeCapWanted(ctx, "studio.local", 4); err != nil {
		t.Fatal(err)
	}
	if n := storetest.Count(t, st, `SELECT count(*) FROM workers WHERE name = 'studio.local' AND train_cap_wanted IS NULL`); n != 1 {
		t.Error("train_cap_wanted not consumed")
	}
	storetest.Exec(t, st, `UPDATE workers SET train_cap_wanted = 3 WHERE name = 'studio.local'`)
	if err := st.ConsumeCapWanted(ctx, "studio.local", 4); err != nil {
		t.Fatal(err)
	}
	if n := storetest.Count(t, st, `SELECT count(*) FROM workers WHERE name = 'studio.local' AND train_cap_wanted = 3`); n != 1 {
		t.Error("a NEWER ask must survive a consume of the old value")
	}
	// An empty note / nam_version is NULL, never ''.
	info.Note, info.NamVersion = "", ""
	if _, _, err := st.Heartbeat(ctx, info); err != nil {
		t.Fatal(err)
	}
	if n := storetest.Count(t, st, `SELECT count(*) FROM workers WHERE name = 'studio.local' AND note IS NULL AND nam_version IS NULL`); n != 1 {
		t.Error("empty optional texts must be written NULL")
	}
}

func TestLibraryReads(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()
	wav := []byte("some wav bytes")
	tk := storetest.SeedTake(t, st, wav)
	got, sha, ok, err := st.TakeAudio(ctx, tk.ID)
	if err != nil || !ok || string(got) != string(wav) || sha != tk.WavSHA {
		t.Errorf("TakeAudio = %q sha %s ok %v err %v", got, sha, ok, err)
	}
	if _, _, ok, err := st.TakeAudio(ctx, 999999); err != nil || ok {
		t.Errorf("TakeAudio(unknown) = ok %v err %v", ok, err)
	}
	if sig, ok, err := st.TakeSignalSHA(ctx, tk.ID); err != nil || !ok || sig != storetest.SignalSHA {
		t.Errorf("TakeSignalSHA = %s ok %v err %v", sig, ok, err)
	}
	parent := storetest.InsertJob(t, st, storetest.JobSpec{Take: tk, Epochs: 50})
	pj := mustClaim(t, st, jobs.LaneTrain)
	if pj.ID != parent {
		t.Fatal("claimed the wrong job")
	}
	if ok, err := st.FinishSucceeded(ctx, parent, pj.ClaimToken, store.Result{Reached: 50, Nam: []byte("nam")}, prov); err != nil || !ok {
		t.Fatal(err)
	}
	start := int64(50)
	child := storetest.InsertJob(t, st, storetest.JobSpec{Take: tk, Kind: jobs.KindTrainMore, Epochs: 100, BaseJobID: &parent, StartEpoch: &start})
	// The weights the parent trained are on the TAKE — that is what the child continues from, with
	// nobody copying anything anywhere.
	storetest.SeedTakeCheckpoint(t, st, tk.ID, 50, []byte("nam"), []byte("parent-ckpt"))
	if c, ok, err := st.Checkpoint(ctx, tk.ID); err != nil || !ok || string(c.Ckpt) != "parent-ckpt" {
		t.Errorf("Checkpoint(take) = %q ok %v err %v", c.Ckpt, ok, err)
	}
	// The request block is frozen after claim (the schema's trigger).
	cj := mustClaim(t, st, jobs.LaneTrain)
	if cj.ID != child || cj.StartEpoch == nil || *cj.StartEpoch != 50 {
		t.Fatalf("child claim = %+v", cj)
	}
	if _, err := st.Pool().Exec(ctx, `UPDATE jobs SET epochs = 999 WHERE id = $1`, child); err == nil {
		t.Error("changing epochs after claim must be refused by jobs_freeze")
	}
}

// A cancel that lands after the last control poll must still win: the run the user
// threw away may not come back as an installed model.
func TestCancelWinsTheTerminalTransaction(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()
	id := queue(t, st, jobs.KindTrain, 100, "normal", time.Now())
	j := mustClaim(t, st, jobs.LaneTrain)
	storetest.Exec(t, st, `UPDATE jobs SET cancel_requested_at = now() WHERE id = $1`, id)
	esr := 0.004
	ok, err := st.FinishSucceeded(ctx, id, j.ClaimToken,
		store.Result{Reached: 100, ESR: &esr, Nam: []byte("nam-bytes")}, prov)
	if err != nil || !ok {
		t.Fatalf("FinishSucceeded: ok=%v err=%v", ok, err)
	}
	got := get(t, st, id)
	if got.State != jobs.StateCancelled {
		t.Errorf("state = %s, want cancelled (the user's cancel outranks the trainer's finish)", got.State)
	}
	if got.ErrorCode == nil || *got.ErrorCode != "cancelled" {
		t.Errorf("error_code = %v, want cancelled", got.ErrorCode)
	}
	if n := storetest.Count(t, st, `SELECT count(*) FROM job_result WHERE job_id = $1`, id); n != 0 {
		t.Error("a cancelled run keeps NOTHING: no weights, no checkpoint")
	}

	// A probe finishing into a cancel is the same story.
	pid := queue(t, st, jobs.KindProbeSelf, 1, "normal", time.Now())
	pj := mustClaim(t, st, jobs.LaneProbe)
	storetest.Exec(t, st, `UPDATE jobs SET cancel_requested_at = now() WHERE id = $1`, pid)
	if ok, err := st.FinishProbeSelf(ctx, pid, pj.ClaimToken, "pass", &esr, prov); err != nil || !ok {
		t.Fatalf("FinishProbeSelf: ok=%v err=%v", ok, err)
	}
	if p := get(t, st, pid); p.State != jobs.StateCancelled {
		t.Errorf("probe state = %s, want cancelled", p.State)
	}
}

// SOMEBODY ASKS THIS DAEMON TO STOP, and says HOW: the queue lives in the database, so a flag kept
// inside one app would stop nobody — and the two answers are not interchangeable. "after" lets an
// eight-hundred-epoch run finish; "now" stops it this second, keeping everything up to the last
// completed epoch. Both are BOUNDED, which is what a person waiting for their own GPU needs.
// See migration 0005.
func TestPauseWantedIsAskedAndConsumed(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()
	info := store.WorkerInfo{
		Name: "studio.local", Instance: "i1", Version: "0.3.0", SchemaVersion: store.SupportedQueueContract,
		TrainCap: 2, ProbeCap: 1, Ready: true,
	}
	if _, pause, err := st.Heartbeat(ctx, info); err != nil || pause != nil {
		t.Fatalf("no ask yet: pause=%v err=%v", pause, err)
	}
	for _, manner := range []string{store.PauseAfter, store.PauseNow, store.Resume} {
		storetest.Exec(t, st, `UPDATE workers SET pause_wanted = $1 WHERE name = 'studio.local'`, manner)
		_, pause, err := st.Heartbeat(ctx, info)
		if err != nil || pause == nil || *pause != manner {
			t.Fatalf("ask %q not returned: %v (%v)", manner, pause, err)
		}
		if err := st.ConsumePauseWanted(ctx, "studio.local", manner); err != nil {
			t.Fatal(err)
		}
		if n := storetest.Count(t, st, `SELECT count(*) FROM workers WHERE name = 'studio.local' AND pause_wanted IS NULL`); n != 1 {
			t.Errorf("%q not consumed", manner)
		}
	}
	// An ask that CHANGED between the read and the consume must survive: the person changed their
	// mind while the daemon was mid-beat, and the newer word is the one that counts.
	storetest.Exec(t, st, `UPDATE workers SET pause_wanted = $1 WHERE name = 'studio.local'`, store.PauseNow)
	if err := st.ConsumePauseWanted(ctx, "studio.local", store.PauseAfter); err != nil {
		t.Fatal(err)
	}
	if n := storetest.Count(t, st, `SELECT count(*) FROM workers WHERE name = 'studio.local' AND pause_wanted = 'now'`); n != 1 {
		t.Error("a newer ask must survive a consume of the old one")
	}
	// Nothing else is a pause. The column says so, so a typo cannot quietly become a third meaning.
	if _, err := st.Pool().Exec(ctx, `UPDATE workers SET pause_wanted = 'halt' WHERE name = 'studio.local'`); err == nil {
		t.Error("pause_wanted accepted a word that is not one of the three")
	}
}

// One daemon per worker name — the second process does not start. The lock is
// database-wide (not schema-scoped), so the name is made unique per test run.
func TestClaimIdentityIsExclusive(t *testing.T) {
	st := openWithWorker(t)
	ctx := context.Background()
	name := fmt.Sprintf("studio-%d.local", os.Getpid())
	release, mine, err := st.ClaimIdentity(ctx, name)
	if err != nil || !mine {
		t.Fatalf("first claim: mine=%v err=%v", mine, err)
	}
	// A second ClaimIdentity takes a DIFFERENT pooled connection, so it is a different
	// session — exactly what a second daemon process would be.
	if _, mine2, err := st.ClaimIdentity(ctx, name); err != nil || mine2 {
		t.Errorf("second daemon claimed the same name: mine=%v err=%v", mine2, err)
	}
	if rel3, mine3, err := st.ClaimIdentity(ctx, name+"-other"); err != nil || !mine3 {
		t.Errorf("another worker name must be free: mine=%v err=%v", mine3, err)
	} else {
		rel3()
	}
	release()
	if rel4, mine4, err := st.ClaimIdentity(ctx, name); err != nil || !mine4 {
		t.Errorf("after release the name is free again: mine=%v err=%v", mine4, err)
	} else {
		rel4()
	}
}

// WHY THE DAEMON WAITS FOR ITS FIRST HEARTBEAT INSTEAD OF DYING. A library the app has not migrated
// to this contract has no pause_wanted column, and the heartbeat's RETURNING needs it. That is not a
// broken database — it is a workshop where the daemon was installed before the app was run once, and
// the cure is to wait for the app. This pins the failure the wait is built around; without it, the
// process exits and launchd respawns it every ten seconds for ever, writing nothing anybody reads.
func TestHeartbeatFailsOnALibraryFromBeforeThisContract(t *testing.T) {
	st := storetest.Open(t)
	ctx := context.Background()
	if _, err := st.Pool().Exec(ctx, `ALTER TABLE workers DROP COLUMN pause_wanted`); err != nil {
		t.Fatalf("undo the column: %v", err)
	}
	_, _, err := st.Heartbeat(ctx, store.WorkerInfo{Name: "old-library", TrainCap: 1, ProbeCap: 1, Ready: true})
	if err == nil {
		t.Fatal("want the heartbeat to fail against a pre-contract-2 library")
	}
	if !strings.Contains(err.Error(), "pause_wanted") {
		t.Fatalf("want the missing column named, got %v", err)
	}
}

// …and the same library answers the contract question truthfully, which is what the app's lamp reads.
func TestQueueContractReadsWhatTheLibrarySays(t *testing.T) {
	st := storetest.Open(t)
	ctx := context.Background()
	if _, err := st.Pool().Exec(ctx, `UPDATE library SET queue_contract = $1 WHERE id = 1`,
		store.SupportedQueueContract-1); err != nil {
		t.Fatalf("age the library: %v", err)
	}
	got, err := st.QueueContract(ctx)
	if err != nil {
		t.Fatalf("read contract: %v", err)
	}
	if got != store.SupportedQueueContract-1 {
		t.Fatalf("want %d, got %d", store.SupportedQueueContract-1, got)
	}
}

// A PAUSED ROW CARRYING AN ASK MUST STILL BE CLAIMABLE — it is the only way it can ever be closed.
//
// Cron pauses a run whose task went silent; a hand then presses Stop or Cancel on it. Nobody holds
// that row: the app closes only QUEUED rows, cron closes only rows whose take is gone, and the claim
// used to refuse anything carrying either ask. The row stayed 'running' for ever, and jobs_one_live
// kept its take out of the queue with it — a state only psql could undo. Whoever claims it is now the
// hand that carries the ask out (runJob answers it without spawning a trainer).
func TestAPausedRunCarryingAnAskIsStillClaimable(t *testing.T) {
	st := openWithWorker(t)
	now := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)

	stopped := queue(t, st, jobs.KindTrain, 400, "normal", now)
	storetest.Exec(t, st, `UPDATE jobs SET state = 'running', worker = $2, claim_token = gen_random_uuid(),
	                              claimed_at = now(), started_at = now(), paused_at = now(),
	                              stop_requested_at = now() WHERE id = $1`, stopped, workerA)

	j := mustClaim(t, st, jobs.LaneTrain)
	if j.ID != stopped {
		t.Fatalf("claimed %d, want the paused row %d that was asked to stop", j.ID, stopped)
	}
	if j.StopRequestedAt == nil {
		t.Errorf("the ask must ride along on the claimed row — it is what the claimer acts on")
	}
	if j.PausedAt != nil {
		t.Errorf("taking a paused row clears the mark")
	}

	// A QUEUED row carrying the same ask is a different case and stays unclaimable: it never ran, so
	// there is nothing to keep and nobody has to be sent to close it — the app closes it where it is.
	cancelled := queue(t, st, jobs.KindTrain, 400, "high", now)
	storetest.Exec(t, st, `UPDATE jobs SET cancel_requested_at = now() WHERE id = $1`, cancelled)
	if _, ok := claim(t, st, jobs.LaneTrain); ok {
		t.Errorf("a queued row asked to be cancelled was handed out to a trainer")
	}
}

// WHAT THE PLAYER GETS TO PLAY. The take keeps two pairs: the last epoch, which is where a
// continuation picks up, and the best by validation ESR, which is the model a person listens to. On
// a run whose ESR swings an order of magnitude between neighbouring epochs the last one is close to
// a coin toss, and that is the whole reason the second pair exists.
//
// It was being erased by everyone who did not bring one. A finished run writes its exported pair
// with no best beside it, so best_* went NULL and the app — which reads COALESCE(best, last) — served
// the last epoch AS the best. Measured on a live library: two takes that reached their full 400
// epochs had no best pair at all, while a take stopped short kept one ten times better than its own
// last epoch.
func TestTheBestPairSurvivesWritersWhoDoNotBringOne(t *testing.T) {
	st := storetest.Open(t)
	ctx := context.Background()
	take := storetest.SeedTake(t, st, []byte("WAVE"))
	id := storetest.InsertJob(t, st, storetest.JobSpec{Take: take, Kind: jobs.KindTrain, Epochs: 400})
	token := "22222222-2222-2222-2222-222222222222"
	storetest.Exec(t, st, `UPDATE jobs SET state = 'running', claim_token = $2::uuid,
	                              claimed_at = now(), started_at = now() WHERE id = $1`, id, token)

	esr := func(v float64) *float64 { return &v }
	pair := func(n int, e *float64) store.Pair {
		return store.Pair{Reached: n, ESR: e, Nam: []byte("nam"), Ckpt: []byte("ckpt"),
			NamSHA: strings.Repeat("b", 64)}
	}
	bestOf := func() (reached *int, e *float64) {
		if err := st.Pool().QueryRow(ctx,
			`SELECT best_reached, best_esr FROM take_checkpoint WHERE take_id = $1`, take.ID).
			Scan(&reached, &e); err != nil {
			t.Fatalf("read the best pair: %v", err)
		}
		return
	}

	best := pair(7, esr(0.004))
	if v, err := st.PutCheckpoint(ctx, id, token, take.ID, pair(9, esr(0.05)), &best); err != nil || !v.Mine {
		t.Fatalf("seed: %+v err=%v", v, err)
	}
	if r, e := bestOf(); r == nil || *r != 7 || e == nil || *e > 0.0041 {
		t.Fatalf("best pair after the seed = %v/%v, want epoch 7 at 0.004", r, e)
	}

	// An epoch whose own best did not qualify (caught mid-rotation), and then the finish, which
	// writes the exported pair and offers no best at all. Neither may take epoch 7 away.
	if _, err := st.PutCheckpoint(ctx, id, token, take.ID, pair(10, esr(0.06)), nil); err != nil {
		t.Fatalf("silent writer: %v", err)
	}
	if _, err := st.PutCheckpoint(ctx, id, token, take.ID, pair(400, esr(0.02)), nil); err != nil {
		t.Fatalf("finish: %v", err)
	}
	if r, e := bestOf(); r == nil || *r != 7 || e == nil || *e > 0.0041 {
		t.Fatalf("the finish erased the best pair: %v/%v, want epoch 7 at 0.004", r, e)
	}

	// A worse one does not displace it either — "best" is a comparison, not an arrival order. This is
	// also the handover case: the machine that takes the row over has only its own scratch to scan.
	worse := pair(120, esr(0.03))
	if _, err := st.PutCheckpoint(ctx, id, token, take.ID, pair(120, esr(0.03)), &worse); err != nil {
		t.Fatalf("worse best: %v", err)
	}
	if r, _ := bestOf(); r == nil || *r != 7 {
		t.Fatalf("a worse pair displaced the best: reached = %v, want 7", r)
	}

	better := pair(200, esr(0.0008))
	if _, err := st.PutCheckpoint(ctx, id, token, take.ID, pair(200, esr(0.01)), &better); err != nil {
		t.Fatalf("better best: %v", err)
	}
	if r, e := bestOf(); r == nil || *r != 200 || e == nil || *e > 0.0009 {
		t.Fatalf("a better pair did not win: %v/%v, want epoch 200 at 0.0008", r, e)
	}
}

// THE RACE A CANCEL RUNS EVERY TIME.
//
// A cancel discards the take's weights — the only thing that does. The run's last checkpoint write
// starts at the same moment. ON CONFLICT DO UPDATE, on finding its conflicting row deleted by a
// transaction that committed while it waited, RETRIES THE INSERT — and re-uses the fence it read
// from a snapshot in which the job was still running and unpaused. So without a lock the weights
// come back AFTER the cancel threw them away, on a take whose run is over, and nothing fires again
// to take them off: the trigger only acts on a change of state and there will not be another one.
func TestALateWriteCannotResurrectWeightsACancelDiscarded(t *testing.T) {
	st := storetest.Open(t)
	ctx := context.Background()
	take := storetest.SeedTake(t, st, []byte("WAVE"))
	id := storetest.InsertJob(t, st, storetest.JobSpec{Take: take, Kind: jobs.KindTrain, Epochs: 400})

	token := "11111111-1111-1111-1111-111111111111"
	storetest.Exec(t, st, `UPDATE jobs SET state = 'running', claim_token = $2::uuid,
	                              claimed_at = now(), started_at = now() WHERE id = $1`, id, token)
	pair := func(n int) store.Pair {
		nam := []byte("nam")
		return store.Pair{Reached: n, Nam: nam, Ckpt: []byte("ckpt"), NamSHA: strings.Repeat("a", 64)}
	}
	if v, err := st.PutCheckpoint(ctx, id, token, take.ID, pair(5), nil); err != nil || !v.Mine {
		t.Fatalf("seed the checkpoint: %+v err=%v", v, err)
	}

	// The cancel takes its locks and HOLDS them: the job closed, the weights moved aside, nothing
	// committed yet. This is the window.
	tx, err := st.Pool().Begin(ctx)
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	if _, err := tx.Exec(ctx, `UPDATE jobs SET state = 'cancelled', finished_at = now() WHERE id = $1`, id); err != nil {
		t.Fatalf("cancel: %v", err)
	}

	put := make(chan error, 1)
	go func() {
		_, e := st.PutCheckpoint(ctx, id, token, take.ID, pair(6), nil)
		put <- e
	}()
	time.Sleep(300 * time.Millisecond) // let it reach the lock it must wait on
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("commit: %v", err)
	}
	if err := <-put; err != nil {
		t.Fatalf("the late write must be a no-op, not an error: %v", err)
	}

	if n := storetest.Count(t, st, `SELECT count(*) FROM take_checkpoint WHERE take_id = $1`, take.ID); n != 0 {
		t.Fatalf("the take carries %d rows of weights a cancel discarded", n)
	}
	// …and they are still readable, bytes and all, which is what the discard table is for.
	if n := storetest.Count(t, st,
		`SELECT count(*) FROM take_checkpoint_discarded WHERE take_id = $1 AND reason = 'cancelled'`,
		take.ID); n != 1 {
		t.Fatalf("the discarded weights were not kept (rows = %d)", n)
	}
}

// covered read three command flags every two seconds beside the checkpoint write; a run learns
// everything it needs from that write now, and a listener reads the take's row.)

// (Two RecoverRunning tests lived here and are gone with it. Recovery does not touch the queue any
// more: putting a row back is a decision about the QUEUE, and a trainer does not have that role. The
// library takes it — a run whose task has gone silent is marked claimable by cron, keeping everything
// it had — and what is left of recovery is killing the children a dead process left holding a GPU,
// which only the machine they run on can do.)
