// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"orbit-capture-nam-trainer/internal/applog"
	"orbit-capture-nam-trainer/internal/config"
	"orbit-capture-nam-trainer/internal/jobkey"
	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/testsupport"
)

// The fixture profile the test server advertises; the key formula uses these.
const (
	tpNam    = "0.13.0"
	tpDriver = "drvsha256"
	tpSignal = "sigsha256"
)

func newJobsServer(t *testing.T) (*Server, string, *store.Store) {
	t.Helper()
	base := t.TempDir()
	cfg, err := config.Load(base)
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	st, err := store.Open(context.Background(), cfg.DBPath())
	if err != nil {
		t.Fatalf("store.Open: %v", err)
	}
	t.Cleanup(func() { st.Close() })
	lg, err := applog.Open(filepath.Join(base, "logs", "trainer.log"))
	if err != nil {
		t.Fatalf("applog.Open: %v", err)
	}
	t.Cleanup(func() { lg.Close() })
	srv := New(cfg, st, lg)
	srv.SetProfile(Profile{Ready: true, Nam: tpNam, DriverSHA256: tpDriver, SignalSHA256: tpSignal})
	return srv, cfg.Token, st
}

func keyFor(wav []byte, kind string, epochs int, arch string) string {
	return jobkey.Compute(jobkey.SHA256Hex(wav), kind, jobs.NormalizeEpochs(kind, epochs), arch,
		tpNam, tpDriver, tpSignal)
}

func do(t *testing.T, h http.Handler, method, target, token string, body []byte) *httptest.ResponseRecorder {
	t.Helper()
	var req *http.Request
	if body != nil {
		req = httptest.NewRequest(method, target, bytes.NewReader(body))
	} else {
		req = httptest.NewRequest(method, target, nil)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestPutNewThenIdempotent(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	key := keyFor(wav, jobs.KindTrain, 100, "standard")
	url := "/v1/jobs/" + key + "?kind=train&epochs=100&arch=standard&priority=1"

	rec := do(t, h, http.MethodPut, url, token, wav)
	if rec.Code != http.StatusCreated {
		t.Fatalf("first PUT = %d, want 201 (body %s)", rec.Code, rec.Body.String())
	}
	var pr putResponse
	mustJSON(t, rec, &pr)
	if pr.Key != key || pr.State != jobs.StateQueued || pr.Position == nil || *pr.Position != 1 {
		t.Errorf("PUT response = %+v", pr)
	}

	// Idempotent resubmit → 200, same state.
	rec = do(t, h, http.MethodPut, url, token, wav)
	if rec.Code != http.StatusOK {
		t.Fatalf("resubmit = %d, want 200", rec.Code)
	}

	// GET reflects it.
	rec = do(t, h, http.MethodGet, "/v1/jobs/"+key, token, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET = %d, want 200", rec.Code)
	}
	var jr jobResponse
	mustJSON(t, rec, &jr)
	if jr.State != jobs.StateQueued || jr.Kind != jobs.KindTrain || jr.Epochs != 100 {
		t.Errorf("GET job = %+v", jr)
	}
	if jr.Position == nil || *jr.Position != 1 {
		t.Errorf("position = %v, want 1", jr.Position)
	}
}

func TestPutProbeIgnoresEpochs(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	// Key uses the normalized epochs (1), NOT any epochs param.
	key := keyFor(wav, jobs.KindProbeSelf, 0, "standard")
	url := "/v1/jobs/" + key + "?kind=probe_self&epochs=555"

	rec := do(t, h, http.MethodPut, url, token, wav)
	if rec.Code != http.StatusCreated {
		t.Fatalf("PUT probe = %d, want 201 (body %s)", rec.Code, rec.Body.String())
	}
	rec = do(t, h, http.MethodGet, "/v1/jobs/"+key, token, nil)
	var jr jobResponse
	mustJSON(t, rec, &jr)
	if jr.Epochs != jobs.ProbeSelfEpochs {
		t.Errorf("probe epochs = %d, want %d", jr.Epochs, jobs.ProbeSelfEpochs)
	}
}

func TestPutKeyMismatch(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	url := "/v1/jobs/deadbeefdeadbeef?kind=train&epochs=100"

	rec := do(t, h, http.MethodPut, url, token, wav)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("PUT = %d, want 400", rec.Code)
	}
	if code := errCode(t, rec); code != codeKeyMismatch {
		t.Errorf("error code = %q, want %q", code, codeKeyMismatch)
	}
}

func TestPutWavInvalid(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	h := srv.Handler()
	bad := testsupport.WAV(44100, 45) // wrong sample rate
	// wav validation runs before the key recompute, so any key reaches it.
	url := "/v1/jobs/whatever?kind=train&epochs=100"

	rec := do(t, h, http.MethodPut, url, token, bad)
	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("PUT = %d, want 422 (body %s)", rec.Code, rec.Body.String())
	}
	if code := errCode(t, rec); code != codeWavInvalid {
		t.Errorf("error code = %q, want %q", code, codeWavInvalid)
	}
}

func TestPutBadParams(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	cases := []struct{ name, url string }{
		{"missing kind", "/v1/jobs/k?epochs=100"},
		{"unknown kind", "/v1/jobs/k?kind=bogus&epochs=100"},
		{"train missing epochs", "/v1/jobs/k?kind=train"},
		{"bad priority", "/v1/jobs/k?kind=train&epochs=100&priority=9"},
		{"bad epochs", "/v1/jobs/k?kind=train&epochs=0"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := do(t, h, http.MethodPut, tc.url, token, wav)
			if rec.Code != http.StatusBadRequest {
				t.Errorf("%s: code = %d, want 400 (body %s)", tc.name, rec.Code, rec.Body.String())
			}
		})
	}
}

func TestPutRuntimeUnavailable(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	srv.SetProfile(Profile{}) // no resolved runtime
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	key := keyFor(wav, jobs.KindTrain, 100, "standard")
	url := "/v1/jobs/" + key + "?kind=train&epochs=100"

	rec := do(t, h, http.MethodPut, url, token, wav)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("PUT = %d, want 503", rec.Code)
	}
	if code := errCode(t, rec); code != codeRuntimeUnavailable {
		t.Errorf("error code = %q, want %q", code, codeRuntimeUnavailable)
	}
}

func TestGetUnknownJob(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	rec := do(t, srv.Handler(), http.MethodGet, "/v1/jobs/nope", token, nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("GET = %d, want 404", rec.Code)
	}
}

func TestDeleteThenResubmit(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	key := keyFor(wav, jobs.KindTrain, 100, "standard")
	url := "/v1/jobs/" + key + "?kind=train&epochs=100"

	if rec := do(t, h, http.MethodPut, url, token, wav); rec.Code != http.StatusCreated {
		t.Fatalf("PUT = %d, want 201", rec.Code)
	}
	if rec := do(t, h, http.MethodDelete, "/v1/jobs/"+key, token, nil); rec.Code != http.StatusNoContent {
		t.Fatalf("DELETE = %d, want 204", rec.Code)
	}
	if rec := do(t, h, http.MethodGet, "/v1/jobs/"+key, token, nil); rec.Code != http.StatusNotFound {
		t.Fatalf("GET after delete = %d, want 404", rec.Code)
	}
	// Delete freed the key: resubmit trains again (fresh 201).
	if rec := do(t, h, http.MethodPut, url, token, wav); rec.Code != http.StatusCreated {
		t.Fatalf("resubmit after delete = %d, want 201", rec.Code)
	}
}

func TestDeleteUnknown(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	rec := do(t, srv.Handler(), http.MethodDelete, "/v1/jobs/nope", token, nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("DELETE unknown = %d, want 404", rec.Code)
	}
}

func TestJobBodyEtaAndError(t *testing.T) {
	// Running with progress → eta_s = ceil((epochs-(epoch+1)) * s_per_epoch).
	epoch := int64(29)
	spe := 2.0
	running := jobBody(jobs.Job{
		Kind: jobs.KindTrain, State: jobs.StateRunning, Epochs: 100, Epoch: &epoch, SPerEpoch: &spe,
	})
	if running.EtaS == nil || *running.EtaS != 140 { // (100-30)*2 = 140
		t.Errorf("eta_s = %v, want 140", running.EtaS)
	}

	// Failed → error{code,message}; not populated for other states.
	code, msg := "train_failed", "boom"
	failed := jobBody(jobs.Job{
		Kind: jobs.KindTrain, State: jobs.StateFailed, ErrorCode: &code, ErrorMsg: &msg,
	})
	if failed.Error == nil || failed.Error.Code != "train_failed" || failed.Error.Message != "boom" {
		t.Errorf("error = %+v, want {train_failed, boom}", failed.Error)
	}
	if failed.EtaS != nil {
		t.Errorf("failed job eta_s = %v, want nil", failed.EtaS)
	}
	if running.Error != nil {
		t.Errorf("running job error = %+v, want nil", running.Error)
	}
}

func TestDeleteFiresNotifier(t *testing.T) {
	srv, token, st := newJobsServer(t)
	var notified int
	srv.SetNotifier(func() { notified++ })
	h := srv.Handler()

	// A still-queued job: no worker ever touches it, so the DELETE handler itself
	// must fire the notifier to republish counts and release the keep-awake hold.
	must(t, st.InsertJob(context.Background(), jobs.Job{Key: "q", Kind: jobs.KindTrain, State: jobs.StateQueued, Priority: 1, Epochs: 100, Arch: "standard", CreatedAt: 1}, []byte("q")))
	if rec := do(t, h, http.MethodDelete, "/v1/jobs/q", token, nil); rec.Code != http.StatusNoContent {
		t.Fatalf("DELETE = %d, want 204", rec.Code)
	}
	if notified != 1 {
		t.Errorf("notifier fired %d times, want 1", notified)
	}
}

func TestPatchReorder(t *testing.T) {
	srv, token, st := newJobsServer(t)
	ctx := context.Background()
	h := srv.Handler()

	// Seed two queued jobs: a (earlier) then b (later), same priority.
	must(t, st.InsertJob(ctx, jobs.Job{Key: "a", Kind: jobs.KindTrain, State: jobs.StateQueued, Priority: 1, Epochs: 100, Arch: "standard", CreatedAt: 100}, []byte("a")))
	must(t, st.InsertJob(ctx, jobs.Job{Key: "b", Kind: jobs.KindTrain, State: jobs.StateQueued, Priority: 1, Epochs: 100, Arch: "standard", CreatedAt: 200}, []byte("b")))

	// b starts at position 2.
	if pos := getPosition(t, h, token, "b"); pos != 2 {
		t.Fatalf("b position = %d, want 2", pos)
	}
	// Promote b to high priority.
	if rec := do(t, h, http.MethodPatch, "/v1/jobs/b?priority=0", token, nil); rec.Code != http.StatusNoContent {
		t.Fatalf("PATCH = %d, want 204", rec.Code)
	}
	if pos := getPosition(t, h, token, "b"); pos != 1 {
		t.Errorf("b position after promote = %d, want 1", pos)
	}
}

func TestPatchErrors(t *testing.T) {
	srv, token, st := newJobsServer(t)
	h := srv.Handler()
	must(t, st.InsertJob(context.Background(), jobs.Job{Key: "x", Kind: jobs.KindTrain, State: jobs.StateQueued, Priority: 1, Epochs: 100, Arch: "standard", CreatedAt: 1}, []byte("x")))

	if rec := do(t, h, http.MethodPatch, "/v1/jobs/x", token, nil); rec.Code != http.StatusBadRequest {
		t.Errorf("PATCH no priority = %d, want 400", rec.Code)
	}
	if rec := do(t, h, http.MethodPatch, "/v1/jobs/x?priority=7", token, nil); rec.Code != http.StatusBadRequest {
		t.Errorf("PATCH bad priority = %d, want 400", rec.Code)
	}
	if rec := do(t, h, http.MethodPatch, "/v1/jobs/nope?priority=0", token, nil); rec.Code != http.StatusNotFound {
		t.Errorf("PATCH unknown = %d, want 404", rec.Code)
	}
}

func TestGetModelAndLog(t *testing.T) {
	srv, token, st := newJobsServer(t)
	ctx := context.Background()
	h := srv.Handler()

	// Unknown model → 404.
	if rec := do(t, h, http.MethodGet, "/v1/jobs/nope/model", token, nil); rec.Code != http.StatusNotFound {
		t.Errorf("model unknown = %d, want 404", rec.Code)
	}

	must(t, st.InsertJob(ctx, jobs.Job{Key: "s", Kind: jobs.KindTrain, State: jobs.StateSucceeded, Priority: 1, Epochs: 100, Arch: "standard", CreatedAt: 1}, []byte("s")))
	_, _ = st.DB().ExecContext(ctx, "INSERT INTO results(job_key,nam) VALUES('s',x'cafe')")
	_, _ = st.DB().ExecContext(ctx, "INSERT INTO job_log(job_key,line) VALUES('s','Epoch 0/100')")
	_, _ = st.DB().ExecContext(ctx, "INSERT INTO job_log(job_key,line) VALUES('s','Epoch 1/100')")

	rec := do(t, h, http.MethodGet, "/v1/jobs/s/model", token, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("model = %d, want 200", rec.Code)
	}
	if got := rec.Body.Bytes(); len(got) != 2 || got[0] != 0xca || got[1] != 0xfe {
		t.Errorf("model bytes = % x, want ca fe", got)
	}

	rec = do(t, h, http.MethodGet, "/v1/jobs/s/log", token, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("log = %d, want 200", rec.Code)
	}
	if got := rec.Body.String(); got != "Epoch 0/100\nEpoch 1/100" {
		t.Errorf("log = %q", got)
	}

	// Log for unknown job → 404.
	if rec := do(t, h, http.MethodGet, "/v1/jobs/nope/log", token, nil); rec.Code != http.StatusNotFound {
		t.Errorf("log unknown = %d, want 404", rec.Code)
	}
}

func TestPutPositionViaClock(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	// A monotonically advancing fake clock so created_at strictly increases.
	var tick int64
	srv.SetClock(func() time.Time { tick++; return time.Unix(1000+tick, 0) })
	h := srv.Handler()

	first := testsupport.ValidCapture()
	second := testsupport.Distinct(first, 0x7f)
	k1 := keyFor(first, jobs.KindTrain, 100, "standard")
	k2 := keyFor(second, jobs.KindTrain, 100, "standard")
	if k1 == k2 {
		t.Fatal("distinct captures must yield distinct keys")
	}

	do(t, h, http.MethodPut, "/v1/jobs/"+k1+"?kind=train&epochs=100", token, first)
	do(t, h, http.MethodPut, "/v1/jobs/"+k2+"?kind=train&epochs=100", token, second)

	if pos := getPosition(t, h, token, k1); pos != 1 {
		t.Errorf("first position = %d, want 1", pos)
	}
	if pos := getPosition(t, h, token, k2); pos != 2 {
		t.Errorf("second position = %d, want 2", pos)
	}
}

// ---- test helpers ----

func mustJSON(t *testing.T, rec *httptest.ResponseRecorder, v any) {
	t.Helper()
	if err := json.Unmarshal(rec.Body.Bytes(), v); err != nil {
		t.Fatalf("decode body: %v (body %s)", err, rec.Body.String())
	}
}

func errCode(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var e errorEnvelope
	if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil {
		t.Fatalf("decode error body: %v (body %s)", err, rec.Body.String())
	}
	return e.Error.Code
}

func getPosition(t *testing.T, h http.Handler, token, key string) int {
	t.Helper()
	rec := do(t, h, http.MethodGet, "/v1/jobs/"+key, token, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET %s = %d, want 200", key, rec.Code)
	}
	var jr jobResponse
	mustJSON(t, rec, &jr)
	if jr.Position == nil {
		t.Fatalf("job %s has nil position", key)
	}
	return *jr.Position
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
