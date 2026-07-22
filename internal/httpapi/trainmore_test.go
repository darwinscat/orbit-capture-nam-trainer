// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package httpapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"

	"orbit-capture-nam-trainer/internal/jobkey"
	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/testsupport"
)

// keyForMore computes a train_more child key with the fixture profile the test
// server advertises (mirrors keyFor for the base-line formula).
func keyForMore(wav []byte, epochs int, arch, base string) string {
	return jobkey.ComputeTrainMore(jobkey.SHA256Hex(wav), epochs, arch, tpNam, tpDriver, tpSignal, base)
}

// ---- store-seeding helpers (a train_more parent must be a succeeded job that left
// a stored checkpoint; probes seed the app's probe→train flow) --------------------

func mustExec(t *testing.T, st *store.Store, query string, args ...any) {
	t.Helper()
	if _, err := st.DB().ExecContext(context.Background(), query, args...); err != nil {
		t.Fatalf("exec %q: %v", query, err)
	}
}

// seedSucceededTrain lands a succeeded train row carrying wav (→ wav_sha) and ckpt.
// The finish transition guards on state='running', so the row is nudged there first.
func seedSucceededTrain(t *testing.T, st *store.Store, key string, epochs int, arch string, wav, ckpt []byte) {
	t.Helper()
	ctx := context.Background()
	must(t, st.InsertJob(ctx, jobs.Job{Key: key, Kind: jobs.KindTrain, State: jobs.StateQueued,
		Priority: 1, Epochs: epochs, Arch: arch, CreatedAt: 1}, wav))
	mustExec(t, st, "UPDATE jobs SET state='running', started_at=1 WHERE key=?", key)
	if ok, err := st.FinishTrainSuccess(ctx, key, 2, []byte("nam-"+key), "{}", nil, ckpt); err != nil || !ok {
		t.Fatalf("finish train %s: ok=%v err=%v", key, ok, err)
	}
}

// seedSucceededProbeE10 lands a succeeded probe_e10 that stored a ckpt but no model
// (nam=NULL) — the standard probe→train seed.
func seedSucceededProbeE10(t *testing.T, st *store.Store, key string, wav, ckpt []byte) {
	t.Helper()
	ctx := context.Background()
	must(t, st.InsertJob(ctx, jobs.Job{Key: key, Kind: jobs.KindProbeE10, State: jobs.StateQueued,
		Priority: 1, Epochs: jobs.ProbeE10Epochs, Arch: "standard", CreatedAt: 1}, wav))
	mustExec(t, st, "UPDATE jobs SET state='running', started_at=1 WHERE key=?", key)
	if ok, err := st.FinishProbeE10(ctx, key, 2, 0.05, ckpt); err != nil || !ok {
		t.Fatalf("finish probe_e10 %s: ok=%v err=%v", key, ok, err)
	}
}

// seedSucceededProbeSelf lands a succeeded probe_self — killed pre-epoch, it never
// has a checkpoint, so it is an ineligible parent.
func seedSucceededProbeSelf(t *testing.T, st *store.Store, key string, wav []byte) {
	t.Helper()
	ctx := context.Background()
	must(t, st.InsertJob(ctx, jobs.Job{Key: key, Kind: jobs.KindProbeSelf, State: jobs.StateQueued,
		Priority: 1, Epochs: jobs.ProbeSelfEpochs, Arch: "standard", CreatedAt: 1}, wav))
	mustExec(t, st, "UPDATE jobs SET state='running', started_at=1 WHERE key=?", key)
	if ok, err := st.FinishProbeSelf(ctx, key, 2, jobs.VerdictPass, nil); err != nil || !ok {
		t.Fatalf("finish probe_self %s: ok=%v err=%v", key, ok, err)
	}
}

func errMessage(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var e errorEnvelope
	if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil {
		t.Fatalf("decode error body: %v (body %s)", err, rec.Body.String())
	}
	return e.Error.Message
}

// ---- tests ----------------------------------------------------------------------

// TestPutTrainMore409Matrix drives every ineligible-parent path through the HTTP
// layer. Each is a WELL-FORMED request (valid base shape, arch, epochs, key) that the
// store's single eligibility check rejects — so the observed contract is 409
// base_unavailable with a human reason, never a 400.
func TestPutTrainMore409Matrix(t *testing.T) {
	parentWav := testsupport.ValidCapture()
	otherWav := testsupport.Distinct(parentWav, 0x7f)
	ckpt := []byte("parent-ckpt-blob")

	cases := []struct {
		name string
		// setup seeds the store and returns the base key, the wav the child uploads,
		// the arch it asks for, and its epochs; the child key is computed from those.
		setup func(t *testing.T, st *store.Store) (base string, childWav []byte, childArch string, childEpochs int)
	}{
		{"unknown parent", func(t *testing.T, st *store.Store) (string, []byte, string, int) {
			return strings.Repeat("a", 64), parentWav, "standard", 400 // no such parent row
		}},
		{"failed parent", func(t *testing.T, st *store.Store) (string, []byte, string, int) {
			key := keyFor(parentWav, jobs.KindTrain, 200, "standard")
			seedSucceededTrain(t, st, key, 200, "standard", parentWav, ckpt)
			mustExec(t, st, "UPDATE jobs SET state='failed' WHERE key=?", key) // ckpt present, wrong state
			return key, parentWav, "standard", 400
		}},
		{"probe_self parent (no ckpt)", func(t *testing.T, st *store.Store) (string, []byte, string, int) {
			key := keyFor(parentWav, jobs.KindProbeSelf, 0, "standard")
			seedSucceededProbeSelf(t, st, key, parentWav) // succeeded but never stored a ckpt
			return key, parentWav, "standard", 400
		}},
		{"gc'd-ckpt parent", func(t *testing.T, st *store.Store) (string, []byte, string, int) {
			key := keyFor(parentWav, jobs.KindTrain, 200, "standard")
			seedSucceededTrain(t, st, key, 200, "standard", parentWav, ckpt)
			mustExec(t, st, "UPDATE results SET ckpt=NULL WHERE job_key=?", key) // ckpt aged out
			return key, parentWav, "standard", 400
		}},
		{"wav mismatch", func(t *testing.T, st *store.Store) (string, []byte, string, int) {
			key := keyFor(parentWav, jobs.KindTrain, 200, "standard")
			seedSucceededTrain(t, st, key, 200, "standard", parentWav, ckpt)
			return key, otherWav, "standard", 400 // child uploads a different take
		}},
		{"arch mismatch", func(t *testing.T, st *store.Store) (string, []byte, string, int) {
			key := keyFor(parentWav, jobs.KindTrain, 200, "standard")
			seedSucceededTrain(t, st, key, 200, "standard", parentWav, ckpt)
			return key, parentWav, "lite", 400 // child asks for a different arch
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, token, st := newJobsServer(t)
			h := srv.Handler()
			base, childWav, childArch, childEpochs := tc.setup(t, st)
			childKey := keyForMore(childWav, childEpochs, childArch, base)
			target := "/v1/jobs/" + childKey + "?kind=train_more&base=" + base +
				"&epochs=" + strconv.Itoa(childEpochs) + "&arch=" + childArch

			rec := do(t, h, http.MethodPut, target, token, childWav)
			if rec.Code != http.StatusConflict {
				t.Fatalf("%s: code = %d, want 409 (body %s)", tc.name, rec.Code, rec.Body.String())
			}
			if code := errCode(t, rec); code != codeBaseUnavailable {
				t.Errorf("%s: error code = %q, want %q", tc.name, code, codeBaseUnavailable)
			}
			if msg := errMessage(t, rec); strings.TrimSpace(msg) == "" {
				t.Errorf("%s: human message must be non-empty", tc.name)
			}
		})
	}
}

// TestPutTrainMoreOffProbeE10Accepted covers the app's standard flow: a succeeded
// probe_e10 with a checkpoint seeds a train_more, which resumes at the probe's 10
// epochs.
func TestPutTrainMoreOffProbeE10Accepted(t *testing.T) {
	srv, token, st := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	parentKey := keyFor(wav, jobs.KindProbeE10, 0, "standard")
	seedSucceededProbeE10(t, st, parentKey, wav, []byte("probe-ckpt"))

	childKey := keyForMore(wav, 40, "standard", parentKey)
	target := "/v1/jobs/" + childKey + "?kind=train_more&base=" + parentKey + "&epochs=40&arch=standard"
	rec := do(t, h, http.MethodPut, target, token, wav)
	if rec.Code != http.StatusCreated {
		t.Fatalf("PUT = %d, want 201 (body %s)", rec.Code, rec.Body.String())
	}

	rec = do(t, h, http.MethodGet, "/v1/jobs/"+childKey, token, nil)
	var jr jobResponse
	mustJSON(t, rec, &jr)
	if jr.Kind != jobs.KindTrainMore || jr.Epochs != 40 {
		t.Errorf("job = %+v, want train_more epochs 40", jr)
	}
	if jr.StartEpoch == nil || *jr.StartEpoch != jobs.ProbeE10Epochs {
		t.Errorf("start_epoch = %v, want %d (the probe's epochs)", jr.StartEpoch, jobs.ProbeE10Epochs)
	}
}

// TestPutTrainMoreEpochsNotGreaterIs409 pins the layering decision: epochs≤parent is
// left to the store's ONE eligibility check (the plan puts epochs>parent in the
// eligibility set), so the HTTP layer does not re-implement it. It therefore surfaces
// as the store's 409 base_unavailable — not the 400 the plan's prose nominally hangs
// on a "client bug". The observed contract wins; the rule lives in exactly one place.
func TestPutTrainMoreEpochsNotGreaterIs409(t *testing.T) {
	srv, token, st := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	parentKey := keyFor(wav, jobs.KindTrain, 200, "standard")
	seedSucceededTrain(t, st, parentKey, 200, "standard", wav, []byte("ckpt"))

	childKey := keyForMore(wav, 200, "standard", parentKey) // == parent, must be strictly greater
	target := "/v1/jobs/" + childKey + "?kind=train_more&base=" + parentKey + "&epochs=200&arch=standard"
	rec := do(t, h, http.MethodPut, target, token, wav)
	if rec.Code != http.StatusConflict {
		t.Fatalf("epochs≤parent = %d, want 409 (store eligibility), body %s", rec.Code, rec.Body.String())
	}
	if code := errCode(t, rec); code != codeBaseUnavailable {
		t.Errorf("error code = %q, want %q", code, codeBaseUnavailable)
	}
}

// TestPutTrainMoreMissingEpochs: the epochs parse gate covers train_more (crew F5),
// so a missing epochs is a clean 400 here rather than a misleading key_mismatch after
// the recompute.
func TestPutTrainMoreMissingEpochs(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	h := srv.Handler()
	base := strings.Repeat("a", 64)
	target := "/v1/jobs/k?kind=train_more&base=" + base // no epochs
	rec := do(t, h, http.MethodPut, target, token, testsupport.ValidCapture())
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("missing epochs = %d, want 400 (body %s)", rec.Code, rec.Body.String())
	}
	if code := errCode(t, rec); code != codeBadRequest {
		t.Errorf("error code = %q, want %q", code, codeBadRequest)
	}
	if msg := errMessage(t, rec); !strings.Contains(msg, "epochs required") {
		t.Errorf("message = %q, want it to mention 'epochs required'", msg)
	}
}

// TestPutTrainMoreBaseParamValidation: a malformed/absent base for train_more, and a
// stray base on any other kind, are client bugs → 400 (never 409).
func TestPutTrainMoreBaseParamValidation(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	cases := []struct{ name, target string }{
		{"train_more missing base", "/v1/jobs/k?kind=train_more&epochs=400"},
		{"train_more short base", "/v1/jobs/k?kind=train_more&epochs=400&base=abc"},
		{"train_more uppercase base", "/v1/jobs/k?kind=train_more&epochs=400&base=" + strings.Repeat("A", 64)},
		{"train_more non-hex base", "/v1/jobs/k?kind=train_more&epochs=400&base=" + strings.Repeat("g", 64)},
		{"base on kind=train", "/v1/jobs/k?kind=train&epochs=400&base=" + strings.Repeat("a", 64)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := do(t, h, http.MethodPut, tc.target, token, wav)
			if rec.Code != http.StatusBadRequest {
				t.Errorf("%s: code = %d, want 400 (body %s)", tc.name, rec.Code, rec.Body.String())
			}
			if code := errCode(t, rec); code != codeBadRequest {
				t.Errorf("%s: code = %q, want %q", tc.name, code, codeBadRequest)
			}
		})
	}
}

// TestPutBadArch: arch validation applies to every kind, so a path-traversal or
// whitespace-bearing arch is rejected at 400 before it can reach the key preimage.
func TestPutBadArch(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	for _, bad := range []string{"../x", "a b", "a/b", "a.b", "a\tb"} {
		target := "/v1/jobs/k?kind=train&epochs=100&arch=" + url.QueryEscape(bad)
		rec := do(t, h, http.MethodPut, target, token, wav)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("arch=%q: code = %d, want 400 (body %s)", bad, rec.Code, rec.Body.String())
		}
		if code := errCode(t, rec); code != codeBadRequest {
			t.Errorf("arch=%q: code = %q, want %q", bad, code, codeBadRequest)
		}
	}
}

// TestPutTrainMoreHappyThenIdempotentAfterParentDelete: 201 with a queue position,
// then — after DELETing the parent — a resubmit still answers 200 from the child's
// own row, PROVING the idempotent short-circuit never re-checks the (now gone) parent.
func TestPutTrainMoreHappyThenIdempotentAfterParentDelete(t *testing.T) {
	srv, token, st := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	parentKey := keyFor(wav, jobs.KindTrain, 200, "standard")
	seedSucceededTrain(t, st, parentKey, 200, "standard", wav, []byte("parent-ckpt"))

	childKey := keyForMore(wav, 400, "standard", parentKey)
	target := "/v1/jobs/" + childKey + "?kind=train_more&base=" + parentKey + "&epochs=400&arch=standard"

	rec := do(t, h, http.MethodPut, target, token, wav)
	if rec.Code != http.StatusCreated {
		t.Fatalf("first PUT = %d, want 201 (body %s)", rec.Code, rec.Body.String())
	}
	var pr putResponse
	mustJSON(t, rec, &pr)
	if pr.Key != childKey || pr.State != jobs.StateQueued || pr.Position == nil || *pr.Position != 1 {
		t.Errorf("PUT response = %+v", pr)
	}

	// The snapshot made the child self-contained: delete the PARENT, then resubmit.
	if rec := do(t, h, http.MethodDelete, "/v1/jobs/"+parentKey, token, nil); rec.Code != http.StatusNoContent {
		t.Fatalf("DELETE parent = %d, want 204", rec.Code)
	}
	rec = do(t, h, http.MethodPut, target, token, wav)
	if rec.Code != http.StatusOK {
		t.Fatalf("idempotent resubmit after parent delete = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
}

// TestTrainMoreStartEpochInGetAndQueueBodies: start_epoch is present (and null for a
// plain job) on BOTH the single-job GET and the /v1/queue batch bodies.
func TestTrainMoreStartEpochInGetAndQueueBodies(t *testing.T) {
	srv, token, st := newJobsServer(t)
	h := srv.Handler()

	// A plain train job (start_epoch must be present and null).
	plainWav := testsupport.ValidCapture()
	plainKey := keyFor(plainWav, jobs.KindTrain, 100, "standard")
	if rec := do(t, h, http.MethodPut, "/v1/jobs/"+plainKey+"?kind=train&epochs=100", token, plainWav); rec.Code != http.StatusCreated {
		t.Fatalf("PUT plain = %d, want 201 (body %s)", rec.Code, rec.Body.String())
	}

	// A train_more child (start_epoch = the parent's 200).
	parentWav := testsupport.Distinct(plainWav, 0x55)
	parentKey := keyFor(parentWav, jobs.KindTrain, 200, "standard")
	seedSucceededTrain(t, st, parentKey, 200, "standard", parentWav, []byte("ckpt"))
	childKey := keyForMore(parentWav, 400, "standard", parentKey)
	childTarget := "/v1/jobs/" + childKey + "?kind=train_more&base=" + parentKey + "&epochs=400&arch=standard"
	if rec := do(t, h, http.MethodPut, childTarget, token, parentWav); rec.Code != http.StatusCreated {
		t.Fatalf("PUT child = %d, want 201 (body %s)", rec.Code, rec.Body.String())
	}

	// GET plain: assert the raw wire form (a *int64 decode can't tell null from absent).
	rec := do(t, h, http.MethodGet, "/v1/jobs/"+plainKey, token, nil)
	if body := rec.Body.String(); !strings.Contains(body, `"start_epoch":null`) {
		t.Errorf("plain GET must encode start_epoch:null, got %s", body)
	}
	// GET child: start_epoch = 200.
	rec = do(t, h, http.MethodGet, "/v1/jobs/"+childKey, token, nil)
	var jr jobResponse
	mustJSON(t, rec, &jr)
	if jr.StartEpoch == nil || *jr.StartEpoch != 200 {
		t.Errorf("child GET start_epoch = %v, want 200", jr.StartEpoch)
	}

	// POST /v1/queue carries start_epoch for both entries.
	rec = do(t, h, http.MethodPost, "/v1/queue", token, []byte(`{"keys":["`+plainKey+`","`+childKey+`"]}`))
	if body := rec.Body.String(); !strings.Contains(body, `"start_epoch":null`) {
		t.Errorf("queue body must include start_epoch:null for the plain job, got %s", body)
	}
	var qresp struct {
		Jobs map[string]struct {
			StartEpoch *int64 `json:"start_epoch"`
		} `json:"jobs"`
	}
	mustJSON(t, rec, &qresp)
	if got := qresp.Jobs[childKey].StartEpoch; got == nil || *got != 200 {
		t.Errorf("queue child start_epoch = %v, want 200", got)
	}
	if got := qresp.Jobs[plainKey].StartEpoch; got != nil {
		t.Errorf("queue plain start_epoch = %v, want null", got)
	}
}

// TestPutTrainMoreKeyMismatchPlainFormula: a client that keys a train_more with the
// base-free (plain) formula gets a 400 key_mismatch — the base line is load-bearing.
func TestPutTrainMoreKeyMismatchPlainFormula(t *testing.T) {
	srv, token, st := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	parentKey := keyFor(wav, jobs.KindTrain, 200, "standard")
	seedSucceededTrain(t, st, parentKey, 200, "standard", wav, []byte("ckpt"))

	// Wrong: the plain formula (jobkey.Compute, no base line) for a train_more key.
	wrongKey := keyFor(wav, jobs.KindTrainMore, 400, "standard")
	target := "/v1/jobs/" + wrongKey + "?kind=train_more&base=" + parentKey + "&epochs=400&arch=standard"
	rec := do(t, h, http.MethodPut, target, token, wav)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("plain-formula key = %d, want 400 (body %s)", rec.Code, rec.Body.String())
	}
	if code := errCode(t, rec); code != codeKeyMismatch {
		t.Errorf("error code = %q, want %q", code, codeKeyMismatch)
	}
}
