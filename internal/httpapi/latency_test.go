// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package httpapi

import (
	"context"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"testing"

	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/testsupport"
)

func latPtr(n int64) *int64 { return &n }

// TestPutLatencyAcceptedAndStored: the param rides the key and lands in the row.
// It is valid on EVERY kind — the trim is a property of the take, not of the work
// (a probe measured against another alignment predicts a different task).
func TestPutLatencyAcceptedAndStored(t *testing.T) {
	for _, kind := range []string{jobs.KindTrain, jobs.KindProbeSelf, jobs.KindProbeE10} {
		t.Run(kind, func(t *testing.T) {
			srv, token, st := newJobsServer(t)
			h := srv.Handler()
			wav := testsupport.ValidCapture()
			key := keyForLatency(wav, kind, 100, "standard", latPtr(1101))
			url := "/v1/jobs/" + key + "?kind=" + kind + "&epochs=100&arch=standard&priority=1&latency=1101"

			if rec := do(t, h, http.MethodPut, url, token, wav); rec.Code != http.StatusCreated {
				t.Fatalf("PUT = %d, want 201 (body %s)", rec.Code, rec.Body.String())
			}
			j, ok, err := st.GetJob(context.Background(), key)
			if err != nil || !ok {
				t.Fatalf("GetJob: ok=%v err=%v", ok, err)
			}
			if j.Latency == nil || *j.Latency != 1101 {
				t.Errorf("stored latency = %v, want 1101", j.Latency)
			}
		})
	}
}

// TestPutWithoutLatencyStoresNull pins the unchanged path: no param ⇒ no column
// value ⇒ no --latency ⇒ the trainer's own detector, exactly as before.
func TestPutWithoutLatencyStoresNull(t *testing.T) {
	srv, token, st := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	key := keyFor(wav, jobs.KindTrain, 100, "standard")

	rec := do(t, h, http.MethodPut,
		"/v1/jobs/"+key+"?kind=train&epochs=100&arch=standard&priority=1", token, wav)
	if rec.Code != http.StatusCreated {
		t.Fatalf("PUT = %d, want 201 (body %s)", rec.Code, rec.Body.String())
	}
	j, ok, err := st.GetJob(context.Background(), key)
	if err != nil || !ok {
		t.Fatalf("GetJob: ok=%v err=%v", ok, err)
	}
	if j.Latency != nil {
		t.Errorf("stored latency = %v, want NULL", *j.Latency)
	}
}

// TestPutLatencyIsPartOfTheKey: the same wav at another trim is another job, so a
// key computed without the latency line is a key_mismatch, and 0 ≠ absent.
func TestPutLatencyIsPartOfTheKey(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()

	// The base-formula key (no latency line) submitted WITH ?latency= is rejected.
	plain := keyFor(wav, jobs.KindTrain, 100, "standard")
	rec := do(t, h, http.MethodPut,
		"/v1/jobs/"+plain+"?kind=train&epochs=100&arch=standard&latency=1101", token, wav)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("stale key with latency = %d, want 400", rec.Code)
	}
	if code := errCode(t, rec); code != codeKeyMismatch {
		t.Errorf("error code = %q, want %q", code, codeKeyMismatch)
	}

	// …and the latency-bearing key submitted WITHOUT the param is equally rejected.
	withLat := keyForLatency(wav, jobs.KindTrain, 100, "standard", latPtr(1101))
	rec = do(t, h, http.MethodPut,
		"/v1/jobs/"+withLat+"?kind=train&epochs=100&arch=standard", token, wav)
	if rec.Code != http.StatusBadRequest || errCode(t, rec) != codeKeyMismatch {
		t.Errorf("latency key without param = %d/%s, want 400 key_mismatch", rec.Code, errCode(t, rec))
	}

	// latency=0 is a value of its own: it does not collide with an absent latency.
	if keyForLatency(wav, jobs.KindTrain, 100, "standard", latPtr(0)) == plain {
		t.Fatal("latency=0 computed the same key as an absent latency")
	}
	zero := keyForLatency(wav, jobs.KindTrain, 100, "standard", latPtr(0))
	if rec := do(t, h, http.MethodPut,
		"/v1/jobs/"+zero+"?kind=train&epochs=100&arch=standard&latency=0", token, wav); rec.Code != http.StatusCreated {
		t.Fatalf("latency=0 PUT = %d, want 201 (body %s)", rec.Code, rec.Body.String())
	}
}

// TestPutLatencyRejectsGarbage: out of range or unparseable is a clean 400 — the
// daemon never clamps a nonsense value into something plausible.
func TestPutLatencyRejectsGarbage(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	h := srv.Handler()
	wav := testsupport.ValidCapture()
	key := keyFor(wav, jobs.KindTrain, 100, "standard")

	for _, raw := range []string{
		"-1",
		strconv.Itoa(jobs.MaxLatencySamples + 1),
		"1101.5",
		"abc",
		" 1101",
		"1e3",
		"01101", // non-canonical spellings hash differently on the app side: name the
		"+1101", // real problem here instead of answering a puzzling key_mismatch
	} {
		rec := do(t, h, http.MethodPut,
			"/v1/jobs/"+key+"?kind=train&epochs=100&arch=standard&latency="+url.QueryEscape(raw), token, wav)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("latency=%q → %d, want 400", raw, rec.Code)
			continue
		}
		if code := errCode(t, rec); code != codeBadRequest {
			t.Errorf("latency=%q → code %q, want %q", raw, code, codeBadRequest)
		}
	}
	// The bad requests are refused BEFORE any row is written.
	if rec := do(t, h, http.MethodGet, "/v1/jobs/"+key, token, nil); rec.Code != http.StatusNotFound {
		t.Errorf("a rejected PUT left a row: GET = %d, want 404", rec.Code)
	}
	// The ceiling itself is legal.
	max := int64(jobs.MaxLatencySamples)
	maxKey := keyForLatency(wav, jobs.KindTrain, 100, "standard", &max)
	if rec := do(t, h, http.MethodPut,
		"/v1/jobs/"+maxKey+"?kind=train&epochs=100&arch=standard&latency="+strconv.FormatInt(max, 10),
		token, wav); rec.Code != http.StatusCreated {
		t.Errorf("latency=%d → %d, want 201", max, rec.Code)
	}
}

// TestPutTrainMoreLatencyMustMatchParent drives the store's eligibility rule through
// the wire: a mismatch (absence included) is 409 base_unavailable, never a silent
// continuation on the wrong alignment.
func TestPutTrainMoreLatencyMustMatchParent(t *testing.T) {
	wav := testsupport.ValidCapture()
	ckpt := []byte("parent-ckpt-blob")

	cases := []struct {
		name          string
		parent, child *int64
		wantCode      int
	}{
		{"both absent", nil, nil, http.StatusCreated},
		{"equal", latPtr(1101), latPtr(1101), http.StatusCreated},
		{"different", latPtr(1101), latPtr(1102), http.StatusConflict},
		{"parent absent, child set", nil, latPtr(1101), http.StatusConflict},
		{"parent set, child absent", latPtr(1101), nil, http.StatusConflict},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, token, st := newJobsServer(t)
			h := srv.Handler()

			// Seed a succeeded parent carrying tc.parent's latency.
			parentKey := keyForLatency(wav, jobs.KindTrain, 200, "standard", tc.parent)
			ctx := context.Background()
			must(t, st.InsertJob(ctx, jobs.Job{Key: parentKey, Kind: jobs.KindTrain,
				State: jobs.StateQueued, Priority: 1, Epochs: 200, Arch: "standard",
				CreatedAt: 1, Latency: tc.parent}, wav))
			mustExec(t, st, "UPDATE jobs SET state='running', started_at=1 WHERE key=?", parentKey)
			if ok, err := st.FinishTrainSuccess(ctx, parentKey, 2, []byte("nam"), "{}", nil, ckpt); err != nil || !ok {
				t.Fatalf("finish parent: ok=%v err=%v", ok, err)
			}

			childKey := keyForMoreLatency(wav, 400, "standard", parentKey, tc.child)
			url := "/v1/jobs/" + childKey + "?kind=train_more&epochs=400&arch=standard&base=" + parentKey
			if tc.child != nil {
				url += "&latency=" + strconv.FormatInt(*tc.child, 10)
			}
			rec := do(t, h, http.MethodPut, url, token, wav)
			if rec.Code != tc.wantCode {
				t.Fatalf("PUT train_more = %d, want %d (body %s)", rec.Code, tc.wantCode, rec.Body.String())
			}
			if tc.wantCode != http.StatusConflict {
				return
			}
			if code := errCode(t, rec); code != codeBaseUnavailable {
				t.Errorf("error code = %q, want %q", code, codeBaseUnavailable)
			}
			if msg := errMessage(t, rec); !strings.Contains(msg, "latency does not match the parent") {
				t.Errorf("error message = %q, want the latency reason", msg)
			}
		})
	}
}
