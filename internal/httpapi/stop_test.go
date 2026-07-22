// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/worker"
)

// fakeStopper is a stand-in Stopper: it returns a canned error and records whether
// StopJob was actually invoked — so a test can assert that the unknown / terminal /
// queued / probe short-circuits turn the request away BEFORE ever reaching it (the
// crew F-3 wall keeping a probe key out of StopJob). onCall lets a case mutate the
// store mid-call to reproduce the crew-F4 stop-vs-just-finished race. No worker
// machinery involved.
type fakeStopper struct {
	err    error
	onCall func() // optional side effect run inside StopJob (e.g. flip the row terminal)
	called bool
}

func (f *fakeStopper) StopJob(key string) error {
	f.called = true
	if f.onCall != nil {
		f.onCall()
	}
	return f.err
}

// TestStopRouteAbsentWithoutStopper: with no Stopper wired (the API-only build or an
// old daemon), Handler never registers POST /v1/jobs/{key}/stop, so the mux answers
// its DEFAULT plain-text 404 — NOT the JSON error envelope — the same "server too
// old" cue the client falls back on. Asserting the text/plain body shape proves the
// route is genuinely absent: a registered handler would emit the JSON envelope.
func TestStopRouteAbsentWithoutStopper(t *testing.T) {
	srv, token, st := newJobsServer(t)
	// Seed a RUNNING train: were the route present it would 202, so the 404 here is
	// unambiguously the missing route, never a job-state rejection.
	seedJobState(t, st, "job", jobs.KindTrain, jobs.StateRunning)
	// No SetStopper call — the route must not exist.

	rec := do(t, srv.Handler(), http.MethodPost, "/v1/jobs/job/stop", token, nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404 (route absent), body %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Errorf("Content-Type = %q, want text/plain (mux default 404)", ct)
	}
	if body := strings.TrimSpace(rec.Body.String()); body != "404 page not found" {
		t.Errorf("body = %q, want the mux default %q", body, "404 page not found")
	}
	// It must NOT be the uniform JSON error envelope.
	var e errorEnvelope
	if json.Unmarshal(rec.Body.Bytes(), &e) == nil && e.Error.Code != "" {
		t.Errorf("got a JSON error envelope %+v, want the plain mux 404", e)
	}
}

// TestStopJob drives the full status mapping of POST /stop with a fake Stopper that
// records whether it was called: the short-circuits that must NOT touch the Stopper
// (unknown / terminal / queued / running probe — the F-3 wall), the running-train
// StopJob outcomes (nil→202, ErrNoCheckpoint→409, other→500), and the F4
// ErrNotRunning re-read (row flipped terminal → 204, still running → 409).
func TestStopJob(t *testing.T) {
	const key = "job"
	cases := []struct {
		name           string
		kind           string // "" => train
		state          string // "" => don't seed (unknown key)
		stopErr        error  // fake StopJob's return (only reached for a running train-lane job)
		raceToTerminal bool   // fake flips the row succeeded before returning (the F4 race)
		wantStatus     int
		wantCode       string // error-envelope code; "" for 202/204
		wantCalled     bool
	}{
		{name: "unknown key → 404 not_found", wantStatus: 404, wantCode: codeNotFound, wantCalled: false},
		{name: "terminal succeeded → 204", state: jobs.StateSucceeded, wantStatus: 204, wantCalled: false},
		{name: "terminal failed → 204", state: jobs.StateFailed, wantStatus: 204, wantCalled: false},
		{name: "queued → 409 no_checkpoint", state: jobs.StateQueued, wantStatus: 409, wantCode: codeNoCheckpoint, wantCalled: false},
		{
			name: "running probe_self → 409 (F-3 wall, not called)",
			kind: jobs.KindProbeSelf, state: jobs.StateRunning,
			wantStatus: 409, wantCode: codeNoCheckpoint, wantCalled: false,
		},
		{
			name: "running probe_e10 → 409 (F-3 wall, not called)",
			kind: jobs.KindProbeE10, state: jobs.StateRunning,
			wantStatus: 409, wantCode: codeNoCheckpoint, wantCalled: false,
		},
		{
			name:  "running train nil → 202 stopping",
			state: jobs.StateRunning, wantStatus: 202, wantCalled: true,
		},
		{
			name:  "running train ErrNoCheckpoint → 409",
			state: jobs.StateRunning, stopErr: worker.ErrNoCheckpoint,
			wantStatus: 409, wantCode: codeNoCheckpoint, wantCalled: true,
		},
		{
			name:  "running train ErrNotRunning + row now terminal → 204 (F4)",
			state: jobs.StateRunning, stopErr: worker.ErrNotRunning, raceToTerminal: true,
			wantStatus: 204, wantCalled: true,
		},
		{
			name:  "running train ErrNotRunning + row still running → 409 (F4)",
			state: jobs.StateRunning, stopErr: worker.ErrNotRunning,
			wantStatus: 409, wantCode: codeNoCheckpoint, wantCalled: true,
		},
		{
			name:  "running train other error → 500",
			state: jobs.StateRunning, stopErr: errors.New("boom"),
			wantStatus: 500, wantCode: codeInternal, wantCalled: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, token, st := newJobsServer(t)
			kind := tc.kind
			if kind == "" {
				kind = jobs.KindTrain
			}
			if tc.state != "" {
				seedJobState(t, st, key, kind, tc.state)
			}
			fake := &fakeStopper{err: tc.stopErr}
			if tc.raceToTerminal {
				fake.onCall = flipTerminal(t, st, key)
			}
			srv.SetStopper(fake)

			rec := do(t, srv.Handler(), http.MethodPost, "/v1/jobs/"+key+"/stop", token, nil)

			if rec.Code != tc.wantStatus {
				t.Fatalf("status = %d, want %d (body %s)", rec.Code, tc.wantStatus, rec.Body.String())
			}
			if tc.wantCode != "" {
				if code := errCode(t, rec); code != tc.wantCode {
					t.Errorf("error code = %q, want %q", code, tc.wantCode)
				}
			}
			if fake.called != tc.wantCalled {
				t.Errorf("StopJob called = %v, want %v", fake.called, tc.wantCalled)
			}
			// The 202 body is exactly {"state":"stopping"}.
			if tc.wantStatus == http.StatusAccepted {
				if got := strings.TrimSpace(rec.Body.String()); got != `{"state":"stopping"}` {
					t.Errorf("202 body = %q, want %q", got, `{"state":"stopping"}`)
				}
			}
		})
	}
}

// flipTerminal returns an onCall that flips the row to succeeded — the store side of
// the crew-F4 race (the run finished on its own between the handler's GetJob and its
// StopJob call, so StopJob's ErrNotRunning must re-read into a 204).
func flipTerminal(t *testing.T, st *store.Store, key string) func() {
	t.Helper()
	return func() {
		if _, err := st.DB().ExecContext(context.Background(),
			"UPDATE jobs SET state='succeeded' WHERE key=?", key); err != nil {
			t.Fatalf("race flip to terminal: %v", err)
		}
	}
}

// TestReachedInBodies: `reached` is null for probes and queued jobs and carries the
// stamped count for a terminal train, present on BOTH the single-job GET and the
// /v1/queue batch bodies (jobBody feeds both). The raw wire form is asserted for the
// null cases — a *int64 decode cannot tell null from absent.
func TestReachedInBodies(t *testing.T) {
	srv, token, st := newJobsServer(t)
	h := srv.Handler()
	ctx := context.Background()

	// A queued train (reached null), a running probe_e10 (probes never carry it), and
	// a terminal train stamped with a reached (an early stop at epoch 87).
	seedJobState(t, st, "q", jobs.KindTrain, jobs.StateQueued)
	seedJobState(t, st, "p", jobs.KindProbeE10, jobs.StateRunning)
	seedJobState(t, st, "t", jobs.KindTrain, jobs.StateSucceeded)
	if _, err := st.DB().ExecContext(ctx, "UPDATE jobs SET reached=87 WHERE key='t'"); err != nil {
		t.Fatalf("stamp reached: %v", err)
	}

	// GET: raw wire form for the null cases, decoded value for the stamped one.
	if body := do(t, h, http.MethodGet, "/v1/jobs/q", token, nil).Body.String(); !strings.Contains(body, `"reached":null`) {
		t.Errorf("queued GET must encode reached:null, got %s", body)
	}
	if body := do(t, h, http.MethodGet, "/v1/jobs/p", token, nil).Body.String(); !strings.Contains(body, `"reached":null`) {
		t.Errorf("probe GET must encode reached:null, got %s", body)
	}
	var jr jobResponse
	mustJSON(t, do(t, h, http.MethodGet, "/v1/jobs/t", token, nil), &jr)
	if jr.Reached == nil || *jr.Reached != 87 {
		t.Errorf("terminal train reached = %v, want 87", jr.Reached)
	}

	// POST /v1/queue carries reached for all three (null for q/p, 87 for t).
	rec := do(t, h, http.MethodPost, "/v1/queue", token, []byte(`{"keys":["q","p","t"]}`))
	if body := rec.Body.String(); !strings.Contains(body, `"reached":null`) {
		t.Errorf("queue body must include reached:null, got %s", body)
	}
	var qresp struct {
		Jobs map[string]struct {
			Reached *int64 `json:"reached"`
		} `json:"jobs"`
	}
	mustJSON(t, rec, &qresp)
	if got := qresp.Jobs["t"].Reached; got == nil || *got != 87 {
		t.Errorf("queue t reached = %v, want 87", got)
	}
	if got := qresp.Jobs["q"].Reached; got != nil {
		t.Errorf("queue q reached = %v, want null", got)
	}
	if got := qresp.Jobs["p"].Reached; got != nil {
		t.Errorf("queue p reached = %v, want null", got)
	}
}
