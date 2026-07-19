// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package httpapi

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/store"
)

// queueEntryWire is a decode-side view of one /v1/queue entry.
type queueEntryWire struct {
	Found       bool   `json:"found"`
	State       string `json:"state"`
	Kind        string `json:"kind"`
	Position    *int   `json:"position"`
	EpochsAhead *int64 `json:"epochs_ahead"`
	Epochs      int    `json:"epochs"`
}

func queueJob(t *testing.T, st *store.Store, key string, createdAt int64, epochs int) {
	t.Helper()
	must(t, st.InsertJob(context.Background(), jobs.Job{
		Key: key, Kind: jobs.KindTrain, State: jobs.StateQueued,
		Priority: 1, Epochs: epochs, Arch: "standard", CreatedAt: createdAt,
	}, []byte(key)))
}

func TestQueueBatch(t *testing.T) {
	srv, token, st := newJobsServer(t)
	h := srv.Handler()
	queueJob(t, st, "t1", 100, 100)
	queueJob(t, st, "t2", 200, 100)

	rec := do(t, h, http.MethodPost, "/v1/queue", token, []byte(`{"keys":["t1","t2","ghost"]}`))
	if rec.Code != http.StatusOK {
		t.Fatalf("POST = %d (%s)", rec.Code, rec.Body.String())
	}
	var resp struct {
		Jobs map[string]queueEntryWire `json:"jobs"`
	}
	mustJSON(t, rec, &resp)
	if len(resp.Jobs) != 3 {
		t.Fatalf("jobs = %d, want 3", len(resp.Jobs))
	}

	t1 := resp.Jobs["t1"]
	if !t1.Found || t1.State != jobs.StateQueued || t1.Position == nil || *t1.Position != 1 ||
		t1.EpochsAhead == nil || *t1.EpochsAhead != 0 {
		t.Errorf("t1 = %+v (pos %v ahead %v)", t1, t1.Position, t1.EpochsAhead)
	}
	t2 := resp.Jobs["t2"]
	if t2.Position == nil || *t2.Position != 2 || t2.EpochsAhead == nil || *t2.EpochsAhead != 100 {
		t.Errorf("t2 pos = %v, ahead = %v, want 2 / 100", t2.Position, t2.EpochsAhead)
	}
	if resp.Jobs["ghost"].Found {
		t.Errorf("ghost: found = true, want false")
	}
}

func TestQueueEmptyList(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/queue", token, []byte(`{"keys":[]}`))
	if rec.Code != http.StatusOK {
		t.Fatalf("POST = %d, want 200", rec.Code)
	}
	var resp struct {
		Jobs map[string]queueEntryWire `json:"jobs"`
	}
	mustJSON(t, rec, &resp)
	if len(resp.Jobs) != 0 {
		t.Errorf("jobs = %d, want 0", len(resp.Jobs))
	}
}

func TestQueueTooManyKeys(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	keys := make([]string, maxQueueKeys+1)
	for i := range keys {
		keys[i] = "k"
	}
	body, err := json.Marshal(map[string][]string{"keys": keys})
	if err != nil {
		t.Fatal(err)
	}
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/queue", token, body)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("POST %d keys = %d, want 400", len(keys), rec.Code)
	}
}

func TestQueueOversizeBody(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	// A body over the 64 KiB cap trips MaxBytesReader before parse → 400 (not 413,
	// which is outside §3's status table).
	big := make([]string, 3000)
	for i := range big {
		big[i] = "0123456789012345678901234567890123456789" // 40 chars each → ~130 KiB total
	}
	body, err := json.Marshal(map[string][]string{"keys": big})
	if err != nil {
		t.Fatal(err)
	}
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/queue", token, body)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("oversize POST = %d, want 400", rec.Code)
	}
}

func TestQueueInvalidJSON(t *testing.T) {
	srv, token, _ := newJobsServer(t)
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/queue", token, []byte(`{"keys": nope}`))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("POST bad JSON = %d, want 400", rec.Code)
	}
}
