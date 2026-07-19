// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package httpapi

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

const (
	maxQueueBodyBytes = 64 << 10 // 64 KiB — this is a read; hundreds of 64-hex keys, never a WAV
	maxQueueKeys      = 256
)

// queueRequest is the POST /v1/queue body: the caller's own job keys (it computed
// them, so it already knows them).
type queueRequest struct {
	Keys []string `json:"keys"`
}

// queueEntry is one result: the full job body (inlined) plus epochs_ahead, or just
// {"found":false} for an unknown / GC'd key. The embedded *jobResponse marshals
// its fields inline when present and contributes nothing when nil.
type queueEntry struct {
	*jobResponse
	Found       bool   `json:"found"`
	EpochsAhead *int64 `json:"epochs_ahead"`
}

// queueResponse maps each requested key to its entry. An object (not an array)
// dedupes repeated keys and gives the client an O(1) lookup by key.
type queueResponse struct {
	Jobs map[string]queueEntry `json:"jobs"`
}

// handleQueue answers a batch status+scheduling query: one call for many of the
// caller's jobs, each with its lane position and epochs_ahead. epochs_ahead is the
// lane epochs ahead of that job (a serial-drain sum, exact wall-work at cap=1; the
// client divides by cap for an ETA), summed across ALL callers' jobs in the lane —
// so the client can compute a queue ETA it could not derive from its own keys
// alone. Read-only; the daemon reports the raw numbers and never the ETA. Unknown
// keys come back {"found":false}; the batch still succeeds.
func (s *Server) handleQueue(w http.ResponseWriter, r *http.Request) {
	req, ok := s.readQueueRequest(w, r)
	if !ok {
		return
	}
	entries, err := s.store.QueueView(r.Context(), req.Keys)
	if err != nil {
		s.internal(w, "queue view", err)
		return
	}
	out := make(map[string]queueEntry, len(entries))
	for key, e := range entries {
		if !e.Found {
			out[key] = queueEntry{Found: false}
			continue
		}
		body := jobBody(e.Job)
		body.Position = e.Position
		out[key] = queueEntry{jobResponse: &body, Found: true, EpochsAhead: e.EpochsAhead}
	}
	writeJSON(w, http.StatusOK, queueResponse{Jobs: out})
}

func (s *Server) readQueueRequest(w http.ResponseWriter, r *http.Request) (queueRequest, bool) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxQueueBodyBytes))
	if err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			// 400 (not 413) keeps the §3 status/code table closed — the contract
			// enumerates the codes, and this is just a malformed (oversize) request.
			writeError(w, http.StatusBadRequest, codeBadRequest, "request too large")
			return queueRequest{}, false
		}
		writeError(w, http.StatusBadRequest, codeBadRequest, "error reading request body")
		return queueRequest{}, false
	}
	var req queueRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, codeBadRequest, "invalid JSON body")
		return queueRequest{}, false
	}
	if len(req.Keys) > maxQueueKeys {
		writeError(w, http.StatusBadRequest, codeBadRequest,
			fmt.Sprintf("too many keys (max %d)", maxQueueKeys))
		return queueRequest{}, false
	}
	return req, true
}
