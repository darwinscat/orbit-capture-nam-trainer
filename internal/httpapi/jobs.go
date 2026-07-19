// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package httpapi

import (
	"context"
	"errors"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"

	"orbit-capture-nam-trainer/internal/jobkey"
	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/sysutil"
	"orbit-capture-nam-trainer/internal/wav"
)

// putResponse is the PUT body: compact identity + queue position (the design notes).
type putResponse struct {
	Key      string `json:"key"`
	State    string `json:"state"`
	Position *int   `json:"position"`
}

// jobResponse is the GET body — raw numbers only; the CLIENT formats progress and
// applies all triage policy (the design notes).
type jobResponse struct {
	Kind      string   `json:"kind"`
	State     string   `json:"state"`
	Priority  int      `json:"priority"`
	Position  *int     `json:"position"`
	Epoch     *int64   `json:"epoch"`
	Epochs    int      `json:"epochs"`
	SPerEpoch *float64 `json:"s_per_epoch"`
	EtaS      *int64   `json:"eta_s"`
	Verdict   *string  `json:"verdict"`
	ESR       *float64 `json:"esr"`
	Error     *errBody `json:"error"`
	HasModel  bool     `json:"has_model"`
}

type errBody struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// handlePutJob enqueues a capture. See the design notes for the validation
// order. An existing key is answered idempotently BEFORE the (27 MB) body is
// read, so a client that pre-checks can abort the upload.
func (s *Server) handlePutJob(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	key := r.PathValue("key")

	q := r.URL.Query()
	kind := q.Get("kind")
	if !jobs.ValidKind(kind) {
		writeError(w, http.StatusBadRequest, codeBadRequest, "missing or unknown kind")
		return
	}
	arch := q.Get("arch")
	if arch == "" {
		arch = "standard"
	}
	priority, err := parsePriority(q.Get("priority"))
	if err != nil {
		writeError(w, http.StatusBadRequest, codeBadRequest, err.Error())
		return
	}
	var requestedEpochs int
	if kind == jobs.KindTrain {
		if requestedEpochs, err = parseTrainEpochs(q.Get("epochs")); err != nil {
			writeError(w, http.StatusBadRequest, codeBadRequest, err.Error())
			return
		}
	}
	epochs := jobs.NormalizeEpochs(kind, requestedEpochs)

	// Idempotent short-circuit: a known key is answered from its current state
	// without requiring the upload (same key ⇒ same work).
	if existing, ok, err := s.store.GetJob(ctx, key); err != nil {
		s.internal(w, "get job", err)
		return
	} else if ok {
		writeJSON(w, http.StatusOK, s.putState(ctx, existing))
		return
	}

	// A new job needs the resolved profile to recompute the key. No profile means
	// the runtime has never provisioned — the client should retry once ready.
	p := s.profile.Load()
	if p.Nam == "" || p.SignalSHA256 == "" || p.DriverSHA256 == "" {
		writeError(w, http.StatusServiceUnavailable, codeRuntimeUnavailable,
			"training runtime not provisioned yet")
		return
	}

	// Read (bounded) and validate the WAV before touching the DB.
	body, ok := s.readBody(w, r)
	if !ok {
		return // s.readBody already wrote the error
	}
	if _, err := wav.Validate(body); err != nil {
		writeError(w, http.StatusUnprocessableEntity, codeWavInvalid, err.Error())
		return
	}

	// Disk floor (best-effort: a statfs error does not block admission).
	if free, ferr := sysutil.FreeBytes(s.cfg.DataDir); ferr == nil && free < s.cfg.MinFreeBytes() {
		writeError(w, http.StatusInsufficientStorage, codeDiskFull, "insufficient free disk space")
		return
	}

	// Recompute the key; a mismatch is a client bug.
	computed := jobkey.Compute(jobkey.SHA256Hex(body), kind, epochs, arch,
		p.Nam, p.DriverSHA256, p.SignalSHA256)
	if computed != key {
		writeError(w, http.StatusBadRequest, codeKeyMismatch,
			"submitted key does not match the recomputed content key")
		return
	}

	job := jobs.Job{
		Key:       key,
		Kind:      kind,
		State:     jobs.StateQueued,
		Priority:  priority,
		Epochs:    epochs,
		Arch:      arch,
		CreatedAt: s.now().Unix(),
	}
	// Insert, tolerating two rare interleavings: a concurrent identical PUT wins
	// (ErrExists → answer with the winner's state), and that winner is then
	// DELETEd inside the race window (the key is free again → retry rather than
	// return a spurious 500).
	const maxInsertAttempts = 3
	for attempt := 1; ; attempt++ {
		err := s.store.InsertJob(ctx, job, body)
		if err == nil {
			s.log.Printf("job %s accepted: kind=%s epochs=%d arch=%s priority=%d",
				key, kind, epochs, arch, priority)
			if s.notify != nil {
				s.notify() // wake an idle worker
			}
			writeJSON(w, http.StatusCreated, s.putState(ctx, job))
			return
		}
		if errors.Is(err, store.ErrExists) {
			if existing, ok, gerr := s.store.GetJob(ctx, key); gerr == nil && ok {
				writeJSON(w, http.StatusOK, s.putState(ctx, existing))
				return
			}
			if attempt < maxInsertAttempts {
				continue // winner vanished mid-race; the key is free — try again
			}
		}
		s.internal(w, "insert job", err)
		return
	}
}

// handleGetJob reports a job's state, raw progress numbers, and queue position.
func (s *Server) handleGetJob(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	j, ok, err := s.store.GetJob(ctx, r.PathValue("key"))
	if err != nil {
		s.internal(w, "get job", err)
		return
	}
	if !ok {
		writeError(w, http.StatusNotFound, codeNotFound, "no such job")
		return
	}
	writeJSON(w, http.StatusOK, s.jobResponse(ctx, j))
}

// handlePatchJob reorders a QUEUED job. Running/terminal is a documented no-op
// 204; an unknown key is 404. Changing epochs is DELETE + resubmit (a new key).
func (s *Server) handlePatchJob(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	raw := r.URL.Query().Get("priority")
	if raw == "" {
		writeError(w, http.StatusBadRequest, codeBadRequest, "priority required (0..2)")
		return
	}
	priority, err := parsePriority(raw)
	if err != nil {
		writeError(w, http.StatusBadRequest, codeBadRequest, err.Error())
		return
	}
	existed, err := s.store.SetPriorityIfQueued(ctx, r.PathValue("key"), priority)
	if err != nil {
		s.internal(w, "set priority", err)
		return
	}
	if !existed {
		writeError(w, http.StatusNotFound, codeNotFound, "no such job")
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleDeleteJob frees the key. A running job's process group is SIGKILLed first
// so no orphan trainer survives the row deletion.
func (s *Server) handleDeleteJob(w http.ResponseWriter, r *http.Request) {
	// Once we begin killing + deleting we must finish even if the client
	// disconnects, or a killed job could be stranded in `running` forever.
	ctx := context.WithoutCancel(r.Context())
	key := r.PathValue("key")

	// Fast path: atomically drop a still-queued job. If this deletes a row, no
	// process ever existed, and we've closed the window where a worker could
	// claim the job between our lookup and our delete.
	if deleted, err := s.store.DeleteIfQueued(ctx, key); err != nil {
		s.internal(w, "delete if queued", err)
		return
	} else if deleted {
		s.log.Printf("job %s deleted (queued)", key)
		w.WriteHeader(http.StatusNoContent)
		return
	}

	// Not queued: running, terminal, or unknown.
	j, ok, err := s.store.GetJob(ctx, key)
	if err != nil {
		s.internal(w, "get job", err)
		return
	}
	if !ok {
		writeError(w, http.StatusNotFound, codeNotFound, "no such job")
		return
	}
	if j.State == jobs.StateRunning && s.killer != nil {
		s.killer.Kill(key) // SIGKILL the process group before the row disappears
	}
	if _, err := s.store.DeleteJob(ctx, key); err != nil {
		s.internal(w, "delete job", err)
		return
	}
	s.log.Printf("job %s deleted", key)
	w.WriteHeader(http.StatusNoContent)
}

// handleGetModel serves the trained .nam bytes, or 404 until the job succeeds
// (and after the blob is GC'd — the client's cue to DELETE + resubmit).
func (s *Server) handleGetModel(w http.ResponseWriter, r *http.Request) {
	nam, ok, err := s.store.ModelBytes(r.Context(), r.PathValue("key"))
	if err != nil {
		s.internal(w, "get model", err)
		return
	}
	if !ok {
		writeError(w, http.StatusNotFound, codeNotFound, "no model for this key")
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Disposition", `attachment; filename="model.nam"`)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(nam)
}

// handleGetLog serves the job's training stdout, one line per row joined with
// newlines. One code path for live and historical reads.
func (s *Server) handleGetLog(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	key := r.PathValue("key")
	exists, err := s.store.JobExists(ctx, key)
	if err != nil {
		s.internal(w, "job exists", err)
		return
	}
	if !exists {
		writeError(w, http.StatusNotFound, codeNotFound, "no such job")
		return
	}
	lines, err := s.store.JobLog(ctx, key)
	if err != nil {
		s.internal(w, "job log", err)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, strings.Join(lines, "\n"))
}

// ---- helpers ---------------------------------------------------------------

// readBody reads the request body under a hard size cap. It returns ok=false and
// writes the appropriate error (422 for oversize, 400 for a broken stream).
func (s *Server) readBody(w http.ResponseWriter, r *http.Request) ([]byte, bool) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, wav.MaxSizeBytes))
	if err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeError(w, http.StatusUnprocessableEntity, codeWavInvalid,
				"file too large (max 200 MB)")
			return nil, false
		}
		writeError(w, http.StatusBadRequest, codeBadRequest, "error reading request body")
		return nil, false
	}
	return body, true
}

// putState builds the compact PUT/idempotent response for a job.
func (s *Server) putState(ctx context.Context, j jobs.Job) putResponse {
	return putResponse{Key: j.Key, State: j.State, Position: s.positionPtr(ctx, j.Key, j.State)}
}

// jobResponse builds the detailed GET response, computing position and ETA.
func (s *Server) jobResponse(ctx context.Context, j jobs.Job) jobResponse {
	resp := jobResponse{
		Kind:      j.Kind,
		State:     j.State,
		Priority:  j.Priority,
		Position:  s.positionPtr(ctx, j.Key, j.State),
		Epoch:     j.Epoch,
		Epochs:    j.Epochs,
		SPerEpoch: j.SPerEpoch,
		Verdict:   j.Verdict,
		ESR:       j.ESR,
		HasModel:  j.HasModel,
	}
	if j.State == jobs.StateRunning && j.Epoch != nil && j.SPerEpoch != nil {
		remaining := int64(j.Epochs) - (*j.Epoch + 1)
		if remaining < 0 {
			remaining = 0
		}
		eta := int64(math.Ceil(float64(remaining) * *j.SPerEpoch))
		resp.EtaS = &eta
	}
	if j.State == jobs.StateFailed && j.ErrorCode != nil {
		msg := ""
		if j.ErrorMsg != nil {
			msg = *j.ErrorMsg
		}
		resp.Error = &errBody{Code: *j.ErrorCode, Message: msg}
	}
	return resp
}

// positionPtr returns the 1-based queue position for a queued job, or nil for a
// running/terminal job (which has no position).
func (s *Server) positionPtr(ctx context.Context, key, state string) *int {
	if state != jobs.StateQueued {
		return nil
	}
	pos, ok, err := s.store.QueuedPosition(ctx, key)
	if err != nil || !ok {
		return nil
	}
	return &pos
}

// internal logs and writes a 500.
func (s *Server) internal(w http.ResponseWriter, what string, err error) {
	s.log.Printf("ERROR: %s: %v", what, err)
	writeError(w, http.StatusInternalServerError, codeInternal, "internal error")
}

// parsePriority parses the priority query param. Empty ⇒ default 1 (medium);
// otherwise it must be 0, 1, or 2.
func parsePriority(raw string) (int, error) {
	if raw == "" {
		return 1, nil
	}
	p, err := strconv.Atoi(raw)
	if err != nil || p < 0 || p > 2 {
		return 0, errors.New("priority must be 0, 1, or 2")
	}
	return p, nil
}

// parseTrainEpochs parses and bounds the epochs param for a train job.
func parseTrainEpochs(raw string) (int, error) {
	if raw == "" {
		return 0, errors.New("epochs required for a train job")
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 || n > jobs.MaxTrainEpochs {
		return 0, errors.New("epochs must be an integer in 1.." + strconv.Itoa(jobs.MaxTrainEpochs))
	}
	return n, nil
}
