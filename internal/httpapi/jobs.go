// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package httpapi

import (
	"context"
	"errors"
	"io"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"orbit-capture-nam-trainer/internal/jobkey"
	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/sysutil"
	"orbit-capture-nam-trainer/internal/wav"
	"orbit-capture-nam-trainer/internal/worker"
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
	Kind       string   `json:"kind"`
	State      string   `json:"state"`
	Priority   int      `json:"priority"`
	Position   *int     `json:"position"`
	Epoch      *int64   `json:"epoch"`
	Epochs     int      `json:"epochs"`
	StartEpoch *int64   `json:"start_epoch"` // train_more: parent's epochs (numbering origin); null otherwise
	Reached    *int64   `json:"reached"`     // train-lane computed-epoch count (== epochs on a natural finish, the stop point on an early stop); null for probes, queued rows, and pre-v3 finishes
	SPerEpoch  *float64 `json:"s_per_epoch"`
	EtaS       *int64   `json:"eta_s"`
	Verdict    *string  `json:"verdict"`
	ESR        *float64 `json:"esr"`
	Error      *errBody `json:"error"`
	HasModel   bool     `json:"has_model"`
}

type errBody struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// stopResponse is the 202 body for POST /stop: the async pause has begun. The
// terminal transition lands later, so the client keeps polling GET until the
// ordinary succeeded pipeline reports it.
type stopResponse struct {
	State string `json:"state"`
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
	// base names a train_more's parent and is part of that kind's key; it is required
	// for train_more and a client bug on any other kind. A malformed/absent/stray base
	// is a clean 400 here — 409 base_unavailable is reserved for a well-formed request
	// the store then rejects on the parent's actual state.
	baseKey, err := parseBaseKey(kind, q.Get("base"))
	if err != nil {
		writeError(w, http.StatusBadRequest, codeBadRequest, err.Error())
		return
	}
	arch := q.Get("arch")
	if arch == "" {
		arch = "standard"
	}
	// arch goes verbatim into the key preimage, so keep it to a tame charset — no
	// newlines or separators a caller could smuggle into the hashed text (plan §2 PUT
	// hygiene; applies to every kind).
	if !archRE.MatchString(arch) {
		writeError(w, http.StatusBadRequest, codeBadRequest, "arch must match [A-Za-z0-9_-]+")
		return
	}
	priority, err := parsePriority(q.Get("priority"))
	if err != nil {
		writeError(w, http.StatusBadRequest, codeBadRequest, err.Error())
		return
	}
	// A train_more carries the same epochs contract as a train (the TOTAL target); the
	// parse gate covers both so a missing/zero epochs is a clean 400 here rather than a
	// misleading key_mismatch after the recompute (crew F5).
	var requestedEpochs int
	if kind == jobs.KindTrain || kind == jobs.KindTrainMore {
		if requestedEpochs, err = parseTrainEpochs(q.Get("epochs")); err != nil {
			writeError(w, http.StatusBadRequest, codeBadRequest, err.Error())
			return
		}
	}
	epochs := jobs.NormalizeEpochs(kind, requestedEpochs)

	// Idempotent short-circuit: a known key is answered from its current state
	// without requiring the upload (same key ⇒ same work). This stays FIRST, before
	// the body read and before any parent lookup: a train_more resubmit is answered
	// from the child's own row and never re-checks its parent — the child is
	// self-contained once inserted (its parent may since have been deleted or GC'd).
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

	// Recompute the key; a mismatch is a client bug. train_more folds the parent key
	// into the preimage (ComputeTrainMore) so the parent is part of the child's
	// identity; every other kind uses the base-free formula unchanged.
	wavHex := jobkey.SHA256Hex(body)
	var computed string
	if kind == jobs.KindTrainMore {
		computed = jobkey.ComputeTrainMore(wavHex, epochs, arch,
			p.Nam, p.DriverSHA256, p.SignalSHA256, baseKey)
	} else {
		computed = jobkey.Compute(wavHex, kind, epochs, arch,
			p.Nam, p.DriverSHA256, p.SignalSHA256)
	}
	if computed != key {
		writeError(w, http.StatusBadRequest, codeKeyMismatch,
			"submitted key does not match the recomputed content key")
		return
	}

	// WavSHA is the hex just computed — InsertJob reuses it instead of re-hashing the
	// multi-MB capture. BaseKey is set only for train_more; it drives the store's
	// parent-snapshot inside the insert transaction.
	job := jobs.Job{
		Key:       key,
		Kind:      kind,
		State:     jobs.StateQueued,
		Priority:  priority,
		Epochs:    epochs,
		Arch:      arch,
		CreatedAt: s.now().Unix(),
		WavSHA:    &wavHex,
	}
	if kind == jobs.KindTrainMore {
		job.BaseKey = &baseKey
	}
	// Insert, tolerating two rare interleavings: a concurrent identical PUT wins
	// (ErrExists → answer with the winner's state), and that winner is then
	// DELETEd inside the race window (the key is free again → retry rather than
	// return a spurious 500).
	const maxInsertAttempts = 3
	for attempt := 1; ; attempt++ {
		err := s.store.InsertJob(ctx, job, body)
		if err == nil {
			s.logAccepted(ctx, job)
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
		// An ineligible parent (unknown / not-succeeded / no-ckpt / wav- or
		// arch-mismatch / epochs≤parent) is a 409 carrying the store's own reason —
		// the remedy is always a fresh kind=train.
		if errors.Is(err, store.ErrBaseUnavailable) {
			reason := "parent job is not available for continuation"
			var bu *store.BaseUnavailableError
			if errors.As(err, &bu) {
				reason = bu.Reason
			}
			writeError(w, http.StatusConflict, codeBaseUnavailable,
				reason+" — submit a fresh kind=train to retrain from scratch")
			return
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
		if s.notify != nil {
			s.notify() // republish queue counts (and refresh the keep-awake assertion)
		}
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
	if s.notify != nil {
		s.notify() // republish queue counts (and refresh the keep-awake assertion)
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleStopJob requests an early stop of a RUNNING train/train_more job
// (POST /v1/jobs/{key}/stop) — registered only when a Stopper is wired (see
// Handler). The run becomes a NORMAL succeeded job whose model + retained
// checkpoint are the last completed epoch's pair; the verb only KICKS OFF that
// transition (202 {"state":"stopping"}) and the terminal state lands async, so the
// client keeps polling GET. The status mapping (the design notes, crew F-3/F4):
//
//   - unknown key                                    → 404 not_found
//   - terminal (idempotent; also a stop that raced a finish) → 204
//   - queued, or a kind not in {train, train_more}   → 409 no_checkpoint
//     Probes NEVER reach StopJob: this kind gate is the ONLY wall (crew F-3 — the
//     worker does not defend probes against stop), so a probe_self/probe_e10 is
//     turned away here without ever touching the Stopper.
//   - running train-lane → StopJob(key):
//     nil             → 202 {"state":"stopping"}
//     ErrNoCheckpoint → 409 no_checkpoint (no completed epoch yet, or the final
//     teardown seconds — re-GET before falling back to DELETE)
//     ErrNotRunning   → re-read the row ONCE (crew F4, the stop-vs-just-finished
//     race: the attempt unregistered between our GetJob and the
//     StopJob call): terminal → 204, else → 409 no_checkpoint
//     (the claim→register window)
//     any other error → 500 internal
func (s *Server) handleStopJob(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	key := r.PathValue("key")

	j, ok, err := s.store.GetJob(ctx, key)
	if err != nil {
		s.internal(w, "get job", err)
		return
	}
	if !ok {
		writeError(w, http.StatusNotFound, codeNotFound, "no such job")
		return
	}
	// Terminal is an idempotent no-op — also the answer when the stop raced a finish.
	if jobs.IsTerminal(j.State) {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	// Queued (nothing running to stop), or a kind that is not stoppable
	// (probe_self/probe_e10): 409 no_checkpoint, and — crucially — the Stopper is
	// NEVER called, so a probe key can never reach StopJob (crew F-3, the sole wall).
	if j.State == jobs.StateQueued || (j.Kind != jobs.KindTrain && j.Kind != jobs.KindTrainMore) {
		writeError(w, http.StatusConflict, codeNoCheckpoint, "no checkpoint to keep yet")
		return
	}

	// Running train-lane job: request the stop.
	switch err := s.stopper.StopJob(key); {
	case err == nil:
		writeJSON(w, http.StatusAccepted, stopResponse{State: "stopping"})
	case errors.Is(err, worker.ErrNoCheckpoint):
		writeError(w, http.StatusConflict, codeNoCheckpoint, "no checkpoint to keep yet")
	case errors.Is(err, worker.ErrNotRunning):
		// The stop-vs-just-finished race (crew F4): the attempt unregistered between
		// our GetJob above and this StopJob call. Re-read once — a terminal row is a
		// 204 no-op (the run just finished); anything else is the claim→register
		// window (or a delete inside the race), a 409.
		j2, ok2, err2 := s.store.GetJob(ctx, key)
		if err2 != nil {
			s.internal(w, "get job", err2)
			return
		}
		if ok2 && jobs.IsTerminal(j2.State) {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		writeError(w, http.StatusConflict, codeNoCheckpoint, "no checkpoint to keep yet")
	default:
		s.internal(w, "stop job", err)
	}
}

// handleGetModel serves the trained .nam bytes, or 404 until the job succeeds
// (and after the blob is GC'd — the client's cue to DELETE + resubmit).
//
// With ?live=1 AND a wired exporter it instead auditions a RUNNING train-lane
// job's best-so-far snapshot (serveLiveModel). Only the exact value "1" and a
// non-nil exporter activate that branch; live=0, live=yes, and the API-only build
// all fall through to the plain path an old daemon would serve. A terminal job on
// the live path also falls through — its finished model IS the live artifact.
func (s *Server) handleGetModel(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	key := r.PathValue("key")
	if r.URL.Query().Get("live") == "1" && s.exporter != nil {
		if s.serveLiveModel(w, ctx, key) {
			return
		}
	}
	s.servePlainModel(w, ctx, key)
}

// serveLiveModel handles GET /model?live=1 for a wired exporter. It returns false
// ONLY for a terminal job — whose finished model is the live artifact — so the
// caller serves the plain path byte-identically (incl. its 404 when the blob is
// absent/GC'd). Every other case writes its own response and returns true:
//
//   - unknown key                      → 404 not_found
//   - queued job, or a probe_self      → 404 no_checkpoint (never calls the exporter)
//   - running (train/train_more/probe_e10) → ExportLive, mapping its sentinels:
//     ErrNoLiveJob     → 404 not_found  (claim→register window / just-requeued;
//     an old daemon answers not_found for that poll too)
//     ErrNoCheckpoint  → 404 no_checkpoint
//     ErrLiveTransient → 500 internal
//     success          → 200 octet-stream + live.nam headers
func (s *Server) serveLiveModel(w http.ResponseWriter, ctx context.Context, key string) (handled bool) {
	j, ok, err := s.store.GetJob(ctx, key)
	if err != nil {
		s.internal(w, "get job", err)
		return true
	}
	if !ok {
		writeError(w, http.StatusNotFound, codeNotFound, "no such job")
		return true
	}
	// A finished run: the terminal model is the live artifact — hand back to the
	// plain path for byte-identical bytes (and its own GC'd 404).
	if jobs.IsTerminal(j.State) {
		return false
	}
	// Not checkpointing (yet): a queued job of any kind, or a probe_self, which is
	// killed on verdict before epoch 0 and never writes a checkpoint. Answer
	// without touching the exporter.
	if j.State == jobs.StateQueued || j.Kind == jobs.KindProbeSelf {
		writeError(w, http.StatusNotFound, codeNoCheckpoint, "no live snapshot available yet")
		return true
	}
	// Running train-lane job: audition its best-so-far snapshot.
	nam, epoch, esr, err := s.exporter.ExportLive(ctx, key)
	switch {
	case errors.Is(err, worker.ErrNoLiveJob):
		writeError(w, http.StatusNotFound, codeNotFound, "no model for this key")
	case errors.Is(err, worker.ErrNoCheckpoint):
		writeError(w, http.StatusNotFound, codeNoCheckpoint, "no live snapshot available yet")
	case errors.Is(err, worker.ErrLiveTransient):
		writeError(w, http.StatusInternalServerError, codeInternal, "live snapshot read failed; retry")
	case err != nil:
		s.internal(w, "export live", err)
	default:
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Content-Disposition", `attachment; filename="live.nam"`)
		w.Header().Set("X-Live-Epoch", strconv.FormatInt(epoch, 10))
		// Fixed-point %.8f — NEVER scientific: the app parses X-Live-Esr the same
		// way it parses the driver's esr lines, so 3.5e-05 must render 0.00003500.
		w.Header().Set("X-Live-Esr", strconv.FormatFloat(esr, 'f', 8, 64))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(nam)
	}
	return true
}

// servePlainModel serves the stored .nam bytes, or 404 (no model for this key).
// This is the historical /model path; the live=1 branch falls back to it for a
// terminal job and whenever no exporter is wired.
func (s *Server) servePlainModel(w http.ResponseWriter, ctx context.Context, key string) {
	nam, ok, err := s.store.ModelBytes(ctx, key)
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

// logAccepted writes the accepted-job story-log line. A train_more also records its
// parent (8-hex prefix) and the epoch its numbering resumes at (the parent's epochs,
// read back from the freshly-inserted row — the store, not the client, decided it).
func (s *Server) logAccepted(ctx context.Context, j jobs.Job) {
	if j.Kind != jobs.KindTrainMore {
		s.log.Printf("job %s accepted: kind=%s epochs=%d arch=%s priority=%d",
			j.Key, j.Kind, j.Epochs, j.Arch, j.Priority)
		return
	}
	base := ""
	if j.BaseKey != nil {
		base = shortKey(*j.BaseKey)
	}
	start := int64(-1)
	if fetched, ok, err := s.store.GetJob(ctx, j.Key); err == nil && ok && fetched.StartEpoch != nil {
		start = *fetched.StartEpoch
	}
	s.log.Printf("job %s accepted: kind=%s base=%s start_epoch=%d epochs=%d arch=%s priority=%d",
		j.Key, j.Kind, base, start, j.Epochs, j.Arch, j.Priority)
}

// jobResponse builds the detailed GET response, computing position and ETA.
func (s *Server) jobResponse(ctx context.Context, j jobs.Job) jobResponse {
	resp := jobBody(j)
	resp.Position = s.positionPtr(ctx, j.Key, j.State)
	return resp
}

// jobBody builds the per-job response fields that derive purely from the row —
// everything except the queue position, which needs a store lookup. Shared by the
// single-job GET (which adds position via positionPtr) and the /v1/queue batch
// (which fills position from its one snapshot).
func jobBody(j jobs.Job) jobResponse {
	resp := jobResponse{
		Kind:       j.Kind,
		State:      j.State,
		Priority:   j.Priority,
		Epoch:      j.Epoch,
		Epochs:     j.Epochs,
		StartEpoch: j.StartEpoch,
		Reached:    j.Reached,
		SPerEpoch:  j.SPerEpoch,
		Verdict:    j.Verdict,
		ESR:        j.ESR,
		HasModel:   j.HasModel,
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

// parseTrainEpochs parses and bounds the epochs param. It gates both train and
// train_more (which share the TOTAL-target epochs contract), so the message stays
// kind-neutral.
func parseTrainEpochs(raw string) (int, error) {
	if raw == "" {
		return 0, errors.New("epochs required")
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 || n > jobs.MaxTrainEpochs {
		return 0, errors.New("epochs must be an integer in 1.." + strconv.Itoa(jobs.MaxTrainEpochs))
	}
	return n, nil
}

// archRE bounds the arch param to a tame charset — arch goes verbatim into the key
// preimage, so free-form text there is a latent hazard (plan §2 PUT hygiene).
var archRE = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

// parseBaseKey validates the base query param against the kind. A train_more MUST
// carry a 64-char lower-case hex parent key (it is part of the child's identity); any
// other kind must NOT carry one (a stray base is a client bug). The returned key is
// empty for the non-train_more kinds.
func parseBaseKey(kind, raw string) (string, error) {
	if kind != jobs.KindTrainMore {
		if raw != "" {
			return "", errors.New("base is only valid for kind=train_more")
		}
		return "", nil
	}
	if !isHex64(raw) {
		return "", errors.New("base must be a 64-character lower-case hex parent key")
	}
	return raw, nil
}

// isHex64 reports whether s is exactly 64 lower-case hex characters — the shape of a
// job key.
func isHex64(s string) bool {
	if len(s) != 64 {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

// shortKey returns the first 8 characters of a job key for story-log lines (a full
// 64-hex key is noise in the log).
func shortKey(k string) string {
	if len(k) > 8 {
		return k[:8]
	}
	return k
}
