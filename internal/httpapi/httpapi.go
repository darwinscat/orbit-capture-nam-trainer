// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package httpapi serves the v1 HTTP contract (the design notes). Every
// endpoint requires a bearer token, compared constant-time; errors are uniform
// JSON envelopes. This file holds the server wiring, the auth middleware, and the
// shared error/JSON helpers; individual endpoints live in sibling files.
package httpapi

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"orbit-capture-nam-trainer/internal/applog"
	"orbit-capture-nam-trainer/internal/config"
	"orbit-capture-nam-trainer/internal/store"
)

// Error codes (the design notes).
const (
	codeUnauthorized       = "unauthorized"
	codeForbidden          = "forbidden"
	codeNotFound           = "not_found"
	codeBadRequest         = "bad_request"
	codeKeyMismatch        = "key_mismatch"
	codeWavInvalid         = "wav_invalid"
	codeBaseUnavailable    = "base_unavailable"
	codeDiskFull           = "disk_full"
	codeRuntimeUnavailable = "runtime_unavailable"
	codeInternal           = "internal"
	// codeNoCheckpoint is the additive live-export 404: a running train-lane job
	// (or a queued one) with no best-so-far snapshot to audition yet — before the
	// first completed epoch, a probe_self (never checkpoints), or the run's final
	// teardown seconds. Distinct from not_found so a client can tell "retry, a
	// snapshot is coming" from "this key has no live export / is gone".
	codeNoCheckpoint = "no_checkpoint"
)

// Profile is the resolved trainer profile surfaced by /v1/health. It is empty
// (Ready=false) until the runtime is provisioned at first run
type Profile struct {
	Ready        bool
	Python       string
	Nam          string
	GPU          string // training device: "mps" | "cuda" | "cpu"
	DriverSHA256 string
	SignalSHA256 string
}

// Killer aborts a running job's process group. The worker implements it;
// it is nil until then, so DELETE simply drops the row.
type Killer interface {
	Kill(key string)
}

// Stopper requests an EARLY STOP of a running train-lane job (POST /stop): the run
// becomes a NORMAL succeeded job whose model + retained checkpoint are the last
// completed epoch's pair. The worker's Pool implements it; it is nil in the API-only
// build, and a nil Stopper leaves the /stop route UNREGISTERED (Handler answers the
// plain-text 404 an old daemon would). StopJob is the request side only — it kills
// the group and returns; the terminal transition lands asynchronously, so the handler
// answers 202 {"state":"stopping"}. It returns nil (kill issued), worker.ErrNoCheckpoint
// (no completed epoch to keep yet → 409), or worker.ErrNotRunning (no registered
// attempt → the handler re-reads the row once, crew F4).
type Stopper interface {
	StopJob(key string) error
}

// Capper adjusts and reports the LIVE training-lane width. The worker
// implements it; nil until wired (PATCH /v1/cap then answers 503).
type Capper interface {
	SetCap(n int)
	Cap() int
}

// LiveExporter serves the best-so-far checkpoint snapshot of a RUNNING
// train-lane job — the GET /model?live=1 audition path. The worker's Pool
// implements it; it is nil until wired, and a nil exporter makes live=1 a no-op
// (the plain /model an old daemon would serve). ExportLive returns the snapshot
// .nam bytes, its absolute epoch, its validation ESR, and one of the worker
// sentinels (ErrNoLiveJob / ErrNoCheckpoint / ErrLiveTransient) the handler maps.
type LiveExporter interface {
	ExportLive(ctx context.Context, key string) (nam []byte, epoch int64, esr float64, err error)
}

// Server holds the daemon's HTTP state. Counters are in-memory atomics so
// /v1/health stays O(1) under a client polling it every few seconds.
type Server struct {
	cfg      *config.Config
	store    *store.Store
	log      *applog.Logger
	profile  atomic.Pointer[Profile]
	now      func() time.Time // injectable clock (created_at); real = time.Now
	killer   Killer           // set by the worker; nil in the API-only build
	stopper  Stopper          // early-stop request path; nil in the API-only build (route unregistered)
	capper   Capper           // live train-lane width; nil in the API-only build
	exporter LiveExporter     // live snapshot export; nil in the API-only build
	notify   func()           // wake a worker after a new job is accepted; may be nil

	// counts packs running (high 32 bits) + queued (low 32 bits) into one word so
	// /v1/health never reads a torn running/queued pair from two separate stores.
	counts atomic.Uint64

	// avgSPerEpoch is the cached moving-average seconds/epoch (nil until there is
	// train history). The worker recomputes it on job transitions.
	avgSPerEpoch atomic.Pointer[float64]

	// apiCapAllowed gates PATCH /v1/cap: false (the default) keeps cap
	// admin-only — config file or the tray toggle; a client gets 403. Runtime-
	// flippable from the tray, persisted alongside cap.
	apiCapAllowed atomic.Bool
}

// New builds a Server. The profile starts empty (ready:false).
func New(cfg *config.Config, st *store.Store, lg *applog.Logger) *Server {
	s := &Server{cfg: cfg, store: st, log: lg, now: time.Now}
	s.profile.Store(&Profile{})
	return s
}

// SetClock overrides the wall clock (tests only).
func (s *Server) SetClock(now func() time.Time) { s.now = now }

// SetKiller wires the process-group killer used by DELETE on a running job.
func (s *Server) SetKiller(k Killer) { s.killer = k }

// SetStopper wires the early-stop request path used by POST /v1/jobs/{key}/stop.
// Nil (the default) leaves the route UNREGISTERED, so the mux answers the plain-text
// 404 an old daemon would. FRAGILE: Handler() reads s.stopper ONCE, when it builds
// the mux, so SetStopper MUST run before Handler() — a call after it silently loses
// the route. (main wires it right after SetKiller, well before Handler.)
func (s *Server) SetStopper(st Stopper) { s.stopper = st }

// SetCapper wires the live train-lane width control used by PATCH /v1/cap and
// reported by /v1/health.
func (s *Server) SetCapper(c Capper) { s.capper = c }

// SetLiveExporter wires the best-so-far snapshot exporter used by
// GET /v1/jobs/{key}/model?live=1. Nil (the default) makes live=1 a no-op.
func (s *Server) SetLiveExporter(e LiveExporter) { s.exporter = e }

// SetAPICapAllowed flips the PATCH /v1/cap permission gate (initially the
// config's allow_api_cap; the tray toggle updates it live).
func (s *Server) SetAPICapAllowed(v bool) { s.apiCapAllowed.Store(v) }

// APICapAllowed reports the live gate (the tray checkbox mirrors it).
func (s *Server) APICapAllowed() bool { return s.apiCapAllowed.Load() }

// liveCap is the cap /v1/health reports: the pool's live value once wired,
// else the configured one.
func (s *Server) liveCap() int {
	if s.capper != nil {
		return s.capper.Cap()
	}
	return s.cfg.Cap
}

// SetNotifier wires a callback fired when a new job is accepted, so an idle
// worker wakes immediately instead of waiting for its poll tick.
func (s *Server) SetNotifier(fn func()) { s.notify = fn }

// SetProfile publishes a new trainer profile (called after provisioning).
func (s *Server) SetProfile(p Profile) { s.profile.Store(&p) }

// SetCounts publishes the live running/queued counts for /v1/health atomically.
func (s *Server) SetCounts(running, queued int) {
	s.counts.Store(uint64(uint32(running))<<32 | uint64(uint32(queued)))
}

// loadCounts unpacks the running/queued counts.
func (s *Server) loadCounts() (running, queued int) {
	c := s.counts.Load()
	return int(uint32(c >> 32)), int(uint32(c))
}

// SetAvgSPerEpoch publishes the moving-average seconds/epoch for /v1/health
// (nil = no train history yet).
func (s *Server) SetAvgSPerEpoch(v *float64) {
	if v == nil {
		s.avgSPerEpoch.Store(nil)
		return
	}
	c := *v
	s.avgSPerEpoch.Store(&c)
}

// Handler returns the fully-wired http.Handler (routes + auth middleware).
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/health", s.handleHealth)
	mux.HandleFunc("PUT /v1/jobs/{key}", s.handlePutJob)
	mux.HandleFunc("GET /v1/jobs/{key}", s.handleGetJob)
	mux.HandleFunc("PATCH /v1/jobs/{key}", s.handlePatchJob)
	mux.HandleFunc("DELETE /v1/jobs/{key}", s.handleDeleteJob)
	// POST /v1/jobs/{key}/stop is registered ONLY when an early-stop Stopper is wired
	// (the worker's Pool). Without one — the API-only build, or an old daemon — the
	// route is absent and the mux answers a plain-text 404, the same "server too old"
	// cue the client falls back on (as with /v1/queue). It is a DISTINCT path from
	// GET/PUT/PATCH/DELETE /v1/jobs/{key} (the Go 1.22 mux keys on the full pattern:
	// the {key} wildcard matches one segment, so ".../stop" never collides), so its
	// presence or absence never perturbs those methods' 405 handling. FRAGILE: this
	// reads s.stopper once — SetStopper after Handler() would silently drop the route.
	if s.stopper != nil {
		mux.HandleFunc("POST /v1/jobs/{key}/stop", s.handleStopJob)
	}
	mux.HandleFunc("GET /v1/jobs/{key}/model", s.handleGetModel)
	mux.HandleFunc("GET /v1/jobs/{key}/log", s.handleGetLog)
	mux.HandleFunc("POST /v1/queue", s.handleQueue)
	mux.HandleFunc("PATCH /v1/cap", s.handlePatchCap)
	return s.auth(mux)
}

// auth enforces the bearer token on every request and logs each failure.
func (s *Server) auth(next http.Handler) http.Handler {
	want := []byte(s.cfg.Token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got, ok := bearer(r)
		if !ok || subtle.ConstantTimeCompare([]byte(got), want) != 1 {
			s.log.Printf("auth failed: %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)
			writeError(w, http.StatusUnauthorized, codeUnauthorized, "missing or invalid bearer token")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// bearer extracts the token from an "Authorization: Bearer <token>" header.
func bearer(r *http.Request) (string, bool) {
	h := r.Header.Get("Authorization")
	const prefix = "Bearer "
	if len(h) <= len(prefix) || !strings.EqualFold(h[:len(prefix)], prefix) {
		return "", false
	}
	return strings.TrimSpace(h[len(prefix):]), true
}

// errorEnvelope is the uniform error body: {"error":{"code":..,"message":..}}.
type errorEnvelope struct {
	Error struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	var e errorEnvelope
	e.Error.Code = code
	e.Error.Message = message
	writeJSON(w, status, e)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
