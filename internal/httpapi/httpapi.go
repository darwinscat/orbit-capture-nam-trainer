// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package httpapi serves the v1 HTTP contract (the design notes). Every
// endpoint requires a bearer token, compared constant-time; errors are uniform
// JSON envelopes. This file holds the server wiring, the auth middleware, and the
// shared error/JSON helpers; individual endpoints live in sibling files.
package httpapi

import (
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
	codeNotFound           = "not_found"
	codeBadRequest         = "bad_request"
	codeKeyMismatch        = "key_mismatch"
	codeWavInvalid         = "wav_invalid"
	codeDiskFull           = "disk_full"
	codeRuntimeUnavailable = "runtime_unavailable"
	codeInternal           = "internal"
)

// Profile is the resolved trainer profile surfaced by /v1/health. It is empty
// (Ready=false) until the runtime is provisioned at first run
type Profile struct {
	Ready        bool
	Python       string
	Nam          string
	DriverSHA256 string
	SignalSHA256 string
}

// Killer aborts a running job's process group. The worker implements it;
// it is nil until then, so DELETE simply drops the row.
type Killer interface {
	Kill(key string)
}

// Server holds the daemon's HTTP state. Counters are in-memory atomics so
// /v1/health stays O(1) under a client polling it every few seconds.
type Server struct {
	cfg     *config.Config
	store   *store.Store
	log     *applog.Logger
	profile atomic.Pointer[Profile]
	now     func() time.Time // injectable clock (created_at); real = time.Now
	killer  Killer           // set by the worker; nil in the API-only build
	notify  func()           // wake a worker after a new job is accepted; may be nil

	// counts packs running (high 32 bits) + queued (low 32 bits) into one word so
	// /v1/health never reads a torn running/queued pair from two separate stores.
	counts atomic.Uint64
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

// Handler returns the fully-wired http.Handler (routes + auth middleware).
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/health", s.handleHealth)
	mux.HandleFunc("PUT /v1/jobs/{key}", s.handlePutJob)
	mux.HandleFunc("GET /v1/jobs/{key}", s.handleGetJob)
	mux.HandleFunc("PATCH /v1/jobs/{key}", s.handlePatchJob)
	mux.HandleFunc("DELETE /v1/jobs/{key}", s.handleDeleteJob)
	mux.HandleFunc("GET /v1/jobs/{key}/model", s.handleGetModel)
	mux.HandleFunc("GET /v1/jobs/{key}/log", s.handleGetLog)
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
