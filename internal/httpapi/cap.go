// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package httpapi

import (
	"net/http"
	"strconv"

	"orbit-capture-nam-trainer/internal/config"
)

// handlePatchCap sets the live training-lane cap: PATCH /v1/cap?cap=N → 204.
// Gated by allow_api_cap (default OFF): cap is admin-only — the config file or
// the tray — until the admin opens the API, so a remote client cannot resize
// someone else's training machine; refused → 403 `forbidden`. When allowed it
// applies immediately — raising wakes idle workers, lowering takes effect as
// running jobs finish (nothing is killed) — and persists to config.toml so a
// restart keeps it. Outside 1..config.MaxCap → 400. Additive endpoint:
// protocol stays 1, an old daemon 404s it. The next /v1/health reflects the
// new value.
func (s *Server) handlePatchCap(w http.ResponseWriter, r *http.Request) {
	if !s.apiCapAllowed.Load() {
		writeError(w, http.StatusForbidden, codeForbidden,
			"changing cap over the API is disabled on this daemon — ask its admin to allow it (allow_api_cap in config.toml, or the menu-bar toggle)")
		return
	}
	n, err := strconv.Atoi(r.URL.Query().Get("cap"))
	if err != nil || n < 1 || n > config.MaxCap {
		writeError(w, http.StatusBadRequest, codeBadRequest,
			"cap must be an integer in 1.."+strconv.Itoa(config.MaxCap))
		return
	}
	if s.capper == nil {
		writeError(w, http.StatusServiceUnavailable, codeRuntimeUnavailable, "worker pool not available")
		return
	}
	s.capper.SetCap(n)
	// Persist a COPY so the boot-time cfg the rest of the daemon reads stays
	// immutable; both runtime-mutable fields are written from their LIVE state
	// so persisting one never reverts the other. A failed write keeps the live
	// value (already applied) and is only logged: the daemon must not report
	// failure for work it has done.
	updated := *s.cfg
	updated.Cap = s.capper.Cap() // the LIVE value, so a concurrent setter's win is never overwritten
	updated.AllowAPICap = s.apiCapAllowed.Load()
	if err := updated.Save(); err != nil {
		s.log.Printf("cap: persist cap=%d: %v", n, err)
	}
	s.log.Printf("cap set to %d via API from %s", n, r.RemoteAddr)
	w.WriteHeader(http.StatusNoContent)
}
