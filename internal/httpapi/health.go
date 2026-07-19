// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package httpapi

import (
	"net/http"

	"orbit-capture-nam-trainer/internal/buildinfo"
	"orbit-capture-nam-trainer/internal/sysutil"
)

// healthResponse is the GET /v1/health body (the design notes).
type healthResponse struct {
	Version       string   `json:"version"`
	Protocol      int      `json:"protocol"`
	Ready         bool     `json:"ready"`
	Python        string   `json:"python"`
	Nam           string   `json:"nam"`
	GPU           string   `json:"gpu"`
	DriverSHA256  string   `json:"driver_sha256"`
	SignalSHA256  string   `json:"signal_sha256"`
	Running       int      `json:"running"`
	Queued        int      `json:"queued"`
	AvgSPerEpoch  *float64 `json:"avg_s_per_epoch"`
	DiskFreeBytes uint64   `json:"disk_free_bytes"`
	Cap           int      `json:"cap"`
}

// handleHealth reports liveness + the trainer profile. It MUST stay O(1)-cheap:
// clients poll it every few seconds to drive a liveness lamp. Counts come from
// in-memory atomics; disk-free is a single statfs.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	p := s.profile.Load()
	free, _ := sysutil.FreeBytes(s.cfg.DataDir) // best-effort; 0 on error, never fails health
	running, queued := s.loadCounts()

	writeJSON(w, http.StatusOK, healthResponse{
		Version:       buildinfo.Version,
		Protocol:      buildinfo.Protocol,
		Ready:         p.Ready,
		Python:        p.Python,
		Nam:           p.Nam,
		GPU:           p.GPU,
		DriverSHA256:  p.DriverSHA256,
		SignalSHA256:  p.SignalSHA256,
		Running:       running,
		Queued:        queued,
		AvgSPerEpoch:  s.avgSPerEpoch.Load(),
		DiskFreeBytes: free,
		Cap:           s.cfg.Cap,
	})
}
