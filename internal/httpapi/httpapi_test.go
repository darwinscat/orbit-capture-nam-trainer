// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package httpapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"orbit-capture-nam-trainer/internal/applog"
	"orbit-capture-nam-trainer/internal/buildinfo"
	"orbit-capture-nam-trainer/internal/config"
	"orbit-capture-nam-trainer/internal/store"
)

func newTestServer(t *testing.T) (*Server, string) {
	t.Helper()
	base := t.TempDir()
	cfg, err := config.Load(base)
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	st, err := store.Open(context.Background(), cfg.DBPath())
	if err != nil {
		t.Fatalf("store.Open: %v", err)
	}
	t.Cleanup(func() { st.Close() })
	lg, err := applog.Open(filepath.Join(base, "logs", "trainer.log"))
	if err != nil {
		t.Fatalf("applog.Open: %v", err)
	}
	t.Cleanup(func() { lg.Close() })
	return New(cfg, st, lg), cfg.Token
}

func TestHealthRequiresAuth(t *testing.T) {
	srv, token := newTestServer(t)
	h := srv.Handler()

	cases := []struct {
		name       string
		authHeader string
		wantStatus int
	}{
		{"no header", "", http.StatusUnauthorized},
		{"wrong token", "Bearer deadbeef", http.StatusUnauthorized},
		{"not bearer", "Basic " + token, http.StatusUnauthorized},
		{"correct token", "Bearer " + token, http.StatusOK},
		{"case-insensitive scheme", "bearer " + token, http.StatusOK},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
			if tc.authHeader != "" {
				req.Header.Set("Authorization", tc.authHeader)
			}
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != tc.wantStatus {
				t.Errorf("status = %d, want %d (body: %s)", rec.Code, tc.wantStatus, rec.Body.String())
			}
			if tc.wantStatus == http.StatusUnauthorized {
				var e errorEnvelope
				if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil {
					t.Fatalf("decode error body: %v", err)
				}
				if e.Error.Code != codeUnauthorized {
					t.Errorf("error code = %q, want %q", e.Error.Code, codeUnauthorized)
				}
			}
		})
	}
}

func TestHealthShape(t *testing.T) {
	srv, token := newTestServer(t)
	h := srv.Handler()

	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var got healthResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode health: %v (body: %s)", err, rec.Body.String())
	}
	if got.Version != buildinfo.Version {
		t.Errorf("version = %q, want %q", got.Version, buildinfo.Version)
	}
	if got.Protocol != buildinfo.Protocol {
		t.Errorf("protocol = %d, want %d", got.Protocol, buildinfo.Protocol)
	}
	if got.Ready {
		t.Errorf("ready = true, want false before provisioning")
	}
	if got.Cap != config.DefaultCap {
		t.Errorf("cap = %d, want %d", got.Cap, config.DefaultCap)
	}
	if got.Running != 0 || got.Queued != 0 {
		t.Errorf("running/queued = %d/%d, want 0/0", got.Running, got.Queued)
	}
}

func TestHealthReflectsProfileAndCounts(t *testing.T) {
	srv, token := newTestServer(t)
	srv.SetProfile(Profile{Ready: true, Python: "3.12.13", Nam: "0.13.0"})
	srv.SetCounts(2, 5)
	h := srv.Handler()

	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	var got healthResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !got.Ready || got.Python != "3.12.13" || got.Nam != "0.13.0" {
		t.Errorf("profile not reflected: %+v", got)
	}
	if got.Running != 2 || got.Queued != 5 {
		t.Errorf("counts = %d/%d, want 2/5", got.Running, got.Queued)
	}
}
