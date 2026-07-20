// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package httpapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
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

func TestHealthAvgSPerEpochAndGPU(t *testing.T) {
	srv, token := newTestServer(t)
	h := srv.Handler()
	var got healthResponse

	// Defaults: avg is null, gpu empty. Assert on the raw body too — a decode into
	// *float64 can't tell null from an absent field, but the app relies on the key
	// being present and null, so pin the wire form.
	rec := do(t, h, http.MethodGet, "/v1/health", token, nil)
	if body := rec.Body.String(); !strings.Contains(body, `"avg_s_per_epoch":null`) {
		t.Errorf("default body must encode avg_s_per_epoch as null, got %s", body)
	}
	mustJSON(t, rec, &got)
	if got.AvgSPerEpoch != nil {
		t.Errorf("avg_s_per_epoch = %v, want null by default", got.AvgSPerEpoch)
	}
	if got.GPU != "" {
		t.Errorf("gpu = %q, want empty by default", got.GPU)
	}

	// Published avg + gpu-in-profile show up.
	v := 7.5
	srv.SetAvgSPerEpoch(&v)
	srv.SetProfile(Profile{Ready: true, GPU: "mps"})
	mustJSON(t, do(t, h, http.MethodGet, "/v1/health", token, nil), &got)
	if got.AvgSPerEpoch == nil || *got.AvgSPerEpoch != 7.5 {
		t.Errorf("avg_s_per_epoch = %v, want 7.5", got.AvgSPerEpoch)
	}
	if got.GPU != "mps" {
		t.Errorf("gpu = %q, want mps", got.GPU)
	}

	// Publishing nil resets it to null.
	srv.SetAvgSPerEpoch(nil)
	mustJSON(t, do(t, h, http.MethodGet, "/v1/health", token, nil), &got)
	if got.AvgSPerEpoch != nil {
		t.Errorf("avg_s_per_epoch after nil = %v, want null", got.AvgSPerEpoch)
	}
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

// fakeCapper records SetCap calls and serves Cap().
type fakeCapper struct{ n int }

func (f *fakeCapper) SetCap(n int) { f.n = n }
func (f *fakeCapper) Cap() int     { return f.n }

func TestPatchCapForbiddenByDefault(t *testing.T) {
	srv, token := newTestServer(t)
	srv.SetCapper(&fakeCapper{n: 1})
	h := srv.Handler()

	req := httptest.NewRequest(http.MethodPatch, "/v1/cap?cap=3", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403 (admin-only by default)", rr.Code)
	}
	var env errorEnvelope
	if err := json.Unmarshal(rr.Body.Bytes(), &env); err != nil {
		t.Fatalf("error decode: %v", err)
	}
	if env.Error.Code != "forbidden" {
		t.Errorf("error code = %q, want forbidden", env.Error.Code)
	}
	if !strings.Contains(env.Error.Message, "admin") {
		t.Errorf("message %q should point at the admin", env.Error.Message)
	}
}

func TestPatchCapAppliesPersistsAndReportsLive(t *testing.T) {
	srv, token := newTestServer(t)
	capper := &fakeCapper{n: 1}
	srv.SetCapper(capper)
	srv.SetAPICapAllowed(true)
	h := srv.Handler()

	req := httptest.NewRequest(http.MethodPatch, "/v1/cap?cap=3", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want 204 (%s)", rr.Code, rr.Body.String())
	}
	if capper.n != 3 {
		t.Errorf("pool cap = %d, want 3", capper.n)
	}

	// Persisted: a reload of the same base dir sees cap=3, token intact.
	re, err := config.Load(srv.cfg.BaseDir())
	if err != nil {
		t.Fatalf("reload config: %v", err)
	}
	if re.Cap != 3 {
		t.Errorf("persisted cap = %d, want 3", re.Cap)
	}
	if re.Token != srv.cfg.Token {
		t.Error("token changed across cap persist")
	}
	if !re.AllowAPICap {
		t.Error("persisting cap reverted allow_api_cap to the boot value")
	}

	// /v1/health reports the LIVE value, not the boot-time one.
	req = httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	var health healthResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &health); err != nil {
		t.Fatalf("health decode: %v", err)
	}
	if health.Cap != 3 {
		t.Errorf("health cap = %d, want live 3", health.Cap)
	}
}

func TestPatchCapValidation(t *testing.T) {
	srv, token := newTestServer(t)
	srv.SetCapper(&fakeCapper{n: 1})
	srv.SetAPICapAllowed(true)
	h := srv.Handler()
	for _, bad := range []string{"", "0", "9", "x", "-1", "2.5"} {
		req := httptest.NewRequest(http.MethodPatch, "/v1/cap?cap="+bad, nil)
		req.Header.Set("Authorization", "Bearer "+token)
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, req)
		if rr.Code != http.StatusBadRequest {
			t.Errorf("cap=%q: status = %d, want 400", bad, rr.Code)
		}
	}
	// No token → 401; capper unwired → 503.
	req := httptest.NewRequest(http.MethodPatch, "/v1/cap?cap=2", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("no token: status = %d, want 401", rr.Code)
	}
	unwired, token2 := newTestServer(t)
	unwired.SetAPICapAllowed(true)
	req = httptest.NewRequest(http.MethodPatch, "/v1/cap?cap=2", nil)
	req.Header.Set("Authorization", "Bearer "+token2)
	rr = httptest.NewRecorder()
	unwired.Handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusServiceUnavailable {
		t.Errorf("unwired: status = %d, want 503", rr.Code)
	}
}
