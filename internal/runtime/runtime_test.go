// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package runtime

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestEnsureSignalDownloadsThenCaches(t *testing.T) {
	body := []byte("pretend v3_0_0.wav bytes")
	want := sha256hex(body)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	path := filepath.Join(t.TempDir(), "runtime", SignalName)

	// First call downloads and verifies.
	if err := ensureSignalFrom(context.Background(), path, srv.URL, want, func(string) {}); err != nil {
		t.Fatalf("ensureSignalFrom (download): %v", err)
	}
	if got, _ := os.ReadFile(path); string(got) != string(body) {
		t.Errorf("downloaded signal bytes mismatch")
	}
	// Second call is a cache hit — no re-download.
	if err := ensureSignalFrom(context.Background(), path, srv.URL, want, func(string) {}); err != nil {
		t.Fatalf("ensureSignalFrom (cached): %v", err)
	}
	if hits.Load() != 1 {
		t.Errorf("server hit %d times, want 1 (second call must be a cache hit)", hits.Load())
	}
}

func TestEnsureSignalRejectsWrongSHA(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("wrong bytes"))
	}))
	defer srv.Close()
	path := filepath.Join(t.TempDir(), SignalName)
	err := ensureSignalFrom(context.Background(), path, srv.URL,
		"0000000000000000000000000000000000000000000000000000000000000000", func(string) {})
	if err == nil {
		t.Fatal("expected a sha-mismatch error")
	}
	if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
		t.Error("a signal that fails sha verification must not be installed")
	}
}

func TestFileHasSHA(t *testing.T) {
	body := []byte("abc")
	p := filepath.Join(t.TempDir(), "f")
	_ = os.WriteFile(p, body, 0o644)
	if !fileHasSHA(p, sha256hex(body)) {
		t.Error("fileHasSHA should match the file's real sha")
	}
	if fileHasSHA(p, "deadbeef") {
		t.Error("fileHasSHA should reject a wrong sha")
	}
	if fileHasSHA(filepath.Join(t.TempDir(), "missing"), sha256hex(body)) {
		t.Error("fileHasSHA should be false for a missing file")
	}
}

func TestDriverIsVendoredAndHashed(t *testing.T) {
	if len(DriverBytes()) == 0 {
		t.Fatal("driver not embedded")
	}
	if !strings.Contains(string(DriverBytes()), "trainer_driver.py") {
		t.Error("embedded driver doesn't look like trainer_driver.py")
	}
	// The sha feeds the job key; it must be the exact vendored bytes.
	if got := DriverSHA256(); got != sha256hex(DriverBytes()) || len(got) != 64 {
		t.Errorf("DriverSHA256 = %s", got)
	}
}

func TestDownloadVerify(t *testing.T) {
	body := []byte("pretend python archive bytes")
	want := sha256hex(body)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()
	noop := func(string) {}

	// Correct sha → installed.
	dest := filepath.Join(t.TempDir(), "archive.tar.gz")
	if err := downloadVerify(context.Background(), srv.URL, dest, want, noop); err != nil {
		t.Fatalf("downloadVerify (good): %v", err)
	}
	got, _ := os.ReadFile(dest)
	if string(got) != string(body) {
		t.Errorf("downloaded bytes mismatch")
	}

	// Wrong sha → error, nothing installed, no .part left behind.
	dest2 := filepath.Join(t.TempDir(), "archive2.tar.gz")
	err := downloadVerify(context.Background(), srv.URL, dest2,
		"0000000000000000000000000000000000000000000000000000000000000000", noop)
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Errorf("expected sha mismatch error, got %v", err)
	}
	if _, err := os.Stat(dest2); !os.IsNotExist(err) {
		t.Error("bad download should not be installed")
	}
	if _, err := os.Stat(dest2 + ".part"); !os.IsNotExist(err) {
		t.Error(".part file should be cleaned up on mismatch")
	}
}

func TestDownloadVerifyStallWatchdog(t *testing.T) {
	old := downloadStall
	downloadStall = 150 * time.Millisecond
	defer func() { downloadStall = old }()

	// Server sends 200 + a byte, then hangs — a mid-body stall.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("x"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done() // block until the client gives up
	}))
	defer srv.Close()

	start := time.Now()
	err := downloadVerify(context.Background(), srv.URL,
		filepath.Join(t.TempDir(), "a"), "deadbeef", func(string) {})
	if err == nil || !strings.Contains(err.Error(), "stall") {
		t.Fatalf("want a stall error, got %v", err)
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Errorf("stall took %s — watchdog did not fire promptly", elapsed)
	}
	// No .part left behind.
	if _, err := os.Stat(filepath.Join(t.TempDir(), "a.part")); !os.IsNotExist(err) {
		t.Error(".part should be cleaned up after a stall")
	}
}
