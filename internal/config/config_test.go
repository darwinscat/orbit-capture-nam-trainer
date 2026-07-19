// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package config

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestLoadCreatesConfigWithTokenAnd0600(t *testing.T) {
	base := t.TempDir()
	c, err := Load(base)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if c.Port != DefaultPort {
		t.Errorf("port = %d, want %d", c.Port, DefaultPort)
	}
	if c.Bind != DefaultBind {
		t.Errorf("bind = %q, want %q", c.Bind, DefaultBind)
	}
	if c.Cap != DefaultCap {
		t.Errorf("cap = %d, want %d", c.Cap, DefaultCap)
	}
	if c.RetentionDays != DefaultRetentionDays {
		t.Errorf("retention_days = %d, want %d", c.RetentionDays, DefaultRetentionDays)
	}
	if len(c.Token) != 64 { // 32 random bytes hex-encoded
		t.Errorf("token length = %d, want 64", len(c.Token))
	}
	if c.DataDir != filepath.Join(base, "data") {
		t.Errorf("data_dir = %q, want %q", c.DataDir, filepath.Join(base, "data"))
	}

	// The config file must be created at mode 0600 (token is a secret).
	st, err := os.Stat(c.ConfigPath())
	if err != nil {
		t.Fatalf("stat config: %v", err)
	}
	if runtime.GOOS != "windows" && st.Mode().Perm() != 0o600 {
		t.Errorf("config mode = %v, want 0600", st.Mode().Perm())
	}

	// data_dir and logs dir must exist.
	if _, err := os.Stat(c.DataDir); err != nil {
		t.Errorf("data_dir not created: %v", err)
	}
	if _, err := os.Stat(filepath.Dir(c.LogPath())); err != nil {
		t.Errorf("logs dir not created: %v", err)
	}
}

func TestLoadIsIdempotentAndKeepsToken(t *testing.T) {
	base := t.TempDir()
	first, err := Load(base)
	if err != nil {
		t.Fatalf("first Load: %v", err)
	}
	second, err := Load(base)
	if err != nil {
		t.Fatalf("second Load: %v", err)
	}
	if first.Token != second.Token {
		t.Errorf("token changed across loads: %q != %q", first.Token, second.Token)
	}
}

func TestLoadRepairsMissingToken(t *testing.T) {
	base := t.TempDir()
	// A config with no token at all.
	must(t, os.WriteFile(filepath.Join(base, "config.toml"),
		[]byte("port = 9000\n"), 0o600))
	c, err := Load(base)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(c.Token) != 64 {
		t.Errorf("token not repaired: len=%d", len(c.Token))
	}
	// The repair must persist.
	c2, err := Load(base)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if c2.Token != c.Token {
		t.Errorf("repaired token not persisted")
	}
	if c.Port != 9000 {
		t.Errorf("edited port lost: got %d", c.Port)
	}
}

func TestNormalizeClampsCap(t *testing.T) {
	base := t.TempDir()
	must(t, os.WriteFile(filepath.Join(base, "config.toml"),
		[]byte("token=\"x\"\ncap = 99\n"), 0o600))
	c, err := Load(base)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Cap != MaxCap {
		t.Errorf("cap = %d, want clamped to %d", c.Cap, MaxCap)
	}
}

func TestDefaultBaseDirHonorsEnv(t *testing.T) {
	want := filepath.Join(t.TempDir(), "custom-base")
	t.Setenv("ONCT_BASE_DIR", want)
	got, err := DefaultBaseDir()
	if err != nil {
		t.Fatalf("DefaultBaseDir: %v", err)
	}
	if got != want {
		t.Errorf("base dir = %q, want %q", got, want)
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
