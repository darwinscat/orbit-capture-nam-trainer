// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package config

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestLoadCreatesConfigWith0600(t *testing.T) {
	base := t.TempDir()
	c, err := Load(base)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Cap != DefaultCap {
		t.Errorf("cap = %d, want %d", c.Cap, DefaultCap)
	}
	if c.DSN != "" {
		t.Errorf("fresh dsn = %q, want empty (the operator fills it in)", c.DSN)
	}
	if c.DataDir != filepath.Join(base, "data") {
		t.Errorf("data_dir = %q, want %q", c.DataDir, filepath.Join(base, "data"))
	}
	st, err := os.Stat(c.ConfigPath())
	if err != nil {
		t.Fatalf("stat config: %v", err)
	}
	if runtime.GOOS != "windows" && st.Mode().Perm() != 0o600 {
		t.Errorf("config mode = %v, want 0600 (a dsn may carry a password)", st.Mode().Perm())
	}
	if _, err := os.Stat(c.DataDir); err != nil {
		t.Errorf("data_dir not created: %v", err)
	}
	if _, err := os.Stat(filepath.Dir(c.LogPath())); err != nil {
		t.Errorf("logs dir not created: %v", err)
	}
	body, _ := os.ReadFile(c.ConfigPath())
	for _, want := range []string{`dsn = ""`, "cap = 1", "keep_awake = true", "data_dir = "} {
		if !strings.Contains(string(body), want) {
			t.Errorf("written config missing %q:\n%s", want, body)
		}
	}
	for _, gone := range []string{"\nport =", "\ntoken =", "\nbind =", "retention_days", "allow_api_cap", "min_free_gb"} {
		if strings.Contains(string(body), gone) {
			t.Errorf("written config still mentions %q (the HTTP API is gone):\n%s", gone, body)
		}
	}
}

func TestLoadReadsDSNAndEnvOverride(t *testing.T) {
	base := t.TempDir()
	must(t, os.WriteFile(filepath.Join(base, "config.toml"),
		[]byte("dsn = \"host=studio dbname=orbitnam user=orbitnam\"\ncap = 2\n"), 0o600))
	c, err := Load(base)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.DSN != "host=studio dbname=orbitnam user=orbitnam" || c.Cap != 2 {
		t.Errorf("dsn=%q cap=%d", c.DSN, c.Cap)
	}
	t.Setenv(DSNEnv, "host=dev dbname=orbitnam_dev user=orbitnam_dev")
	c2, err := Load(base)
	if err != nil {
		t.Fatalf("Load (env): %v", err)
	}
	if c2.DSN != "host=dev dbname=orbitnam_dev user=orbitnam_dev" {
		t.Errorf("env override not applied: %q", c2.DSN)
	}
}

func TestKeepAwakeDefaultsOnIncludingLegacyConfigs(t *testing.T) {
	legacy := t.TempDir()
	must(t, os.WriteFile(filepath.Join(legacy, "config.toml"), []byte("cap = 1\n"), 0o600))
	lc, err := Load(legacy)
	if err != nil {
		t.Fatalf("Load legacy: %v", err)
	}
	if !lc.KeepAwake {
		t.Error("legacy config without the key: keep_awake = false, want true")
	}
	off := t.TempDir()
	must(t, os.WriteFile(filepath.Join(off, "config.toml"), []byte("keep_awake = false\n"), 0o600))
	oc, err := Load(off)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if oc.KeepAwake {
		t.Error("explicit keep_awake = false was not honored")
	}
}

func TestNormalizeClampsCap(t *testing.T) {
	base := t.TempDir()
	must(t, os.WriteFile(filepath.Join(base, "config.toml"), []byte("cap = 99\n"), 0o600))
	c, err := Load(base)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Cap != MaxCap {
		t.Errorf("cap = %d, want clamped to %d", c.Cap, MaxCap)
	}
}

// Unknown keys from an older config (port, token, …) must not break the load —
// an upgrade in place keeps working; they are simply dropped on the next Save.
func TestLoadToleratesOldHTTPKeys(t *testing.T) {
	base := t.TempDir()
	must(t, os.WriteFile(filepath.Join(base, "config.toml"),
		[]byte("port = 8626\nbind = \"127.0.0.1\"\ntoken = \"abc\"\nallow_api_cap = false\nretention_days = 0\nmin_free_gb = 2\ncap = 3\n"), 0o600))
	c, err := Load(base)
	if err != nil {
		t.Fatalf("Load with old keys: %v", err)
	}
	if c.Cap != 3 {
		t.Errorf("cap = %d, want 3", c.Cap)
	}
	c.DSN = "host=x"
	if err := c.Save(); err != nil {
		t.Fatal(err)
	}
	body, _ := os.ReadFile(c.ConfigPath())
	if strings.Contains(string(body), "\nport =") || !strings.Contains(string(body), `dsn = "host=x"`) {
		t.Errorf("Save did not rewrite to the new template:\n%s", body)
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

func TestSavePersistsChangedCapAnd0600(t *testing.T) {
	base := t.TempDir()
	c, err := Load(base)
	if err != nil {
		t.Fatal(err)
	}
	c.Cap = 3
	c.DSN = "host=studio dbname=orbitnam user=orbitnam"
	if err := c.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	re, err := Load(base)
	if err != nil {
		t.Fatal(err)
	}
	if re.Cap != 3 || re.DSN != c.DSN {
		t.Errorf("after reload cap=%d dsn=%q", re.Cap, re.DSN)
	}
	info, err := os.Stat(re.ConfigPath())
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("config mode = %o, want 600", perm)
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
