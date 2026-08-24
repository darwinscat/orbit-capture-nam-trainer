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
	// A fresh install is not empty any more: it is the workshop's own numbers, which are either
	// right or a shape to correct in the Setup window — six empty boxes are neither.
	if c.Library.Host != DefaultHost || c.Library.Port != DefaultPort || c.Library.Database != DefaultDatabase ||
		c.Library.User != DefaultUser || c.Library.Schema != DefaultSchema || c.Library.Password != "" {
		t.Errorf("fresh library address = %+v", *c)
	}
	if want := "host=" + DefaultHost + " port=5432 dbname=" + DefaultDatabase + " user=" + DefaultUser +
		" options=-csearch_path=" + DefaultSchema; c.DSN != want {
		t.Errorf("fresh dsn = %q, want %q", c.DSN, want)
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
	for _, want := range []string{"[library]", `host = "` + DefaultHost + `"`, "port = 5432",
		`database = "` + DefaultDatabase + `"`, `user = "` + DefaultUser + `"`, `password = ""`,
		`schema = "public"`, "cap = 1", "keep_awake = true", "data_dir = "} {
		if !strings.Contains(string(body), want) {
			t.Errorf("written config missing %q:\n%s", want, body)
		}
	}
	for _, gone := range []string{"\ntoken =", "\nbind =", "retention_days", "allow_api_cap", "min_free_gb"} {
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
	// The legacy string is split into the fields and composed back — same address, said in full.
	if c.Library.Host != "studio" || c.Library.Database != "orbitnam" || c.Cap != 2 {
		t.Errorf("split legacy dsn = %+v cap=%d", c.Library, c.Cap)
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
	// The HTTP API's `port = 8626` is NOT the database's port: it belongs to no table and is dropped.
	if c.Library.Port != DefaultPort {
		t.Errorf("an HTTP-era port leaked into the library address: %d", c.Library.Port)
	}
	c.Library.Host = "x"
	if err := c.Save(); err != nil {
		t.Fatal(err)
	}
	body, _ := os.ReadFile(c.ConfigPath())
	if strings.Contains(string(body), "\ntoken =") || !strings.Contains(string(body), `host = "x"`) {
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
	c.Library.Host, c.Library.Database = "studio", "orbitnam"
	c.Library.User, c.Library.Schema, c.Library.Password = "orbitnam", "onc_scratch", "hunter2"
	if err := c.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	re, err := Load(base)
	if err != nil {
		t.Fatal(err)
	}
	if re.Cap != 3 || re.Library.Host != "studio" || re.Library.Schema != "onc_scratch" ||
		re.Library.Password != "hunter2" {
		t.Errorf("after reload cap=%d library=%+v", re.Cap, re.Library)
	}
	// The fields ARE the address; the string handed to pgx is composed from them, and the schema
	// rides in the one keyword nobody guesses.
	if want := "host=studio port=5432 dbname=orbitnam user=orbitnam password=hunter2 options=-csearch_path=onc_scratch"; re.DSN != want {
		t.Errorf("composed dsn = %q, want %q", re.DSN, want)
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

// AN INSTALL THAT PREDATES THE FIELDS still says `dsn = "…"`, and nobody should have to type its
// parts again. It is split once, on the first load, and composed back from the fields after that.
func TestALegacyDSNIsSplitIntoFields(t *testing.T) {
	base := t.TempDir()
	legacy := "dsn = \"host=studio.local port=5433 dbname=orbitnam user=bench password=p options=-csearch_path=onc_pair\"\ncap = 2\n"
	if err := os.WriteFile(filepath.Join(base, "config.toml"), []byte(legacy), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := Load(base)
	if err != nil {
		t.Fatal(err)
	}
	if c.Library.Host != "studio.local" || c.Library.Port != 5433 || c.Library.Database != "orbitnam" ||
		c.Library.User != "bench" || c.Library.Password != "p" || c.Library.Schema != "onc_pair" {
		t.Errorf("legacy dsn split = %+v", *c)
	}
	if c.Cap != 2 {
		t.Errorf("cap = %d, want 2 (the rest of the file still counts)", c.Cap)
	}
}
