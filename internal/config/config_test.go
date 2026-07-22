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

func TestKeepAwakeDefaultsOnIncludingLegacyConfigs(t *testing.T) {
	// A fresh config defaults keep_awake on, and writes the key into the file.
	base := t.TempDir()
	c, err := Load(base)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !c.KeepAwake {
		t.Error("fresh config: keep_awake = false, want true (default on)")
	}
	body, err := os.ReadFile(c.ConfigPath())
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	if want := "keep_awake = true"; !strings.Contains(string(body), want) {
		t.Errorf("written config missing %q:\n%s", want, body)
	}

	// A legacy config that predates the key must still default on, not off (the
	// default is set before decode, so an absent key keeps it).
	legacy := t.TempDir()
	must(t, os.WriteFile(filepath.Join(legacy, "config.toml"),
		[]byte("token=\"x\"\ncap = 1\n"), 0o600))
	lc, err := Load(legacy)
	if err != nil {
		t.Fatalf("Load legacy: %v", err)
	}
	if !lc.KeepAwake {
		t.Error("legacy config without the key: keep_awake = false, want true")
	}
}

func TestKeepAwakeRespectsExplicitFalse(t *testing.T) {
	base := t.TempDir()
	must(t, os.WriteFile(filepath.Join(base, "config.toml"),
		[]byte("token=\"x\"\nkeep_awake = false\n"), 0o600))
	c, err := Load(base)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.KeepAwake {
		t.Error("explicit keep_awake = false was not honored")
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

func TestNormalizeRetentionDays(t *testing.T) {
	// 0 is now a valid, meaningful setting (keep forever); only negatives reset to
	// the default (which is itself 0). A positive window is kept as written.
	cases := []struct {
		name  string
		write string
		want  int
	}{
		{"zero kept (keep forever)", "token=\"x\"\nretention_days = 0\n", 0},
		{"negative resets to default", "token=\"x\"\nretention_days = -5\n", DefaultRetentionDays},
		{"positive window kept", "token=\"x\"\nretention_days = 30\n", 30},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base := t.TempDir()
			must(t, os.WriteFile(filepath.Join(base, "config.toml"), []byte(tc.write), 0o600))
			c, err := Load(base)
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if c.RetentionDays != tc.want {
				t.Errorf("retention_days = %d, want %d", c.RetentionDays, tc.want)
			}
		})
	}
}

func TestFreshConfigRetentionKeepForever(t *testing.T) {
	base := t.TempDir()
	c, err := Load(base)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.RetentionDays != 0 {
		t.Errorf("fresh retention_days = %d, want 0 (keep forever, the default)", c.RetentionDays)
	}
	body, err := os.ReadFile(c.ConfigPath())
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	if !strings.Contains(string(body), "retention_days = 0") {
		t.Errorf("fresh config should write retention_days = 0:\n%s", body)
	}
	// The template must document the new semantics, not the old windowed-only one.
	if !strings.Contains(string(body), "keep forever") {
		t.Errorf("config template missing the keep-forever semantics:\n%s", body)
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

func TestSavePersistsChangedCapKeepsTokenAnd0600(t *testing.T) {
	base := t.TempDir()
	c, err := Load(base)
	if err != nil {
		t.Fatal(err)
	}
	tok := c.Token

	c.Cap = 3
	if err := c.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	re, err := Load(base)
	if err != nil {
		t.Fatal(err)
	}
	if re.Cap != 3 {
		t.Errorf("cap after reload = %d, want 3", re.Cap)
	}
	if re.Token != tok {
		t.Error("token changed across Save")
	}
	info, err := os.Stat(re.ConfigPath())
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("config mode = %o, want 600", perm)
	}
}
