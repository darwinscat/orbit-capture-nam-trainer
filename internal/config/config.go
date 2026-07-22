// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package config loads (and, on first start, creates) the daemon's single
// config.toml, plus the derived on-disk paths the rest of the daemon uses.
//
// Layout under the base directory (macOS: ~/Library/Application Support/
// OrbitCaptureNamTrainer; other OSes: os.UserConfigDir()/OrbitCaptureNamTrainer):
//
//	config.toml            this file
//	logs/trainer.log       the human story log (see internal/applog)
//	runtime/               the managed python runtime (provisioned at first run)
//	<data_dir>/trainer.db  the SQLite database
//	<data_dir>/scratch/    per-job scratch dirs
//
// data_dir defaults to <base>/data but is configurable so the churny DB + blobs
// can live on a roomier volume than the config.
package config

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

// AppDirName is the base directory name under the OS config root.
const AppDirName = "OrbitCaptureNamTrainer"

// Defaults (ratified in the design notes).
const (
	DefaultPort = 8626
	DefaultBind = "127.0.0.1"
	DefaultCap  = 1
	MaxCap      = 8 // default 1; an Ultra-class GPU or many-core CPU box can win with more
	// DefaultRetentionDays 0 = keep forever, the default: model blobs and their
	// continuation checkpoints never expire, so every job stays continuable via
	// kind=train_more indefinitely (gcLoop skips GCExpiredModels entirely at 0). A
	// positive value is a retention window in days — the valve for a constrained
	// host — and doubles as the train_more window (the ckpt shares the model's
	// retention). normalize() resets only negatives back to this default.
	DefaultRetentionDays = 0
	DefaultMinFreeGB     = 2
)

// Config is the on-disk config plus the resolved base directory (not serialized).
type Config struct {
	Port          int    `toml:"port"`
	Bind          string `toml:"bind"`
	Token         string `toml:"token"`
	Cap           int    `toml:"cap"`
	AllowAPICap   bool   `toml:"allow_api_cap"` // default false: cap is admin-only (config / tray)
	KeepAwake     bool   `toml:"keep_awake"`
	RetentionDays int    `toml:"retention_days"`
	MinFreeGB     int    `toml:"min_free_gb"`
	DataDir       string `toml:"data_dir"`

	baseDir string // where config.toml lives; source of logs/ and runtime/
}

// BaseDir returns the directory holding config.toml.
func (c *Config) BaseDir() string { return c.baseDir }

// ConfigPath is the config.toml path.
func (c *Config) ConfigPath() string { return filepath.Join(c.baseDir, "config.toml") }

// LogPath is the story-log path (next to the config).
func (c *Config) LogPath() string { return filepath.Join(c.baseDir, "logs", "trainer.log") }

// RuntimeDir is the managed python runtime directory.
func (c *Config) RuntimeDir() string { return filepath.Join(c.baseDir, "runtime") }

// DBPath is the SQLite database path.
func (c *Config) DBPath() string { return filepath.Join(c.DataDir, "trainer.db") }

// ScratchDir is the parent of per-job scratch directories.
func (c *Config) ScratchDir() string { return filepath.Join(c.DataDir, "scratch") }

// Addr is the host:port the HTTP server binds.
func (c *Config) Addr() string { return fmt.Sprintf("%s:%d", c.Bind, c.Port) }

// DefaultBaseDir resolves the base directory. ONCT_BASE_DIR overrides it (used
// by tests and by verification runs that must not touch real app state).
func DefaultBaseDir() (string, error) {
	if v := os.Getenv("ONCT_BASE_DIR"); v != "" {
		return v, nil
	}
	root, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("resolve user config dir: %w", err)
	}
	return filepath.Join(root, AppDirName), nil
}

// Load reads config.toml under baseDir, creating it with defaults (and a freshly
// generated token, mode 0600) if it is absent. It also ensures data_dir and the
// logs directory exist. A config with an empty token is repaired: a token is
// generated and the file rewritten.
func Load(baseDir string) (*Config, error) {
	if err := os.MkdirAll(baseDir, 0o700); err != nil {
		return nil, fmt.Errorf("create base dir: %w", err)
	}
	c := &Config{
		Port:          DefaultPort,
		Bind:          DefaultBind,
		Cap:           DefaultCap,
		KeepAwake:     true, // set before decode so a config lacking the key keeps the default
		RetentionDays: DefaultRetentionDays,
		MinFreeGB:     DefaultMinFreeGB,
		DataDir:       filepath.Join(baseDir, "data"),
		baseDir:       baseDir,
	}

	path := filepath.Join(baseDir, "config.toml")
	switch _, err := os.Stat(path); {
	case err == nil:
		if _, err := toml.DecodeFile(path, c); err != nil {
			return nil, fmt.Errorf("parse %s: %w", path, err)
		}
		c.baseDir = baseDir // DecodeFile can't set an unexported field; keep it
		if c.DataDir == "" {
			c.DataDir = filepath.Join(baseDir, "data")
		}
	case os.IsNotExist(err):
		tok, err := newToken()
		if err != nil {
			return nil, err
		}
		c.Token = tok
		if err := writeConfig(path, c); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}

	// A hand-edited config that lost its token is repaired rather than left to
	// authenticate every request against the empty string.
	if c.Token == "" {
		tok, err := newToken()
		if err != nil {
			return nil, err
		}
		c.Token = tok
		if err := writeConfig(path, c); err != nil {
			return nil, err
		}
	}

	if err := c.normalize(); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(c.DataDir, 0o700); err != nil {
		return nil, fmt.Errorf("create data dir: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(c.LogPath()), 0o700); err != nil {
		return nil, fmt.Errorf("create logs dir: %w", err)
	}
	return c, nil
}

// normalize clamps out-of-range values and validates hard invariants.
func (c *Config) normalize() error {
	if c.Port < 1 || c.Port > 65535 {
		return fmt.Errorf("port %d out of range 1..65535", c.Port)
	}
	if c.Bind == "" {
		c.Bind = DefaultBind
	}
	if c.Cap < 1 {
		c.Cap = 1
	}
	if c.Cap > MaxCap {
		c.Cap = MaxCap
	}
	// Only negatives are invalid. 0 is now meaningful — "keep forever": the gcLoop
	// skips GCExpiredModels entirely at 0 (see cmd/namtrainerd), so it never GCs a
	// checkpoint and continued training (train_more) stays possible indefinitely.
	// That GC-skip is what makes 0 safe; the old guard reset it because the old
	// semantics would have instantly expired everything.
	if c.RetentionDays < 0 {
		c.RetentionDays = DefaultRetentionDays
	}
	if c.MinFreeGB < 0 {
		c.MinFreeGB = 0
	}
	return nil
}

// MinFreeBytes is the disk floor in bytes.
func (c *Config) MinFreeBytes() uint64 { return uint64(c.MinFreeGB) * 1 << 30 }

// Save atomically rewrites config.toml with the receiver's current values —
// the same commented template first-start creates. Hand-added comments do not
// survive, and neither does a hand-edit made after this process loaded the
// file (last writer wins) — exactly like the empty-token repair path. Used by
// the menu-bar and API cap controls to persist the runtime-mutable fields
// (cap, allow_api_cap); both are applied live and written from live state.
func (c *Config) Save() error { return writeConfig(c.ConfigPath(), c) }

func newToken() (string, error) {
	var b [32]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("generate token: %w", err)
	}
	return hex.EncodeToString(b[:]), nil
}

// writeConfig writes a commented config.toml at mode 0600 (atomic via temp+rename).
func writeConfig(path string, c *Config) error {
	content := fmt.Sprintf(configTemplate,
		c.Port, c.Bind, c.Token, c.Cap, c.AllowAPICap, c.KeepAwake, c.RetentionDays, c.MinFreeGB, c.DataDir)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(content), 0o600); err != nil {
		return fmt.Errorf("write config: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("install config: %w", err)
	}
	return nil
}

const configTemplate = `# OrbitCaptureNamTrainer — daemon configuration.
# Created with defaults on first start; edit and restart to apply changes.

# TCP port the HTTP API listens on.
port = %d

# Bind address. Default 127.0.0.1 (localhost only). To reach the daemon over
# Tailscale, set this to the machine's Tailscale IP (e.g. "100.x.y.z").
# NEVER 0.0.0.0 — that would expose the trainer to every network the machine
# is attached to.
bind = "%s"

# Bearer token clients must send as ` + "`Authorization: Bearer <token>`" + `.
# Generated once from 32 random bytes. Keep this file at mode 0600.
token = "%s"

# Max concurrent training jobs. 1 is the safe default; a big GPU (M2 Ultra runs
# 2 comfortably) or a many-core CPU box may win with more. Clamped to at most 8.
cap = %d

# Allow clients to change cap over the API (PATCH /v1/cap). Off by default:
# only this machine's admin — this file, or the macOS menu-bar toggle — may
# resize the training lane; a client's attempt then answers 403.
allow_api_cap = %t

# Keep the machine awake while the queue has work (macOS: an idle-sleep power
# assertion, released once the queue drains). Without it a laptop that idle-sleeps
# freezes the trainer mid-run, so an overnight queue barely advances. It does NOT
# override sleep from closing the lid — keep the lid open, or run clamshell on
# external power + display. No effect on non-macOS.
keep_awake = %t

# How long a finished job's .nam (and its training checkpoint) is kept.
#   0  = keep forever (the default): blobs never expire, and every job stays
#        continuable via kind=train_more indefinitely.
#   N  = a retention window of N days: the model blob AND its checkpoint are freed
#        once finished more than N days ago — the valve for a constrained host.
#        Since the checkpoint expires with the model, N is ALSO how long the job
#        stays continuable via kind=train_more; lowering it shortens that window.
# Job rows and per-job logs are kept indefinitely either way — only blobs expire.
# NB: this default reaches only a FRESH config; an existing config.toml keeps
# whatever is written below, so edit it by hand to change the behavior.
retention_days = %d

# Refuse new jobs when the data volume has less than this many GB free.
min_free_gb = %d

# Where the SQLite database and per-job scratch dirs live. Defaults to
# <base>/data; point it at a roomier volume if the 27 MB capture blobs churn.
data_dir = "%s"
`
