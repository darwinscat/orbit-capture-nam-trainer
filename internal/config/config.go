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
//	<data_dir>/scratch/    per-job scratch dirs (the only data the daemon keeps on disk)
//
// The queue and every result live in the SHARED PostgreSQL library the app owns;
// `dsn` says where it is.
package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

// AppDirName is the base directory name under the OS config root.
const AppDirName = "OrbitCaptureNamTrainer"

// Defaults (ratified in the design notes).
const (
	DefaultCap = 1
	MaxCap     = 8 // default 1; an Ultra-class GPU or many-core CPU box can win with more
)

// DSNEnv overrides config.toml's dsn for one run (handy against a dev database
// without editing the file).
const DSNEnv = "ORBITNAM_DSN"

// Config is the on-disk config plus the resolved base directory (not serialized).
type Config struct {
	DSN       string `toml:"dsn"`
	Cap       int    `toml:"cap"`
	KeepAwake bool   `toml:"keep_awake"`
	DataDir   string `toml:"data_dir"`

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

// ScratchDir is the parent of per-job scratch directories.
func (c *Config) ScratchDir() string { return filepath.Join(c.DataDir, "scratch") }

// PausedFile remembers a pause across restarts. In DataDir and NOT in ScratchDir: recovery wipes
// scratch on every start, which is exactly the moment this has to survive.
func (c *Config) PausedFile() string { return filepath.Join(c.DataDir, "paused") }

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

// Load reads config.toml under baseDir, creating it with defaults (mode 0600 —
// a dsn may carry a password) if it is absent. It also ensures data_dir and the
// logs directory exist. ORBITNAM_DSN, when set, overrides the file's dsn.
func Load(baseDir string) (*Config, error) {
	if err := os.MkdirAll(baseDir, 0o700); err != nil {
		return nil, fmt.Errorf("create base dir: %w", err)
	}
	c := &Config{
		Cap:       DefaultCap,
		KeepAwake: true, // set before decode so a config lacking the key keeps the default
		DataDir:   filepath.Join(baseDir, "data"),
		baseDir:   baseDir,
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
		if err := writeConfig(path, c); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	if v := os.Getenv(DSNEnv); v != "" {
		c.DSN = v
	}

	c.normalize()
	if err := os.MkdirAll(c.DataDir, 0o700); err != nil {
		return nil, fmt.Errorf("create data dir: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(c.LogPath()), 0o700); err != nil {
		return nil, fmt.Errorf("create logs dir: %w", err)
	}
	return c, nil
}

// normalize clamps out-of-range values.
func (c *Config) normalize() {
	if c.Cap < 1 {
		c.Cap = 1
	}
	if c.Cap > MaxCap {
		c.Cap = MaxCap
	}
}

// Save atomically rewrites config.toml with the receiver's current values —
// the same commented template first-start creates. Hand-added comments do not
// survive, and neither does a hand-edit made after this process loaded the
// file (last writer wins). Used by the menu-bar cap control and the app's
// train_cap_wanted to persist the runtime-mutable cap.
func (c *Config) Save() error { return writeConfig(c.ConfigPath(), c) }

// writeConfig writes a commented config.toml at mode 0600 (atomic via temp+rename).
func writeConfig(path string, c *Config) error {
	content := fmt.Sprintf(configTemplate, c.DSN, c.Cap, c.KeepAwake, c.DataDir)
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

# The shared OrbitCapture NAM library (PostgreSQL) — the queue this daemon works
# and the place every result lands. A libpq connection string, e.g.
#   "host=studio.local port=5432 dbname=orbitnam user=orbitnam"
# (trust/peer auth) or with password=… ; keep this file at mode 0600 if it has one.
# The environment variable ORBITNAM_DSN overrides this for one run.
dsn = %q

# Max concurrent training jobs. 1 is the safe default; a big GPU (M2 Ultra runs
# 2 comfortably) or a many-core CPU box may win with more. Clamped to at most 8.
# The app can ask for another value (workers.train_cap_wanted); the ask is applied
# live and written back here.
cap = %d

# Keep the machine awake while the queue has work (macOS: an idle-sleep power
# assertion, released once the queue drains). Without it a laptop that idle-sleeps
# freezes the trainer mid-run, so an overnight queue barely advances. It does NOT
# override sleep from closing the lid — keep the lid open, or run clamshell on
# external power + display. No effect on non-macOS.
keep_awake = %t

# Where per-job scratch dirs live (a take's wav, the trainer's checkpoints while it
# runs). Defaults to <base>/data; point it at a roomier volume if needed.
data_dir = %q
`
