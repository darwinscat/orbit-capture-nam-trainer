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
	"strconv"
	"strings"

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

// Defaults for a library nobody has pointed anywhere yet. The workshop this was
// built for is one Mac Studio on a local network, so these are its numbers; the
// Setup window is where they are changed, and on a fresh machine they are at least
// a shape to correct rather than six empty boxes.
const (
	DefaultHost     = "192.168.178.72"
	DefaultPort     = 5432
	DefaultDatabase = "orbitnam_dev"
	DefaultUser     = "orbitnam_dev"
	DefaultSchema   = "public"
)

// Config is the on-disk config plus the resolved base directory (not serialized).
//
// THE LIBRARY IS SIX FIELDS, NOT A CONNECTION STRING. A DSN is a sentence with a
// grammar, and the part nobody guesses is the schema — it hides inside
// `options=-csearch_path=…`. The Setup window in the menu bar writes these; the DSN
// is composed from them at connect time. A `dsn = "…"` from an older install is
// still read and split into the fields once, so nothing needs a hand-edit.
type Config struct {
	// Its own table, and not six keys at the top level, because an HTTP-era config
	// has a top-level `port` that meant the API's — reading that as a database port
	// is the kind of silent mis-take nobody would ever suspect.
	Library Library `toml:"library"`

	DSN       string `toml:"dsn"` // legacy / ORBITNAM_DSN override; composed from the fields otherwise
	Cap       int    `toml:"cap"`
	KeepAwake bool   `toml:"keep_awake"`
	DataDir   string `toml:"data_dir"`

	baseDir string // where config.toml lives; source of logs/ and runtime/
}

// Library is where the shared library lives, in the parts a person knows.
type Library struct {
	Host     string `toml:"host"`
	Port     int    `toml:"port"`
	Database string `toml:"database"`
	User     string `toml:"user"`
	Password string `toml:"password"`
	Schema   string `toml:"schema"`
}

// ConnString is what pgx is handed. Empty fields are simply left out, so a library
// on a unix socket or with trust auth needs no ceremony.
func (c *Config) ConnString() string {
	var b []string
	add := func(k, v string) {
		if v != "" {
			b = append(b, k+"="+v)
		}
	}
	add("host", c.Library.Host)
	if c.Library.Port > 0 {
		add("port", strconv.Itoa(c.Library.Port))
	}
	add("dbname", c.Library.Database)
	add("user", c.Library.User)
	add("password", c.Library.Password)
	add("options", searchPathOption(c.Library.Schema))
	return strings.Join(b, " ")
}

func searchPathOption(schema string) string {
	if schema == "" {
		return ""
	}
	return "-csearch_path=" + schema
}

// splitDSN fills empty fields from a libpq connection string. Used once for an
// install that predates the fields, and by the Setup window to show what an
// ORBITNAM_DSN override actually says.
func splitDSN(c *Config, dsn string) {
	for _, part := range strings.Fields(dsn) {
		k, v, ok := strings.Cut(part, "=")
		if !ok || v == "" {
			continue
		}
		switch k {
		case "host":
			c.Library.Host = v
		case "port":
			if n, err := strconv.Atoi(v); err == nil {
				c.Library.Port = n
			}
		case "dbname":
			c.Library.Database = v
		case "user":
			c.Library.User = v
		case "password":
			c.Library.Password = v
		case "options":
			if _, sp, ok := strings.Cut(v, "-csearch_path="); ok {
				c.Library.Schema = sp
			}
		}
	}
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
		Library: Library{Host: DefaultHost, Port: DefaultPort, Database: DefaultDatabase,
			User: DefaultUser, Schema: DefaultSchema},
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
	// An install that predates the fields carries `dsn = "…"`: split it once so the
	// window has something true to show, and so nothing has to be typed again.
	if c.DSN != "" {
		splitDSN(c, c.DSN)
	}
	c.DSN = c.ConnString()
	// ONE RUN AGAINST ANOTHER LIBRARY, without editing anything. The string is used verbatim — it may
	// say things these fields cannot — and the fields are filled from it so the Setup window and the
	// log show where this run actually went.
	if v := os.Getenv(DSNEnv); v != "" {
		splitDSN(c, v)
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
	content := fmt.Sprintf(configTemplate, c.Cap, c.KeepAwake, c.DataDir,
		c.Library.Host, c.Library.Port, c.Library.Database, c.Library.User, c.Library.Password,
		c.Library.Schema)
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
# Written by the menu bar (Setup…) and on first start. Hand-editing still works; it
# is simply not the way, and a hand-added comment does not survive the next write.

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

# WHERE THE SHARED LIBRARY IS — the queue this daemon works and the place every
# result lands.
#
# "schema" is the part that has no default worth guessing: "public" is the real
# library, anything else is a scratch one. "password" is often empty (trust auth on
# a local network) — this file is written at mode 0600 for when it is not.
# ORBITNAM_DSN overrides the whole address for one run.
#
# This is a table on purpose: an HTTP-era config had a top-level "port" that meant
# the API's, and reading that as a database port would be a silent mis-take.
[library]
host = %q
port = %d
database = %q
user = %q
password = %q
schema = %q
`
