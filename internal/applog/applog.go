// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package applog is the daemon's human-facing story log: timestamped plain-text
// lines (start, job accepted/started/succeeded/failed/deleted, GC, provisioning)
// written next to config.toml. Its reader is a human ssh'd into an unattended
// studio Mac debugging the daemon in EVERY failure mode — including a broken or
// locked database — so it is deliberately plain text and never SQLite: logging
// must be the last thing that fails. Per-job training output is DATA and lives
// in the job_log table instead.
//
// The file rotates at ~1 MB, keeping one .old generation.
package applog

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// MaxSize is the rotation threshold in bytes (~1 MB).
const MaxSize = 1 << 20

// Logger is a rotating, timestamped line logger safe for concurrent use.
type Logger struct {
	mu   sync.Mutex
	path string
	f    *os.File
	size int64
	max  int64
	now  func() time.Time // injectable for tests
}

// Open opens (creating the parent directory and the file if needed) the story
// log at path in append mode.
func Open(path string) (*Logger, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, fmt.Errorf("create log dir: %w", err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open log: %w", err)
	}
	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("stat log: %w", err)
	}
	return &Logger{path: path, f: f, size: st.Size(), max: MaxSize, now: time.Now}, nil
}

// Printf writes one timestamped line. It never returns an error: the story log
// must not become a failure path of its own — a write error is swallowed (the
// daemon keeps running) rather than propagated.
func (l *Logger) Printf(format string, args ...any) {
	line := fmt.Sprintf("%s %s\n", l.now().Format("2006-01-02T15:04:05.000Z07:00"),
		fmt.Sprintf(format, args...))
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.f == nil {
		return
	}
	if l.size+int64(len(line)) > l.max {
		l.rotate()
	}
	n, err := l.f.WriteString(line)
	if err == nil {
		l.size += int64(n)
	}
}

// rotate closes the current file, moves it to <path>.old (replacing any prior
// generation), and opens a fresh file. Called under l.mu. On any error it leaves
// the current file in place — a failed rotation must not lose the handle.
func (l *Logger) rotate() {
	if err := l.f.Close(); err != nil {
		return
	}
	if err := os.Rename(l.path, l.path+".old"); err != nil {
		// Reopen the original so logging survives a rename failure.
		l.f, _ = os.OpenFile(l.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		return
	}
	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		l.f = nil
		return
	}
	l.f = f
	l.size = 0
}

// Close closes the underlying file.
func (l *Logger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.f == nil {
		return nil
	}
	err := l.f.Close()
	l.f = nil
	return err
}
