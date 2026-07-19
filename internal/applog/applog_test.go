// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package applog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrintfWritesTimestampedLine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "logs", "trainer.log")
	lg, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	lg.Printf("job %s accepted", "abc")
	if err := lg.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	line := strings.TrimSpace(string(b))
	if !strings.HasSuffix(line, "job abc accepted") {
		t.Errorf("line = %q, want it to end with the message", line)
	}
	// A leading ISO-8601 timestamp: starts with a 4-digit year and a 'T'.
	if len(line) < 20 || line[4] != '-' || !strings.Contains(line[:25], "T") {
		t.Errorf("line = %q, want a leading ISO-8601 timestamp", line)
	}
}

func TestRotationKeepsOneOldGeneration(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trainer.log")
	lg, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	lg.max = 200 // force rotation quickly

	for i := 0; i < 50; i++ {
		lg.Printf("line number %d with some padding to grow the file", i)
	}
	if err := lg.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if _, err := os.Stat(path); err != nil {
		t.Errorf("current log missing after rotation: %v", err)
	}
	if _, err := os.Stat(path + ".old"); err != nil {
		t.Errorf(".old generation missing after rotation: %v", err)
	}
	// The live file must be small (rotated), not the whole run.
	st, _ := os.Stat(path)
	if st.Size() > lg.max {
		t.Errorf("live log size %d exceeds max %d after rotation", st.Size(), lg.max)
	}
}
