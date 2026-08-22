package main

import (
	"os"
	"testing"
)

// ONE DAEMON PER MACHINE is the rule, and the hostname is what enforces it: workers.name is a primary
// key, so two daemons on one box would fight over one row. The override exists so the rule can be
// TESTED — a second claimant on this machine is the only way to watch two of them race for one job.
func TestWorkerNameFallsBackToTheHostname(t *testing.T) {
	t.Setenv(WorkerNameEnv, "")
	host, err := os.Hostname()
	if err != nil {
		t.Skip("no hostname on this box")
	}
	got, err := workerName()
	if err != nil {
		t.Fatalf("workerName: %v", err)
	}
	if got != host {
		t.Errorf("workerName = %q, want the hostname %q", got, host)
	}
}

func TestWorkerNameHonoursTheOverride(t *testing.T) {
	t.Setenv(WorkerNameEnv, "  bench-two  ")
	got, err := workerName()
	if err != nil {
		t.Fatalf("workerName: %v", err)
	}
	// Trimmed: workers.name is a key, and a name with a stray space is a second worker that looks
	// like the first in every log line that prints it.
	if got != "bench-two" {
		t.Errorf("workerName = %q, want %q", got, "bench-two")
	}
}

// Whitespace only is not a name. It must not become one, or a copy-pasted empty export would give a
// worker the name " " and quietly split the queue in two.
func TestWorkerNameIgnoresBlankOverride(t *testing.T) {
	t.Setenv(WorkerNameEnv, "   ")
	host, _ := os.Hostname()
	got, err := workerName()
	if err != nil {
		t.Fatalf("workerName: %v", err)
	}
	if got != host {
		t.Errorf("workerName = %q, want the hostname %q", got, host)
	}
}
