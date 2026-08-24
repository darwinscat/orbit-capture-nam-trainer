package main

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
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

// The first heartbeat against a library the app has not migrated yet must WAIT, not kill the process:
// under launchd that death is a respawn loop at the 10 s throttle with no diagnostic anywhere.
func TestTheFirstHeartbeatWaitsForTheLibrary(t *testing.T) {
	calls := 0
	var said []string
	err := awaitFirstBeat(context.Background(), func(context.Context) error {
		calls++
		if calls < 4 {
			return errors.New(`ERROR: column "pause_wanted" does not exist (SQLSTATE 42703)`)
		}
		return nil
	}, func(string, ...any) {}, time.Millisecond, func(e error) { said = append(said, e.Error()) })
	if err != nil {
		t.Fatalf("want the fourth beat to land, got %v", err)
	}
	if calls != 4 {
		t.Fatalf("want 4 attempts, got %d", calls)
	}
	// AND EVERY FAILED ATTEMPT IS SAID OUT LOUD, because until the beat lands nothing else
	// touches the menu bar: an icon that went red at the socket and then said nothing is a
	// colour without a diagnosis, which is what this whole state exists to avoid.
	if len(said) != 3 || !strings.Contains(said[0], "pause_wanted") {
		t.Errorf("reasons reported = %v, want the three failures with the library's own words", said)
	}
}

// …and it gives up the moment the daemon is asked to stop, rather than retrying through a shutdown.
func TestTheWaitEndsWhenTheDaemonIsStopped(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	err := awaitFirstBeat(ctx, func(context.Context) error {
		calls++
		if calls == 2 {
			cancel()
		}
		return errors.New("library unreachable")
	}, func(string, ...any) {}, time.Millisecond, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}
