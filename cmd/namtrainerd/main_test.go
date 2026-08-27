package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"orbit-capture-nam-trainer/internal/applog"
	"orbit-capture-nam-trainer/internal/config"
	"orbit-capture-nam-trainer/internal/tray"
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

// recordingTray is the menu bar as a test can see it: it remembers what the daemon wired into it
// and lets the test act the moment a fault is reported.
type recordingTray struct {
	mu      sync.Mutex
	ctl     tray.Controls
	faults  []string
	onFault func(string)
}

func (r *recordingTray) Live() bool                    { return true }
func (r *recordingTray) SetTitle(string)               {}
func (r *recordingTray) SetQueue([]tray.QueueRow, int) {}
func (r *recordingTray) SetPaused(tray.PauseState)     {}
func (r *recordingTray) SetTally(tray.Tally)           {}
func (r *recordingTray) SetCap(int)                    {}

func (r *recordingTray) SetControls(c tray.Controls) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ctl = c
}

func (r *recordingTray) SetFault(reason string) {
	r.mu.Lock()
	r.faults = append(r.faults, reason)
	f := r.onFault
	r.mu.Unlock()
	if f != nil && reason != "" {
		f(reason)
	}
}

func (r *recordingTray) controls() tray.Controls {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.ctl
}

func (r *recordingTray) reported() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.faults...)
}

// runUntilTheLibraryFails starts the daemon against a port nothing listens on and ends it at the
// first reported fault — the deepest the waiting state ever gets. Bounded, because the failure this
// guards against is a loop that does not end: a hang would otherwise take the whole binary down with
// a panic instead of naming what broke.
func runUntilTheLibraryFails(t *testing.T) (*recordingTray, error) {
	t.Helper()
	t.Setenv("ONCT_BASE_DIR", t.TempDir())
	t.Setenv(config.DSNEnv, "host=127.0.0.1 port=1 dbname=x user=x connect_timeout=1")
	t.Setenv(WorkerNameEnv, "recording-tray")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	trayHandle := &recordingTray{onFault: func(string) { cancel() }}
	return trayHandle, runBounded(t, ctx, trayHandle)
}

// runBounded runs the daemon body and refuses to wait for ever. The failures these tests guard
// against are loops that do not end, and a hang would take the whole binary down with a timeout
// panic instead of naming what broke.
func runBounded(t *testing.T, ctx context.Context, trayHandle tray.Handle) error {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- run(ctx, trayHandle) }()
	select {
	case err := <-done:
		return err
	case <-time.After(10 * time.Second):
		t.Fatal("the daemon never came back: it ignored the shutdown")
		return nil
	}
}

// CLEARING THE BOXES IN SETUP MUST NOT COST THE MENU THEY ARE TYPED INTO. Six empty fields compose
// an empty connection string, and the daemon used to answer that by exiting before the menu bar
// existed — under launchd a blink every ten seconds and no way back except editing config.toml by
// hand. Now it says so and waits, with Setup… live.
func TestAnEmptyLibraryAddressKeepsTheMenu(t *testing.T) {
	base := t.TempDir()
	blank := "[library]\nhost = \"\"\nport = 0\ndatabase = \"\"\nuser = \"\"\npassword = \"\"\nschema = \"\"\n"
	if err := os.WriteFile(filepath.Join(base, "config.toml"), []byte(blank), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	t.Setenv("ONCT_BASE_DIR", base)
	t.Setenv(config.DSNEnv, "")
	t.Setenv(WorkerNameEnv, "recording-tray")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	trayHandle := &recordingTray{onFault: func(string) { cancel() }}

	if err := runBounded(t, ctx, trayHandle); err != nil {
		t.Fatalf("run = %v, want it to wait and then stop cleanly", err)
	}
	if reported := trayHandle.reported(); len(reported) == 0 || !strings.Contains(reported[0], "Setup") {
		t.Fatalf("faults reported = %v, want one naming Setup as the way out", reported)
	}
	if trayHandle.controls().OpenSetup == nil {
		t.Error("Setup… was dead on a daemon that has no library to open at all")
	}
}

// THE MENU IS WHERE A WRONG LIBRARY GETS CORRECTED, so Setup… has to be alive while the library is
// still wrong. It was not: the controls were wired only after the first heartbeat landed, so on a
// machine that had never connected — or one pointed at a database the app has not migrated — the
// item sat in the menu looking enabled, the click reached a nil func, and nothing happened and
// nothing was logged.
func TestSetupIsWiredBeforeTheLibraryIsTouched(t *testing.T) {
	trayHandle, _ := runUntilTheLibraryFails(t)

	if reported := trayHandle.reported(); len(reported) == 0 || !strings.Contains(reported[0], "library:") {
		t.Fatalf("faults reported = %v, want the library's own reason first", reported)
	}
	ctl := trayHandle.controls()
	if ctl.OpenSetup == nil {
		t.Error("Setup… was dead while the library was unreachable — the one moment it is the answer")
	}
	if ctl.Restart == nil {
		t.Error("Restart was dead while the library was unreachable")
	}
}

// …and that wait answers a shutdown ASKED FOR THE WAY LAUNCHD ASKS: a signal, against the real root
// context. The loop used to select on the context.Background() that run() made for itself, with the
// signal handler installed pages later, so a daemon stuck on an unreachable database ignored SIGTERM
// and had to be killed after the agent's timeout. Nothing else in the file can catch that — a test
// that hands run() a cancellable root cancels the very thing the bug was about.
func TestTheConnectLoopHonoursASignal(t *testing.T) {
	t.Setenv("ONCT_BASE_DIR", t.TempDir())
	t.Setenv(config.DSNEnv, "host=127.0.0.1 port=1 dbname=x user=x connect_timeout=1")
	t.Setenv(WorkerNameEnv, "recording-tray")

	// Safe to signal ourselves: run() installs the handler before it ever touches the database, so
	// by the time a fault is reported SIGTERM is caught rather than fatal, and it is unregistered
	// only once run() has returned and no further fault can fire.
	trayHandle := &recordingTray{onFault: func(string) {
		if err := syscall.Kill(os.Getpid(), syscall.SIGTERM); err != nil {
			panic("signal self: " + err.Error())
		}
	}}

	done := make(chan error, 1)
	go func() { done <- run(context.Background(), trayHandle) }()
	select {
	case err := <-done:
		// AND IT ENDS CLEANLY. Returning the cancellation made main() print it to stderr and exit
		// 1, so under systemd (Restart=on-failure) a `systemctl stop` of a daemon whose library is
		// down left the unit `failed`. A stop that was asked for is not a failure.
		if err != nil {
			t.Fatalf("run = %v, want a clean exit on a signal", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("SIGTERM was ignored: the wait for the library outlives a shutdown")
	}
}

// …and it goes quietly. A stop during the dial comes back from pgx as "ping database: context
// canceled" — our own shutdown wearing the library's clothes. Reporting it painted a fault line
// into the menu on the way out and logged an attempt that never happened.
func TestAShutdownDuringTheDialSaysNothing(t *testing.T) {
	t.Setenv("ONCT_BASE_DIR", t.TempDir())
	t.Setenv(config.DSNEnv, "host=127.0.0.1 port=1 dbname=x user=x connect_timeout=1")
	t.Setenv(WorkerNameEnv, "recording-tray")

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already asked to stop before the first dial
	trayHandle := &recordingTray{}

	if err := runBounded(t, ctx, trayHandle); err != nil {
		t.Fatalf("run = %v, want a clean exit", err)
	}
	if reported := trayHandle.reported(); len(reported) != 0 {
		t.Errorf("faults reported = %v, want none: the library never said anything", reported)
	}
}

// A CAP SET FROM THE MENU AN HOUR AGO MUST NOT COME BACK REVERTED because somebody typed a new
// password. Save() rewrites config.toml whole and the Setup window knows seven of its nine values;
// it used to carry the rest from the struct this process booted with, where `cap` is frozen at its
// start value — the live one lives in the pool, and persistCap only ever wrote a copy of it to disk.
func TestSavingTheAddressKeepsWhatTheWindowDoesNotAskAbout(t *testing.T) {
	base := t.TempDir()
	t.Setenv("ONCT_BASE_DIR", base)
	t.Setenv(config.DSNEnv, "")

	booted, err := config.Load(base) // cap 1, the default — and frozen here for the run
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	lg, err := applog.Open(booted.LogPath())
	if err != nil {
		t.Fatalf("open log: %v", err)
	}
	defer lg.Close()

	// The menu control (or the app, through train_cap_wanted) raises the cap. It lands on disk and
	// never in `booted` — which is the whole trap.
	raised, err := config.Load(base)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	raised.Cap = 4
	if err := raised.Save(); err != nil {
		t.Fatalf("save cap: %v", err)
	}

	var stopped bool
	applySetup(booted, lg, func() { stopped = true })(tray.Setup{
		Host: "studio.local", Port: 5432, Database: "orbitnam", User: "orbitnam",
		Password: "secret", Schema: "public", KeepAwake: false,
	})
	if !stopped {
		t.Error("a saved address must restart the daemon: nothing else applies it")
	}

	after, err := config.Load(base)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if after.Cap != 4 {
		t.Errorf("cap = %d, want the 4 that was on disk", after.Cap)
	}
	if after.Library.Host != "studio.local" || after.Library.Schema != "public" {
		t.Errorf("library = %+v, want what the window typed", after.Library)
	}
	if after.KeepAwake {
		t.Error("keep_awake = true, want the false the window said")
	}
}
