// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Command namtrainerd is the OrbitCapture NAM training daemon: a worker on the
// SHARED PostgreSQL library. It claims queued jobs, runs the managed python trainer
// on the take's audio, and writes progress, logs, snapshots and the result back
// into the same database the capture app reads. There is no HTTP API: the schema
// is the whole contract.
package main

import (
	"context"
	"crypto/rand"
	"fmt"
	"os"
	"os/signal"
	"regexp"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"orbit-capture-nam-trainer/internal/applog"
	"orbit-capture-nam-trainer/internal/awake"
	"orbit-capture-nam-trainer/internal/buildinfo"
	"orbit-capture-nam-trainer/internal/config"
	"orbit-capture-nam-trainer/internal/jobs"
	"orbit-capture-nam-trainer/internal/runtime"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/tray"
	"orbit-capture-nam-trainer/internal/worker"
)

// The probe lane is fixed at one worker (a self-check is seconds); the training
// lane width comes from config and the app's train_cap_wanted.
const probeCap = 1

func main() {
	// On macOS with a GUI session, tray.Main parks the main thread in the
	// AppKit run loop and runs the daemon body on a goroutine; everywhere else
	// it is a plain inline call.
	var exit atomic.Int32
	tray.Main(func(h tray.Handle) {
		if err := run(context.Background(), h); err != nil {
			fmt.Fprintln(os.Stderr, "namtrainerd:", err)
			exit.Store(1)
		}
	})
	os.Exit(int(exit.Load()))
}

// run is the daemon body. The root context is a parameter rather than a Background() inside
// because the ORDER of what happens here is the contract — the menu is wired before the library
// is touched — and a test can only watch that order if it can end the run.
func run(rootCtx context.Context, trayHandle tray.Handle) error {
	baseDir, err := config.DefaultBaseDir()
	if err != nil {
		return err
	}
	cfg, err := config.Load(baseDir)
	if err != nil {
		return err
	}
	lg, err := applog.Open(cfg.LogPath())
	if err != nil {
		return err
	}
	defer lg.Close()

	name, err := workerName()
	if err != nil {
		return err
	}
	instance := newInstanceID()
	lg.Printf("starting namtrainerd %s (pid %d) as worker %s instance %s, cap %d, data_dir %s, db %s",
		buildinfo.Version, os.Getpid(), name, instance, cfg.Cap, cfg.DataDir, redactDSN(cfg.DSN))

	// SIGNALS FROM HERE, not from after the library answers. The connect loop below waits on
	// this context; until it was installed this early, a daemon stuck on an unreachable database
	// waited on context.Background() and ignored SIGTERM completely — launchd asked it to stop
	// and then had to kill it.
	ctx, stop := signal.NotifyContext(rootCtx, os.Interrupt, syscall.SIGTERM)
	defer stop()

	// THE MENU HAS TO WORK BEFORE THE LIBRARY DOES, because the menu is where a wrong library
	// gets corrected. A click lands on whatever Controls hold at that moment, and these two need
	// nothing but the config: Setup rewrites where the library is, Restart re-reads it. Wiring
	// them only after the first heartbeat meant that exactly when Setup WAS the answer — a fresh
	// machine that has never connected, a database the app has not migrated — the item was there,
	// the click did nothing, and the log stayed empty. The rest of the menu is wired below, as
	// soon as the pool it acts on exists.
	trayHandle.SetControls(tray.Controls{
		OpenSetup: setupAction(cfg, lg, stop),
		Restart:   restartAction(lg, stop),
	})

	// AND NEITHER IS AN EMPTY ADDRESS, where there is a menu to fix it with. Clearing the boxes in
	// Setup and saving is enough to produce one, and a daemon that exits on it takes away the very
	// window the fields would be typed back into — leaving config.toml by hand as the only way home.
	// Headless there is no window, so there it stays what the README promises: a refusal to start.
	if cfg.DSN == "" {
		if !trayHandle.Live() {
			return fmt.Errorf("no database configured: set dsn in %s (or %s)", cfg.ConfigPath(), config.DSNEnv)
		}
		lg.Printf("no database configured: waiting for Setup… (%s)", cfg.ConfigPath())
		trayHandle.SetPaused(tray.StateNoLibrary)
		trayHandle.SetFault("library: not configured — open Setup…")
		<-ctx.Done()
		return ctx.Err()
	}

	// THE LIBRARY MAY NOT BE THERE YET, and that is not a reason to die. Under launchd this
	// returned an error and the agent respawned us every ten seconds — the menu bar icon
	// blinking in and out, with the reason only in a log nobody has open. A workshop whose
	// Studio is asleep, or a laptop carried out of the house, is exactly this case. So: say
	// so in the menu, in red, with the reason on the line under it, and keep trying.
	var st *store.Store
	for attempt := 1; ; attempt++ {
		var err error
		if st, err = store.Open(ctx, cfg.DSN, int32(config.MaxCap+4)); err == nil {
			break
		}
		lg.Printf("open database (attempt %d): %v", attempt, err)
		trayHandle.SetPaused(tray.StateNoLibrary)
		trayHandle.SetFault("library: " + firstLine(err.Error()))
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(5 * time.Second):
		}
	}
	// NOT cleared here. A socket that opened is not a library that works: the next thing is
	// the first heartbeat, and on a database the app has not migrated there is no `workers`
	// table to beat into. Clearing the reason at this point left the icon red with nothing
	// under it — a colour and no diagnosis, which is the state this was meant to end.
	defer st.Close()

	// One daemon per worker name, enforced by the database itself. A second process on
	// this box would requeue the first one's running jobs at boot (recovery sweeps by
	// hostname) and the two would fight over one workers row forever.
	releaseIdentity, mine, err := st.ClaimIdentity(ctx, name)
	if err != nil {
		lg.Printf("FATAL: %v", err)
		return err
	}
	if !mine {
		lg.Printf("FATAL: another namtrainerd is already running as worker %s against this database", name)
		return fmt.Errorf("worker %s is already taken", name)
	}
	defer releaseIdentity()

	// The capture signal (the trainer --input) is downloaded + sha-verified during
	// provisioning; the worker only spawns a trainer once ready, by which point it
	// is present.
	signalPath := runtime.SignalPath(cfg.RuntimeDir())

	// Keep the machine awake while the queue has work, so a laptop that idle-sleeps
	// doesn't freeze a training run mid-queue. Released the moment the queue drains.
	keeper := awake.New(cfg.KeepAwake, lg.Printf)
	defer keeper.Close()

	hb := &heartbeat{cfg: cfg, lg: lg, st: st, name: name, instance: instance}

	var pool *worker.Pool // declared ahead: OnCounts below reads pool.Paused()
	pool = worker.New(worker.Options{
		Store: st,
		Log:   lg,
		Runner: worker.ProcessRunner{
			Python: runtime.VenvPython(cfg.RuntimeDir()),
			Driver: runtime.DriverPath(cfg.RuntimeDir()),
		},
		WorkerName:  name,
		Instance:    instance,
		SignalPath:  signalPath,
		ScratchRoot: cfg.ScratchDir(),
		Cap:         cfg.Cap,       // training lane (GPU-bound), live-adjustable
		CapLimit:    config.MaxCap, // spawn width: SetCap can raise up to this without a restart
		ProbeCap:    probeCap,
		OnCounts: func(running, queued int) {
			// Hold while anything RUNS (a draining pause must not let the lid
			// freeze a train mid-epoch), or while queued work is claimable.
			// Fully paused with only queued work → release: a pause means the
			// machine is the musician's again, sleep included.
			keeper.Set(running > 0 || (queued > 0 && !pool.Paused()))
		},
		Ready:          hb.ready,
		Profile:        hb.provenance,
		PauseStatePath: cfg.PausedFile(),
	})
	hb.pool = pool

	// A PAUSE OUTLIVES THE PROCESS THAT WAS ASKED FOR IT. It used to live only in memory, so every
	// restart resumed — and a restart is what an upgrade, a config re-read from the tray, and a crash
	// all are. Whoever paused this trainer was usually sitting at the machine wanting their GPU back;
	// a relaunch they did not ask for took it away again. Lifting a pause is a hand's gesture: the
	// tray here, or Resume from the app. Never a launch.
	if worker.PauseWasRemembered(cfg.PausedFile()) {
		pool.Pause(false) // nothing is running yet, so there is nothing to stop — only the gate
		lg.Printf("starting paused: this trainer was paused before it last stopped (%s)", cfg.PausedFile())
	}

	// The rest of the menu, wired the moment the pool it drives exists — still before the first
	// heartbeat, so a daemon waiting on the library can be paused and capped like any other.
	// Pause lives in the pool only and is reported in workers.paused; a paused daemon still
	// heartbeats truthfully.
	kick := make(chan struct{}, 1)
	if trayHandle.Live() {
		nudge := func() {
			select {
			case kick <- struct{}{}:
			default:
			}
		}
		trayHandle.SetControls(tray.Controls{
			PauseNow:          func() { pool.Pause(true); nudge() },
			PauseAfterCurrent: func() { pool.Pause(false); nudge() },
			Resume:            func() { pool.Resume(); nudge() },
			Restart:           restartAction(lg, stop),
			OpenSetup:         setupAction(cfg, lg, stop),
			// SetCap applies LIVE (nothing killed) and persists so the next boot
			// keeps it — the same path the app's train_cap_wanted takes.
			SetCap: func(n int) {
				if n == pool.Cap() {
					return
				}
				pool.SetCap(n)
				hb.persistCap()
				nudge()
			},
		})
		trayHandle.SetCap(pool.Cap())
	}

	// The first heartbeat runs BEFORE any claim: jobs.worker references
	// workers(name), and it is also the first schema check.
	//
	// AND IT RETRIES INSTEAD OF DYING. This returned the error, and under launchd
	// (KeepAlive) that is a respawn loop at the 10 s throttle, for ever — with
	// nothing to read anywhere. The case is not exotic, it is the FIRST install on
	// a fresh workshop: a library the app has not migrated yet has no `workers`
	// table to write a note into and no `pause_wanted` column for the heartbeat's
	// RETURNING, so the beat fails and the app's lamp says "no trainer has
	// reported" — the exact opposite of the diagnosis. The README already promised
	// this behaviour ("it keeps heartbeating and retries every few seconds"); now
	// it is true. Every attempt says why, so the log names the cause once a minute
	// instead of the process dying silently.
	if err := awaitFirstBeat(ctx, hb.beat, lg.Printf, 5*time.Second, func(err error) {
		// The menu bar is the only place a person looks at this, and until the first beat
		// lands nothing else touches it — the tray loop starts below.
		trayHandle.SetPaused(tray.StateNoLibrary)
		trayHandle.SetFault("library: " + firstLine(err.Error()))
	}); err != nil {
		return err
	}
	trayHandle.SetFault("") // it beat: whatever was wrong with the library is over

	// The title/list refresher READS the queue, so it is the one piece that has to wait for a
	// library that answers — the controls above do not.
	if trayHandle.Live() {
		go trayLoop(ctx, trayHandle, st, pool, name, kick)
	}

	if err := pool.Start(ctx); err != nil {
		lg.Printf("FATAL: start worker pool: %v", err)
		return err
	}
	defer pool.Stop()

	// Provision the runtime in the background; the heartbeat reports ready=false
	// with a note meanwhile, and jobs start the moment it is up.
	go provisionLoop(ctx, cfg, lg, hb, pool)
	go hb.loop(ctx)

	<-ctx.Done()
	lg.Printf("shutdown signal received")
	pool.Stop() // join workers (in-flight jobs are requeued, not failed)
	hb.farewell()
	lg.Printf("stopped")
	return nil
}

// firstLine keeps a menu line to one line: a pgx error carries the address it tried, then a
// second line per attempt, and the menu is not a log.
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// trayLoop drives the menu-bar status item from ONE query every few seconds: what
// this worker is running. The title is its own load (running/cap) and the clock time
// it expects to be free; the list is its own runs; the pause/resume item reflects the
// pool gate. The shared queue — everybody's rows, waiting work, who holds what — is
// the app's view, and a menu bar answers "what is THIS machine doing".
func trayLoop(ctx context.Context, h tray.Handle, st *store.Store, pool *worker.Pool, name string, kick <-chan struct{}) {
	const maxRows = 12 // mirrors the menu's pre-created slots
	update := func() {
		// THE LIBRARY NOT ANSWERING IS THE ONE REAL FAULT, and it is this loop that finds
		// out first — it asks something of it every few seconds. It used to return quietly
		// and leave the menu bar showing whatever was true a minute ago.
		mine, err := st.MyRuns(ctx, name)
		if err != nil {
			h.SetTitle("")
			h.SetPaused(tray.StateNoLibrary)
			h.SetFault("library: " + firstLine(err.Error()))
			return
		}
		avg, err := st.AvgSPerEpoch(ctx, name)
		if err != nil {
			return
		}
		h.SetFault("")
		// The estimate is about this box only: its runs go on at the same time, so the
		// longest of them is when it is free. Queued work is not counted — the queue is
		// shared, and which machine claims the next row is decided when it is claimed.
		var trainRuns int
		var remaining []int64
		for _, r := range mine {
			if r.Lane == jobs.LaneTrain {
				trainRuns++
				remaining = append(remaining, r.Remaining)
			}
		}
		var etaSecs *float64
		if avg != nil {
			if secs := tray.MineSeconds(remaining, *avg); secs > 0 {
				etaSecs = &secs
			}
		}
		h.SetTitle(tray.Format(time.Now(), trainRuns, pool.Cap(), etaSecs, avg))

		shown := mine
		if len(shown) > maxRows {
			shown = shown[:maxRows]
		}
		list := make([]tray.QueueRow, len(shown))
		for i, r := range shown {
			list[i] = tray.QueueRow{Running: true, Kind: r.Kind, Epochs: r.Epochs, Epoch: r.Epoch, Label: r.Label}
		}
		h.SetQueue(list, len(mine)-len(shown))
		h.SetPaused(tray.DeriveState(pool.Paused(), pool.Running()))
		h.SetCap(pool.Cap()) // dynamic: the app's ask shows in the menu a tick later
	}
	update()
	t := time.NewTicker(3 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-kick:
			update()
		case <-t.C:
			update()
		}
	}
}

// provisionLoop brings the runtime up, retrying with capped backoff. On success
// it publishes the resolved profile (the heartbeat turns ready, the claim filter
// learns the nam version) and wakes the workers.
func provisionLoop(ctx context.Context, cfg *config.Config, lg *applog.Logger, hb *heartbeat, pool *worker.Pool) {
	backoff := 5 * time.Second
	const maxBackoff = 2 * time.Minute
	for {
		lg.Printf("provisioning runtime at %s", cfg.RuntimeDir())
		hb.setNote("provisioning runtime")
		prof, err := runtime.Provision(ctx, cfg.RuntimeDir(), func(s string) {
			lg.Printf("provision: %s", s)
			hb.setNote("provisioning: " + s)
		})
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			lg.Printf("provisioning failed: %v (retry in %s)", err, backoff)
			hb.setNote("provisioning failed: " + err.Error())
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
			if backoff < maxBackoff {
				backoff *= 2
			}
			continue
		}
		hb.setProfile(prof)
		lg.Printf("runtime ready: python=%s nam=%s driver=%s…", prof.Python, prof.Nam, prof.DriverSHA256[:12])
		if err := hb.beat(ctx); err != nil && ctx.Err() == nil {
			lg.Printf("heartbeat after provisioning: %v", err)
		}
		pool.Notify()
		return
	}
}

// newInstanceID mints workers.instance: a random uuid (v4 shape) per process start.
// WorkerNameEnv overrides workers.name. Empty or unset means the hostname, which is the rule this
// daemon runs under: ONE DAEMON PER MACHINE, so the machine's name is its identity and two daemons
// on one box would fight over one row.
//
// The override exists so that rule can be TESTED. `workers.name` is a primary key and every claim,
// heartbeat and count is keyed by it, so the only way to watch two claimants race for one job —
// which is the whole point of a shared queue, and the thing a single daemon can never demonstrate —
// is to let a second one on this machine call itself something else. It sits beside ONCT_BASE_DIR,
// which exists for the same reason: a verification run must not touch real state.
//
// Not a config.toml key on purpose. A machine that legitimately wants a different name is a machine
// whose hostname is wrong; this is scaffolding, and scaffolding belongs in the environment.
const WorkerNameEnv = "ONCT_WORKER_NAME"

// workerName resolves workers.name: the override if it is set, else this machine's hostname.
func workerName() (string, error) {
	if v := strings.TrimSpace(os.Getenv(WorkerNameEnv)); v != "" {
		return v, nil
	}
	name, err := os.Hostname()
	if err != nil || name == "" {
		return "", fmt.Errorf("resolve hostname (workers.name): %w", err)
	}
	return name, nil
}

func newInstanceID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("pid-%d-%d", os.Getpid(), time.Now().UnixNano())
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// setupAction is the menu's Setup… — WHERE THE LIBRARY IS, asked in six boxes rather than in one
// connection string nobody should have to compose by hand (the schema alone hides in
// `options=-csearch_path=…`). Saving writes config.toml and restarts, because a daemon cannot
// change the database under a run: the restart puts what it was training back in the queue, and
// it continues from its last epoch.
//
// It takes the config and the stop and NOTHING else on purpose — that is what lets it be wired
// before the database is even reachable, which is the one moment it is needed most.
func setupAction(cfg *config.Config, lg *applog.Logger, stop func()) func() {
	return func() {
		tray.ShowSetup(tray.Setup{
			Host: cfg.Library.Host, Port: cfg.Library.Port, Database: cfg.Library.Database,
			User: cfg.Library.User, Password: cfg.Library.Password, Schema: cfg.Library.Schema,
			KeepAwake: cfg.KeepAwake,
		}, func(v tray.Setup) {
			cfg.Library.Host, cfg.Library.Port = v.Host, v.Port
			cfg.Library.Database, cfg.Library.User = v.Database, v.User
			cfg.Library.Password, cfg.Library.Schema = v.Password, v.Schema
			cfg.KeepAwake = v.KeepAwake
			if err := cfg.Save(); err != nil {
				lg.Printf("tray: save setup: %v", err)
				return
			}
			lg.Printf("tray: setup saved (%s:%d/%s schema=%s) — restarting",
				v.Host, v.Port, v.Database, v.Schema)
			stop()
		})
	}
}

// restartAction is the menu's Restart: a graceful stop, which under launchd (KeepAlive) is a
// config re-read — the agent relaunches us in seconds.
func restartAction(lg *applog.Logger, stop func()) func() {
	return func() {
		lg.Printf("tray: restart requested (re-read config)")
		stop()
	}
}

var reDSNPassword = regexp.MustCompile(`(?i)(password=)\S+`)

// redactDSN hides a password before the dsn reaches the story log.
func redactDSN(dsn string) string { return reDSNPassword.ReplaceAllString(dsn, "${1}***") }

// awaitFirstBeat retries the first heartbeat until it lands or the context ends.
//
// It is a loop and not a return because of what the failure IS. Under launchd (KeepAlive) returning
// an error is a respawn at the 10 s throttle, for ever, with nothing to read: on a library the app
// has not migrated yet there is no `workers` table to write a note into and no `pause_wanted` column
// for the heartbeat's RETURNING, so the beat fails, the daemon dies, and the app's lamp says "no
// trainer has reported" — which is the opposite of the diagnosis. Waiting is the correct answer:
// the app migrates, and then this succeeds by itself.
//
// Loud for the first three tries, then once a minute — a person watching an install sees it at once,
// a machine left overnight does not fill a disk with it.
func awaitFirstBeat(ctx context.Context, beat func(context.Context) error,
	logf func(string, ...any), every time.Duration, say func(error)) error {
	for attempt := 1; ; attempt++ {
		err := beat(ctx)
		if err == nil {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if say != nil {
			say(err)
		}
		if attempt <= 3 || attempt%12 == 0 {
			logf("waiting for the library: %v (has the app opened it yet? it is the app that migrates)", err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(every):
		}
	}
}
