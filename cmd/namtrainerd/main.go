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
		if err := run(h); err != nil {
			fmt.Fprintln(os.Stderr, "namtrainerd:", err)
			exit.Store(1)
		}
	})
	os.Exit(int(exit.Load()))
}

func run(trayHandle tray.Handle) error {
	rootCtx := context.Background()

	baseDir, err := config.DefaultBaseDir()
	if err != nil {
		return err
	}
	cfg, err := config.Load(baseDir)
	if err != nil {
		return err
	}
	if cfg.DSN == "" {
		return fmt.Errorf("no database configured: set dsn in %s (or %s)", cfg.ConfigPath(), config.DSNEnv)
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

	st, err := store.Open(rootCtx, cfg.DSN, int32(config.MaxCap+4))
	if err != nil {
		lg.Printf("FATAL: open database: %v", err)
		return err
	}
	defer st.Close()

	// One daemon per worker name, enforced by the database itself. A second process on
	// this box would requeue the first one's running jobs at boot (recovery sweeps by
	// hostname) and the two would fight over one workers row forever.
	releaseIdentity, mine, err := st.ClaimIdentity(rootCtx, name)
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

	ctx, stop := signal.NotifyContext(rootCtx, os.Interrupt, syscall.SIGTERM)
	defer stop()

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
	if err := awaitFirstBeat(ctx, hb.beat, lg.Printf, 5*time.Second); err != nil {
		return err
	}

	// Wire the menu-bar controls and start the title/list refresher (skipped
	// entirely when headless). Pause lives in the pool only and is reported in
	// workers.paused; a paused daemon still heartbeats truthfully.
	if trayHandle.Live() {
		kick := make(chan struct{}, 1)
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
			// Restart = graceful stop; under launchd (KeepAlive) that is a
			// config re-read — the agent relaunches us in seconds.
			Restart: func() {
				lg.Printf("tray: restart requested (re-read config)")
				stop()
			},
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

// trayLoop drives the menu-bar status item from ONE query every few seconds: what
// this worker is running. The title is its own load (running/cap) and the clock time
// it expects to be free; the list is its own runs; the pause/resume item reflects the
// pool gate. The shared queue — everybody's rows, waiting work, who holds what — is
// the app's view, and a menu bar answers "what is THIS machine doing".
func trayLoop(ctx context.Context, h tray.Handle, st *store.Store, pool *worker.Pool, name string, kick <-chan struct{}) {
	const maxRows = 12 // mirrors the menu's pre-created slots
	update := func() {
		mine, err := st.MyRuns(ctx, name)
		if err != nil {
			return
		}
		avg, err := st.AvgSPerEpoch(ctx, name)
		if err != nil {
			return
		}
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
	logf func(string, ...any), every time.Duration) error {
	for attempt := 1; ; attempt++ {
		err := beat(ctx)
		if err == nil {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
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
