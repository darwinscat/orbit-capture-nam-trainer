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
	"sync/atomic"
	"syscall"
	"time"

	"orbit-capture-nam-trainer/internal/applog"
	"orbit-capture-nam-trainer/internal/awake"
	"orbit-capture-nam-trainer/internal/buildinfo"
	"orbit-capture-nam-trainer/internal/config"
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

	name, err := os.Hostname()
	if err != nil || name == "" {
		return fmt.Errorf("resolve hostname (workers.name): %w", err)
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
		Ready:   hb.ready,
		Profile: hb.provenance,
	})
	hb.pool = pool

	ctx, stop := signal.NotifyContext(rootCtx, os.Interrupt, syscall.SIGTERM)
	defer stop()

	// The first heartbeat runs BEFORE any claim: jobs.worker references
	// workers(name), and it is also the first schema check.
	if err := hb.beat(ctx); err != nil {
		lg.Printf("FATAL: first heartbeat: %v", err)
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

// trayLoop drives the menu-bar status item from one queue snapshot every few
// seconds: the title (running/queued counts over the whole shared queue, the
// clock-time ETA for every lane to drain at this worker's speed), the dropdown
// queue list, and the pause/resume item state.
func trayLoop(ctx context.Context, h tray.Handle, st *store.Store, pool *worker.Pool, name string, kick <-chan struct{}) {
	const maxRows = 12 // mirrors the menu's pre-created slots
	update := func() {
		running, queued, remaining, err := st.QueueTotals(ctx)
		if err != nil {
			return
		}
		avg, err := st.AvgSPerEpoch(ctx, name)
		if err != nil {
			return
		}
		var etaSecs *float64
		if avg != nil {
			secs := tray.QueueSeconds(remaining, *avg, pool.Cap(), probeCap)
			if secs > 0 {
				etaSecs = &secs
			}
		}
		h.SetTitle(tray.Format(time.Now(), running, queued, etaSecs, avg))

		rows, err := st.QueueRows(ctx, maxRows)
		if err != nil {
			return
		}
		list := make([]tray.QueueRow, len(rows))
		for i, r := range rows {
			list[i] = tray.QueueRow{Running: r.Running, Kind: r.Kind, Epochs: r.Epochs, Epoch: r.Epoch, Label: r.Label}
		}
		h.SetQueue(list, running+queued-len(rows))
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
