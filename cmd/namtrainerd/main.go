// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Command namtrainerd is the OrbitCapture NAM training daemon: it accepts capture
// WAVs over HTTP, queues them, runs the managed python trainer, and serves back
// the .nam result.
package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"sync/atomic"
	"syscall"
	"time"

	"orbit-capture-nam-trainer/internal/applog"
	"orbit-capture-nam-trainer/internal/awake"
	"orbit-capture-nam-trainer/internal/buildinfo"
	"orbit-capture-nam-trainer/internal/config"
	"orbit-capture-nam-trainer/internal/httpapi"
	"orbit-capture-nam-trainer/internal/runtime"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/tray"
	"orbit-capture-nam-trainer/internal/worker"
)

// Probe lanes are fixed at one worker each (a self-check is seconds, an E@10
// probe ~10 epochs); the training lane width comes from config.
const (
	probeSelfCap = 1
	probeE10Cap  = 1
)

func main() {
	// On macOS with a GUI session, tray.Main parks the main thread in the
	// AppKit run loop and runs the daemon body on a goroutine; everywhere else
	// it is a plain inline call. exit is atomic because the write happens on
	// the body goroutine and the read on the main one after the loop returns.
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

	lg, err := applog.Open(cfg.LogPath())
	if err != nil {
		return err
	}
	defer lg.Close()
	lg.Printf("starting namtrainerd %s (pid %d), binding %s, cap %d, data_dir %s",
		buildinfo.Version, os.Getpid(), cfg.Addr(), cfg.Cap, cfg.DataDir)

	st, err := store.Open(rootCtx, cfg.DBPath())
	if err != nil {
		lg.Printf("FATAL: open database: %v", err)
		return err
	}
	defer st.Close()

	srv := httpapi.New(cfg, st, lg)

	// The capture signal (the trainer --input) is downloaded + sha-verified during
	// provisioning; the worker only spawns a trainer once ready, by which point it
	// is present.
	signalPath := runtime.SignalPath(cfg.RuntimeDir())

	// ready flips true once the python runtime is provisioned; workers idle until then.
	var ready atomic.Bool

	// Keep the machine awake while the queue has work, so a laptop that idle-sleeps
	// doesn't freeze a training run mid-queue (an overnight backlog otherwise barely
	// advances). Released the moment the queue drains. No-op when keep_awake=false.
	keeper := awake.New(cfg.KeepAwake, lg.Printf)
	defer keeper.Close()

	var pool *worker.Pool // declared ahead: OnCounts below reads pool.Paused()
	pool = worker.New(worker.Options{
		Store: st,
		Log:   lg,
		Runner: worker.ProcessRunner{
			Python: runtime.VenvPython(cfg.RuntimeDir()),
			Driver: runtime.DriverPath(cfg.RuntimeDir()),
		},
		SignalPath:  signalPath,
		ScratchRoot: cfg.ScratchDir(),
		Cap:         cfg.Cap,       // training lane (GPU-bound), live-adjustable
		CapLimit:    config.MaxCap, // spawn width: SetCap can raise up to this without a restart
		// Probe lanes run alongside training so a rig-side self-ESR verdict is
		// seconds away even during a long train. One worker each is plenty: a
		// self-check is seconds (kill-on-verdict), an E@10 probe ~10 epochs.
		ProbeSelfCap: probeSelfCap,
		ProbeE10Cap:  probeE10Cap,
		OnCounts: func(running, queued int) {
			srv.SetCounts(running, queued)
			// Hold while anything RUNS (a draining pause must not let the lid
			// freeze a train mid-epoch), or while queued work is claimable.
			// Fully paused with only queued work → release: a pause means the
			// machine is the musician's again, sleep included. pool is bound
			// below; publishes only happen after Start.
			keeper.Set(running > 0 || (queued > 0 && !pool.Paused()))
		},
		OnAvgSPerEpoch: srv.SetAvgSPerEpoch,
		Ready:          ready.Load,
	})
	srv.SetKiller(pool)
	srv.SetCapper(pool)
	srv.SetLiveExporter(pool)
	srv.SetNotifier(pool.Notify)
	srv.SetAPICapAllowed(cfg.AllowAPICap)

	ctx, stop := signal.NotifyContext(rootCtx, os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Wire the menu-bar controls and start the title/list refresher (skipped
	// entirely when headless — no point polling the store for a no-op Handle).
	// Pause lives in the pool only — no HTTP surface, the §3 contract does not
	// move; a paused daemon still reports truthfully. Each control kicks an
	// immediate refresh so the icon and items flip on click, not a tick later.
	if trayHandle.Live() {
		kick := make(chan struct{}, 1)
		nudge := func() {
			select {
			case kick <- struct{}{}:
			default:
			}
		}
		// persistDynamic writes BOTH runtime-mutable fields from their live
		// state, so persisting one never reverts the other; the boot cfg the
		// rest of the daemon reads stays untouched.
		persistDynamic := func() {
			updated := *cfg
			updated.Cap = pool.Cap()
			updated.AllowAPICap = srv.APICapAllowed()
			if err := updated.Save(); err != nil {
				lg.Printf("tray: persist config: %v", err)
			}
		}
		trayHandle.SetControls(tray.Controls{
			PauseNow:          func() { pool.Pause(true); nudge() },
			PauseAfterCurrent: func() { pool.Pause(false); nudge() },
			Resume:            func() { pool.Resume(); nudge() },
			// Restart = graceful stop; under launchd (KeepAlive) that is a
			// config re-read — the agent relaunches us in seconds. Run by
			// hand, it simply stops the daemon (documented in the README).
			Restart: func() {
				lg.Printf("tray: restart requested (re-read config)")
				stop()
			},
			// SetCap applies LIVE (pool.SetCap — same path as PATCH /v1/cap; no
			// restart, nothing killed) and persists so the next boot keeps it.
			SetCap: func(n int) {
				if n == pool.Cap() {
					return
				}
				pool.SetCap(n)
				persistDynamic()
				nudge()
			},
			// ToggleAPICap flips whether clients may PATCH /v1/cap (persisted;
			// the admin decision lives on this machine, not with the caller).
			ToggleAPICap: func() {
				v := !srv.APICapAllowed()
				srv.SetAPICapAllowed(v)
				lg.Printf("tray: cap via API %s", map[bool]string{true: "allowed", false: "disallowed"}[v])
				persistDynamic()
				nudge()
			},
		})
		trayHandle.SetCap(pool.Cap())
		trayHandle.SetAPICapAllowed(srv.APICapAllowed())
		go trayLoop(ctx, trayHandle, st, pool, srv, kick)
	}

	if err := pool.Start(ctx); err != nil {
		lg.Printf("FATAL: start worker pool: %v", err)
		return err
	}
	defer pool.Stop()

	// Provision the runtime in the background so the daemon is live (accepting +
	// queuing jobs) while python + the trainer install. Jobs start the moment it
	// is ready.
	go provisionLoop(ctx, cfg, lg, srv, &ready, pool)

	// Age-GC the re-download window (model blobs), at start and daily.
	go gcLoop(ctx, cfg, lg, st)

	httpSrv := &http.Server{
		Addr:              cfg.Addr(),
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	errc := make(chan error, 1)
	go func() {
		lg.Printf("http listening on %s", cfg.Addr())
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errc <- err
		}
	}()

	select {
	case err := <-errc:
		lg.Printf("FATAL: http server: %v", err)
		return err
	case <-ctx.Done():
		lg.Printf("shutdown signal received")
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		lg.Printf("http shutdown error: %v", err)
	}
	pool.Stop() // join workers (in-flight jobs are requeued, not failed)
	lg.Printf("stopped")
	return nil
}

// trayLoop drives the menu-bar status item from one queue snapshot every few
// seconds: the title (running/queued counts, the clock-time ETA for every lane
// to drain, the same moving-average s/epoch /v1/health reports), the dropdown
// queue list, and the pause/resume item state. A store error just leaves the
// display as-is until the next tick; the status item vanishes with the
// process, so shutdown needs no cleanup here.
func trayLoop(ctx context.Context, h tray.Handle, st *store.Store, pool *worker.Pool, srv *httpapi.Server, kick <-chan struct{}) {
	const maxRows = 12 // mirrors the menu's pre-created slots
	update := func() {
		running, queued, remaining, err := st.QueueTotals(ctx)
		if err != nil {
			return
		}
		avg, err := st.AvgSPerEpoch(ctx)
		if err != nil {
			return
		}
		var etaSecs *float64
		if avg != nil {
			secs := tray.QueueSeconds(remaining, *avg, pool.Cap(), probeSelfCap, probeE10Cap)
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
			list[i] = tray.QueueRow{Running: r.Running, Kind: r.Kind, Epochs: r.Epochs, Epoch: r.Epoch, Key: r.Key}
		}
		h.SetQueue(list, running+queued-len(rows))
		h.SetPaused(tray.DeriveState(pool.Paused(), running))
		h.SetCap(pool.Cap()) // dynamic: an API cap change shows in the menu a tick later
		h.SetAPICapAllowed(srv.APICapAllowed())
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
// it publishes the resolved profile, flips ready, and wakes the workers.
func provisionLoop(ctx context.Context, cfg *config.Config, lg *applog.Logger,
	srv *httpapi.Server, ready *atomic.Bool, pool *worker.Pool) {
	backoff := 5 * time.Second
	const maxBackoff = 2 * time.Minute
	for {
		lg.Printf("provisioning runtime at %s", cfg.RuntimeDir())
		prof, err := runtime.Provision(ctx, cfg.RuntimeDir(), func(s string) {
			lg.Printf("provision: %s", s)
		})
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			lg.Printf("provisioning failed: %v (retry in %s)", err, backoff)
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
		srv.SetProfile(httpapi.Profile{
			Ready:        true,
			Python:       prof.Python,
			Nam:          prof.Nam,
			GPU:          prof.GPU,
			DriverSHA256: prof.DriverSHA256,
			SignalSHA256: prof.SignalSHA256,
		})
		ready.Store(true)
		pool.Notify()
		lg.Printf("runtime ready: python=%s nam=%s driver=%s…", prof.Python, prof.Nam, prof.DriverSHA256[:12])
		return
	}
}

// gcLoop runs a GC/vacuum pass at start and once a day. When retention_days > 0 it
// expires re-downloadable model blobs older than the window; retention_days == 0
// (keep forever, the default) skips that step entirely. The incremental vacuum
// ALWAYS runs — it is the DB's only page reclaim, so freed wav pages return to the
// OS even when nothing was expired. Job rows and per-job logs are history and are
// never GC'd here — only the blob and the freed pages.
func gcLoop(ctx context.Context, cfg *config.Config, lg *applog.Logger, st *store.Store) {
	gcPass(ctx, cfg, lg, st)
	t := time.NewTicker(24 * time.Hour)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			gcPass(ctx, cfg, lg, st)
		}
	}
}

// gcPass is one GC/vacuum cycle (the split above, one pass). The expiry step is
// gated on retention_days > 0; the incremental vacuum runs unconditionally and its
// story-log line makes that observable — the vacuum line always appears, the
// "expired …" line only when retention is enabled and something was actually freed.
func gcPass(ctx context.Context, cfg *config.Config, lg *applog.Logger, st *store.Store) {
	if cfg.RetentionDays > 0 {
		cutoff := time.Now().Add(-time.Duration(cfg.RetentionDays) * 24 * time.Hour).Unix()
		if n, err := st.GCExpiredModels(ctx, cutoff); err != nil {
			lg.Printf("gc: expire models: %v", err)
		} else if n > 0 {
			lg.Printf("gc: expired %d model blob(s) past %d-day retention", n, cfg.RetentionDays)
		}
	}
	// Always reclaim: a terminal finish frees ~27 MB of wav pages regardless of
	// whether any model expired, and incremental_vacuum is the only thing that
	// returns them.
	if err := st.IncrementalVacuum(ctx); err != nil {
		lg.Printf("gc: incremental vacuum: %v", err)
	} else {
		lg.Printf("gc: incremental vacuum done")
	}
}
