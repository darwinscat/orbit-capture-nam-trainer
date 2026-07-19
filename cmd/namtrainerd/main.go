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
	"orbit-capture-nam-trainer/internal/buildinfo"
	"orbit-capture-nam-trainer/internal/config"
	"orbit-capture-nam-trainer/internal/httpapi"
	"orbit-capture-nam-trainer/internal/runtime"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/worker"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "namtrainerd:", err)
		os.Exit(1)
	}
}

func run() error {
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

	pool := worker.New(worker.Options{
		Store: st,
		Log:   lg,
		Runner: worker.ProcessRunner{
			Python: runtime.VenvPython(cfg.RuntimeDir()),
			Driver: runtime.DriverPath(cfg.RuntimeDir()),
		},
		SignalPath:  signalPath,
		ScratchRoot: cfg.ScratchDir(),
		Cap:         cfg.Cap, // training lane (GPU-bound)
		// Probe lanes run alongside training so a rig-side self-ESR verdict is
		// seconds away even during a long train. One worker each is plenty: a
		// self-check is seconds (kill-on-verdict), an E@10 probe ~10 epochs.
		ProbeSelfCap:   1,
		ProbeE10Cap:    1,
		OnCounts:       srv.SetCounts,
		OnAvgSPerEpoch: srv.SetAvgSPerEpoch,
		Ready:          ready.Load,
	})
	srv.SetKiller(pool)
	srv.SetNotifier(pool.Notify)

	ctx, stop := signal.NotifyContext(rootCtx, os.Interrupt, syscall.SIGTERM)
	defer stop()

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

// gcLoop expires re-downloadable model blobs older than retention_days and runs
// an incremental vacuum, at start and once a day. Job rows and per-job logs are
// history and are never GC'd here — only the blob and the freed pages.
func gcLoop(ctx context.Context, cfg *config.Config, lg *applog.Logger, st *store.Store) {
	runGC := func() {
		cutoff := time.Now().Add(-time.Duration(cfg.RetentionDays) * 24 * time.Hour).Unix()
		n, err := st.GCExpiredModels(ctx, cutoff)
		if err != nil {
			lg.Printf("gc: %v", err)
			return
		}
		if err := st.IncrementalVacuum(ctx); err != nil {
			lg.Printf("gc: incremental vacuum: %v", err)
		}
		if n > 0 {
			lg.Printf("gc: expired %d model blob(s) past %d-day retention, vacuumed", n, cfg.RetentionDays)
		}
	}
	runGC()
	t := time.NewTicker(24 * time.Hour)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			runGC()
		}
	}
}
