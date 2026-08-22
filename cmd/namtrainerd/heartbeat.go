// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package main

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"orbit-capture-nam-trainer/internal/applog"
	"orbit-capture-nam-trainer/internal/buildinfo"
	"orbit-capture-nam-trainer/internal/config"
	"orbit-capture-nam-trainer/internal/runtime"
	"orbit-capture-nam-trainer/internal/store"
	"orbit-capture-nam-trainer/internal/sysutil"
	"orbit-capture-nam-trainer/internal/worker"
)

// HeartbeatInterval is how often the workers row is rewritten. The app treats a
// worker as usable while last_seen_at is within 15 s.
const HeartbeatInterval = 5 * time.Second

// heartbeat is the daemon's self-report: every few seconds it UPSERTs the workers
// row (what used to be GET /v1/health), checks library.queue_contract against what
// this build knows (the schema gate: ready=false + a note when the library moved
// past us — claims stop, running jobs finish), and applies the app's
// train_cap_wanted (what used to be PATCH /v1/cap).
type heartbeat struct {
	cfg      *config.Config
	lg       *applog.Logger
	st       *store.Store
	pool     *worker.Pool
	name     string
	instance string

	profile  atomic.Pointer[runtime.Profile] // set once provisioning succeeds
	schemaOK atomic.Bool
	note     atomic.Value // string: why not ready, while provisioning
	lastNote atomic.Value // string: the last note written, for transition logging
}

func (h *heartbeat) setProfile(p runtime.Profile) { h.profile.Store(&p) }
func (h *heartbeat) setNote(s string)             { h.note.Store(s) }

// ready is the pool's claim gate: provisioned AND the schema is the one we know.
func (h *heartbeat) ready() bool { return h.profile.Load() != nil && h.schemaOK.Load() }

// provenance is what the pool stamps on every run (and filters claims by).
func (h *heartbeat) provenance() (worker.Provenance, bool) {
	p := h.profile.Load()
	if p == nil {
		return worker.Provenance{}, false
	}
	return worker.Provenance{NamVersion: p.Nam, DriverSHA256: p.DriverSHA256, SignalSHA256: p.SignalSHA256}, true
}

// loop beats every HeartbeatInterval until ctx ends.
func (h *heartbeat) loop(ctx context.Context) {
	t := time.NewTicker(HeartbeatInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := h.beat(ctx); err != nil && ctx.Err() == nil {
				h.lg.Printf("heartbeat: %v", err)
			}
		}
	}
}

// beat is one heartbeat.
func (h *heartbeat) beat(ctx context.Context) error {
	bctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()

	note := ""
	contract, err := h.st.QueueContract(bctx)
	switch {
	case err != nil:
		h.schemaOK.Store(false)
		note = "library: " + err.Error()
	case contract != store.SupportedQueueContract:
		h.schemaOK.Store(false)
		note = fmt.Sprintf("schema moved to %d, daemon knows %d", contract, store.SupportedQueueContract)
	default:
		h.schemaOK.Store(true)
	}
	prof := h.profile.Load()
	if note == "" && prof == nil {
		note, _ = h.note.Load().(string)
		if note == "" {
			note = "provisioning runtime"
		}
	}
	if last, _ := h.lastNote.Load().(string); last != note {
		h.lastNote.Store(note)
		if note != "" {
			h.lg.Printf("not ready: %s", note)
		} else {
			h.lg.Printf("ready: worker %s reports for duty (schema %d)", h.name, store.SupportedQueueContract)
		}
	}

	info := store.WorkerInfo{
		Name:          h.name,
		Instance:      h.instance,
		Version:       buildinfo.Version,
		DriverSHA256:  runtime.DriverSHA256(),
		SignalSHA256:  runtime.SignalSHA256,
		SchemaVersion: store.SupportedQueueContract,
		Note:          note,
		TrainCap:      h.pool.Cap(),
		ProbeCap:      probeCap,
		Paused:        h.pool.Paused(),
		Ready:         note == "",
	}
	if prof != nil {
		info.NamVersion, info.GPU, info.Python = prof.Nam, prof.GPU, prof.Python
	}
	if running, _, err := h.st.CountByState(bctx, h.name); err == nil {
		info.Running = running
	} else {
		info.Running = h.pool.Running()
	}
	if avg, err := h.st.AvgSPerEpoch(bctx, h.name); err == nil {
		info.AvgSPerEpoch = avg
	}
	if free, err := sysutil.FreeBytes(h.cfg.DataDir); err == nil {
		n := int64(free)
		info.DiskFreeBytes = &n
	}

	wanted, pause, err := h.st.Heartbeat(bctx, info)
	if err != nil {
		return err
	}
	// SOMEBODY ASKS THIS TRAINER TO STOP, and says how. Three of them stop it and they are not
	// interchangeable — see migration 0005. The app may ask any worker by name; the tray asks this
	// one directly and does not come through here.
	if pause != nil {
		h.lg.Printf("asked to %q — applying", *pause)
		switch *pause {
		case store.PauseAfter:
			h.pool.Pause(false) // stop claiming; what is running runs to its full epoch count
		case store.PauseKeep:
			h.pool.PauseKeep(bctx) // stop claiming; end this epoch, keep the best
		case store.PauseNow:
			h.pool.Pause(true) // stop claiming; kill it — the epochs are lost and it requeues
		case store.Resume:
			h.pool.Resume()
		default:
			h.lg.Printf("unknown pause ask %q — ignored", *pause)
		}
		if err := h.st.ConsumePauseWanted(bctx, h.name, *pause); err != nil {
			h.lg.Printf("heartbeat: %v", err)
		}
	}
	if wanted != nil {
		if *wanted != h.pool.Cap() {
			h.lg.Printf("app asks cap %d (was %d) — applying", *wanted, h.pool.Cap())
			h.pool.SetCap(*wanted)
			h.persistCap()
		}
		if err := h.st.ConsumeCapWanted(bctx, h.name, *wanted); err != nil {
			h.lg.Printf("heartbeat: %v", err)
		}
	}
	return nil
}

// persistCap writes the live cap to config.toml so the next boot keeps it.
func (h *heartbeat) persistCap() {
	updated := *h.cfg
	updated.Cap = h.pool.Cap()
	if err := updated.Save(); err != nil {
		h.lg.Printf("persist cap: %v", err)
	}
}

// farewell is the last beat at shutdown: ready=false, running=0, so the app sees
// the worker gone now rather than after the 15 s last_seen_at window.
func (h *heartbeat) farewell() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	info := store.WorkerInfo{
		Name: h.name, Instance: h.instance, Version: buildinfo.Version,
		DriverSHA256: runtime.DriverSHA256(), SignalSHA256: runtime.SignalSHA256,
		SchemaVersion: store.SupportedQueueContract, Note: "stopped",
		TrainCap: h.pool.Cap(), ProbeCap: probeCap, Paused: h.pool.Paused(), Ready: false,
	}
	if p := h.profile.Load(); p != nil {
		info.NamVersion, info.GPU, info.Python = p.Nam, p.GPU, p.Python
	}
	if _, _, err := h.st.Heartbeat(ctx, info); err != nil {
		h.lg.Printf("farewell heartbeat: %v", err)
	}
}
