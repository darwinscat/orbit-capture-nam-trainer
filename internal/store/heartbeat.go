// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package store

import (
	"context"
	"fmt"
)

// WorkerInfo is one heartbeat: the workers row as this daemon sees itself (what
// used to be GET /v1/health). Name is the hostname (one daemon per box), Instance
// a fresh uuid per process start.
type WorkerInfo struct {
	Name          string
	Instance      string
	Version       string
	NamVersion    string // "" until provisioned → NULL
	DriverSHA256  string
	SignalSHA256  string
	GPU           string
	Python        string
	SchemaVersion int    // the queue_contract this build supports
	Note          string // why it is not ready, in words; "" → NULL
	TrainCap      int
	ProbeCap      int
	Running       int
	Paused        bool
	Ready         bool
	AvgSPerEpoch  *float64
	DiskFreeBytes *int64
}

// Heartbeat UPSERTs the workers row and returns what the app has ASKED for: a
// train_cap, and whether to pause (nil each when there is no ask). The queue lives
// in the database, so a pause kept inside one app would pause nobody — the daemon
// would go on draining the queue. The app's columns are never touched by the upsert.
// PauseManner is what somebody is asking of this trainer — see migration 0005. Three of them stop it
// and they are not interchangeable: "after" lets an eight-hundred-epoch run finish, "now" throws it
// away, and "keep" stops at the end of the current EPOCH with the best weights kept, which is what a
// person who needs their own GPU back is asking for. Empty = nothing asked.
const (
	PauseAfter = "after"
	PauseKeep  = "keep"
	PauseNow   = "now"
	Resume     = "resume"
)

func (s *Store) Heartbeat(ctx context.Context, w WorkerInfo) (capWanted *int, pauseWanted *string, err error) {
	err = s.pool.QueryRow(ctx,
		`INSERT INTO workers (name, instance, version, nam_version, driver_sha256, signal_sha256, gpu, python,
		                      schema_version, note, train_cap, probe_cap, running, paused, ready,
		                      avg_s_per_epoch, disk_free_bytes, last_seen_at)
		 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, now())
		 ON CONFLICT (name) DO UPDATE SET
		   instance = EXCLUDED.instance, version = EXCLUDED.version, nam_version = EXCLUDED.nam_version,
		   driver_sha256 = EXCLUDED.driver_sha256, signal_sha256 = EXCLUDED.signal_sha256,
		   gpu = EXCLUDED.gpu, python = EXCLUDED.python, schema_version = EXCLUDED.schema_version,
		   note = EXCLUDED.note, train_cap = EXCLUDED.train_cap, probe_cap = EXCLUDED.probe_cap,
		   running = EXCLUDED.running, paused = EXCLUDED.paused, ready = EXCLUDED.ready,
		   avg_s_per_epoch = EXCLUDED.avg_s_per_epoch, disk_free_bytes = EXCLUDED.disk_free_bytes,
		   last_seen_at = now()
		 RETURNING train_cap_wanted, pause_wanted`,
		w.Name, w.Instance, strArg(w.Version), strArg(w.NamVersion), shaArg(w.DriverSHA256), shaArg(w.SignalSHA256),
		strArg(w.GPU), strArg(w.Python), w.SchemaVersion, strArg(w.Note), clampCap(w.TrainCap), clampCap(w.ProbeCap),
		max(w.Running, 0), w.Paused, w.Ready, positiveArg(w.AvgSPerEpoch), w.DiskFreeBytes).Scan(&capWanted, &pauseWanted)
	if err != nil {
		return nil, nil, fmt.Errorf("heartbeat: %w", err)
	}
	return capWanted, pauseWanted, nil
}

// RequestStopKeep asks THIS worker's own running train jobs to stop at the end of the current epoch
// and keep the best weights. It writes the very column the app writes for the same gesture, so the
// existing control path does the rest — and anybody watching the queue sees one thing, whoever asked.
//
// The run becomes a normal early-stopped model (reached < epochs) that "Continue" can pick up later,
// on this machine or on another one: the checkpoint travels through the library, not the disk. That
// is the whole difference from killing it, and it is why this is the pause a person who needs their
// own GPU back actually wants.
func (s *Store) RequestStopKeep(ctx context.Context, worker string) (int, error) {
	ct, err := s.pool.Exec(ctx,
		`UPDATE jobs SET stop_requested_at = COALESCE(stop_requested_at, now()),
		                 stop_reason       = COALESCE(stop_reason, 'pause')
		  WHERE worker = $1 AND state = 'running' AND lane = 'train'`, worker)
	if err != nil {
		return 0, fmt.Errorf("request stop-and-keep: %w", err)
	}
	return int(ct.RowsAffected()), nil
}

// ConsumeCapWanted clears the app's ask once it has been applied — guarded by the
// value, so an ask that changed in between is not lost.
func (s *Store) ConsumeCapWanted(ctx context.Context, name string, applied int) error {
	_, err := s.pool.Exec(ctx,
		`UPDATE workers SET train_cap_wanted = NULL WHERE name = $1 AND train_cap_wanted = $2`, name, applied)
	if err != nil {
		return fmt.Errorf("consume cap wanted: %w", err)
	}
	return nil
}

// ConsumePauseWanted clears the ask once applied, guarded by the value so an ask that changed in
// between is not swallowed.
func (s *Store) ConsumePauseWanted(ctx context.Context, name string, applied string) error {
	_, err := s.pool.Exec(ctx,
		`UPDATE workers SET pause_wanted = NULL WHERE name = $1 AND pause_wanted = $2`, name, applied)
	if err != nil {
		return fmt.Errorf("consume pause wanted: %w", err)
	}
	return nil
}

// clampCap keeps a cap inside the schema's 1..8 CHECK.
func clampCap(n int) int {
	if n < 1 {
		return 1
	}
	if n > 8 {
		return 8
	}
	return n
}

// positiveArg renders a non-positive average as NULL (the column CHECKs > 0).
func positiveArg(v *float64) *float64 {
	if v == nil || !(*v > 0) {
		return nil
	}
	return v
}
