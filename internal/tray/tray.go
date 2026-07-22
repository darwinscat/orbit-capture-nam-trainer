// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package tray shows the daemon's queue in the macOS menu bar: an icon plus a
// "2/20 13:36 5.14" title (running/queued, clock-time ETA for the queue to
// drain, moving-average seconds per epoch — the same number /v1/health
// reports). Display only: policy stays with the HTTP clients. On Linux, in a
// CGO_ENABLED=0 build, with ONCT_NO_TRAY set, or in a session with no window
// server, Main is a plain pass-through and the daemon stays fully headless.
package tray

import (
	"fmt"
	"time"

	"orbit-capture-nam-trainer/internal/jobs"
)

// QueueRow is one line of the dropdown queue list.
type QueueRow struct {
	Running bool
	Kind    string // raw job kind: train | train_more | probe_self | probe_e10
	Epochs  int64
	Epoch   *int64 // running: last reported 0-based epoch; nil until one prints
	Key     string // the content sha256 hex
}

// Controls are the daemon actions behind the menu items. Headless they are
// never invoked; nil funcs are simply ignored.
type Controls struct {
	PauseNow          func() // stop claiming AND kill running jobs (they requeue)
	PauseAfterCurrent func() // stop claiming; running jobs finish
	Resume            func()
	Restart           func()      // graceful stop; under launchd (KeepAlive) that re-reads config
	SetCap            func(n int) // resize the train lane LIVE and persist cap=n to config.toml
	ToggleAPICap      func()      // flip the PATCH /v1/cap permission gate and persist it
}

// PauseState is what the icon and the menu items reflect. Paused-but-draining
// (a "pause after current" with a job still finishing) keeps "Pause now"
// available as the escalation that kills the straggler.
type PauseState int

const (
	StateActive         PauseState = iota // claiming; template icon
	StatePausedDraining                   // gate closed, jobs still running; orange
	StatePaused                           // gate closed, nothing running; red
)

// DeriveState maps the pool gate + the live running count to the tray state.
func DeriveState(paused bool, running int) PauseState {
	switch {
	case !paused:
		return StateActive
	case running > 0:
		return StatePausedDraining
	default:
		return StatePaused
	}
}

// Handle updates the menu-bar status item. The headless implementation is a
// no-op; the macOS implementation is safe to call from any goroutine.
type Handle interface {
	Live() bool // false → headless no-op; skip the refresh loop
	SetTitle(title string)
	SetQueue(rows []QueueRow, moreQueued int) // list + "… N more" overflow count
	SetPaused(s PauseState)                   // reflects the pool gate in the menu + icon
	SetCap(current int)                       // check-marks the active cap in the submenu
	SetAPICapAllowed(allowed bool)            // check-marks the "Allow cap via API" toggle
	SetControls(c Controls)                   // wire the menu clicks; call once
}

// noTray is the headless Handle.
type noTray struct{}

func (noTray) Live() bool               { return false }
func (noTray) SetTitle(string)          {}
func (noTray) SetQueue([]QueueRow, int) {}
func (noTray) SetPaused(PauseState)     {}
func (noTray) SetCap(int)               {}
func (noTray) SetAPICapAllowed(bool)    {}
func (noTray) SetControls(Controls)     {}

// QueueSeconds estimates the wall seconds until every lane drains. Lanes run
// concurrently, so it is the max over lanes of remaining-epochs × sPerEpoch ÷
// lane cap. An estimate, not a bound: exact serial work at cap 1 (probes
// overcosted at the training s/epoch — a self-check really runs seconds); at
// cap>1 the division assumes epochs split evenly across workers, which an
// atomic job can beat (same caveat QueueView documents for epochs_ahead).
func QueueSeconds(remaining map[string]int64, sPerEpoch float64, trainCap, probeSelfCap, probeE10Cap int) float64 {
	lane := func(kind string, workers int) float64 {
		if workers < 1 {
			workers = 1
		}
		return float64(remaining[kind]) * sPerEpoch / float64(workers)
	}
	secs := lane(jobs.KindTrain, trainCap)
	if s := lane(jobs.KindProbeSelf, probeSelfCap); s > secs {
		secs = s
	}
	if s := lane(jobs.KindProbeE10, probeE10Cap); s > secs {
		secs = s
	}
	return secs
}

// Format renders the title. Idle (nothing running or queued) is "" so the menu
// bar shows just the icon. Otherwise "running/total" — 2/4 reads "2 of the 4
// jobs in the queue are running" — then the ETA as clock time when known
// ("24h+" once it stops fitting on today's clock), then the average s/epoch
// when known — each part simply omitted until it exists.
func Format(now time.Time, running, queued int, etaSecs, sPerEpoch *float64) string {
	if running == 0 && queued == 0 {
		return ""
	}
	title := fmt.Sprintf("%d/%d", running, running+queued)
	if etaSecs != nil {
		if d := time.Duration(*etaSecs * float64(time.Second)); d >= 24*time.Hour {
			title += " 24h+"
		} else {
			title += " " + now.Add(d).Format("15:04")
		}
	}
	if sPerEpoch != nil {
		title += fmt.Sprintf(" %.2f", *sPerEpoch)
	}
	return title
}

// FormatRow renders one queue-list menu line: a running job as
// "▶ train 42/300 cbd531ab" (1-based epoch, "–" before the first one prints),
// a queued one as "train 300 ep cbd531ab". The short key is what a caller
// would recognize from its own job URLs; the daemon knows no names.
func FormatRow(r QueueRow) string {
	key := r.Key
	if len(key) > 8 {
		key = key[:8]
	}
	if r.Running {
		ep := "–"
		if r.Epoch != nil {
			ep = fmt.Sprintf("%d", *r.Epoch+1)
		}
		return fmt.Sprintf("▶ %s %s/%d %s", r.Kind, ep, r.Epochs, key)
	}
	return fmt.Sprintf("%s %d ep %s", r.Kind, r.Epochs, key)
}
