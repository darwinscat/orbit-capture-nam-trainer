// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package tray shows the shared queue in the macOS menu bar: an icon plus a
// "2/20 13:36 5.14" title (running/queued, clock-time ETA for the queue to
// drain, moving-average seconds per epoch — the same number the heartbeat
// reports in workers.avg_s_per_epoch). Display only: policy stays with the app.
// On Linux, in a CGO_ENABLED=0 build, with ONCT_NO_TRAY set, or in a session with
// no window server, Main is a plain pass-through and the daemon stays fully
// headless.
package tray

import (
	"fmt"
	"time"
)

// QueueRow is one line of the dropdown queue list.
type QueueRow struct {
	Running bool
	Kind    string // raw job kind: train | train_more | probe_self
	Epochs  int64
	Epoch   *int64 // running: last reported 0-based epoch; nil until one prints
	Label   string // jobs.take_label — "RAT 2-0008"
}

// Controls are the daemon actions behind the menu items. Headless they are
// never invoked; nil funcs are simply ignored.
type Controls struct {
	PauseNow          func() // stop claiming; stop running jobs AT ONCE, keeping up to the last epoch
	PauseAfterCurrent func() // stop claiming; running jobs finish their full epoch count
	Resume            func()
	Restart           func()      // graceful stop; under launchd (KeepAlive) that re-reads config
	SetCap            func(n int) // resize the train lane LIVE and persist cap=n to config.toml
}

// PauseState is what the icon and the menu items reflect. Paused-but-draining is "pause after
// current" with a job still finishing — which can be an hour and a half — so "Pause now" stays
// available as the escalation. It no longer KILLS the straggler: it stops it keeping everything up to
// the last completed epoch, so escalating costs seconds of GPU rather than the run.
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
	SetControls(c Controls)                   // wire the menu clicks; call once
}

// noTray is the headless Handle.
type noTray struct{}

func (noTray) Live() bool               { return false }
func (noTray) SetTitle(string)          {}
func (noTray) SetQueue([]QueueRow, int) {}
func (noTray) SetPaused(PauseState)     {}
func (noTray) SetCap(int)               {}
func (noTray) SetControls(Controls)     {}

// MineSeconds estimates the wall seconds until THIS trainer is done with what it
// is holding right now: its runs go on at the same time, so it is the LONGEST of
// them — remaining epochs × this box's own seconds per epoch. `remaining` is one
// entry per running job of this worker (jobs.LaneTrain only; a self-check is
// seconds and never the thing anybody waits for).
//
// Queued work is deliberately not counted. The queue is shared: which box claims
// the next row is decided when it is claimed, and by whom decides how fast it
// goes. A title that adds it in is promising somebody else's time.
func MineSeconds(remaining []int64, sPerEpoch float64) float64 {
	var longest int64
	for _, r := range remaining {
		if r > longest {
			longest = r
		}
	}
	return float64(longest) * sPerEpoch
}

// Format renders the title: this machine's own load, "running/cap" — 1/1 is a box
// at cap 1 with a run on it, 3/8 is three of eight lanes busy. Idle (nothing of mine
// running) is "" so the menu bar shows just the icon.
//
// It used to be "running/total" over the WHOLE shared queue, which on a second trainer
// read as this box's own business and was not: 2/4 while this machine held one run.
// Then the ETA as clock time when known ("24h+" once it stops fitting on today's
// clock), then the average s/epoch when known — each part simply omitted until it
// exists.
func Format(now time.Time, running, cap int, etaSecs, sPerEpoch *float64) string {
	if running == 0 {
		return ""
	}
	if cap < running {
		cap = running // a cap lowered under what is already going: never draw 2/1
	}
	title := fmt.Sprintf("%d/%d", running, cap)
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
// "▶ train 42/300 RAT 2-0008" (1-based epoch, "–" before the first one prints),
// a queued one as "train 300 ep RAT 2-0008" — the take's label, the handle
// people use.
//
// Every line is THIS trainer's own run — the shared queue is the app's view, not a
// menu bar's, and a machine's menu answers "what am I doing".
func FormatRow(r QueueRow) string {
	if r.Running {
		ep := "–"
		if r.Epoch != nil {
			ep = fmt.Sprintf("%d", *r.Epoch+1)
		}
		return fmt.Sprintf("▶ %s %s/%d %s", r.Kind, ep, r.Epochs, r.Label)
	}
	return fmt.Sprintf("%s %d ep %s", r.Kind, r.Epochs, r.Label)
}
