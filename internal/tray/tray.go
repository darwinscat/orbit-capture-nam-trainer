// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package tray shows this machine's own work in the macOS menu bar: an icon plus a
// "1/1" title — the runs it holds against its train cap, and nothing else — with what the
// box has computed in its life at the foot of the menu. Display only: policy stays with
// the app.
// On Linux, in a CGO_ENABLED=0 build, with ONCT_NO_TRAY set, or in a session with
// no window server, Main is a plain pass-through and the daemon stays fully
// headless.
package tray

import (
	"fmt"
	"strconv"
	"strings"
)

// QueueRow is one line of the dropdown queue list.
type QueueRow struct {
	Running bool
	Kind    string // raw job kind: train | train_more | probe_self
	Epochs  int64
	Epoch   *int64 // running: last reported 0-based epoch; nil until one prints
	Label   string // jobs.take_label — "RAT 2-0008"
}

// Tally is what THIS machine has computed, ever — the one thing in the menu that is not
// about right now. Epochs is the total: every epoch row the library holds for this box,
// plus its self-checks, which are one epoch each. Hours is the time its epochs took, SUMMED
// PER RUN — two runs sharing the GPU for an hour are two hours here, because that is two
// hours of training done — and SPerEpoch is the mean of the same seconds, which is what one
// epoch costs this box as it actually runs them.
type Tally struct {
	Epochs    int64
	Probes    int64
	Hours     int64
	SPerEpoch float64
}

// Setup is what the settings sheet shows and returns: where the shared library is,
// and whether this machine may sleep while the queue has work. The cap is not here —
// it is one click in the menu and applies live, which is a different kind of answer.
type Setup struct {
	Host      string
	Port      int
	Database  string
	User      string
	Password  string
	Schema    string
	KeepAwake bool
}

// Controls are the daemon actions behind the menu items. Headless they are
// never invoked; nil funcs are simply ignored — so a Controls wired before everything it
// needs exists is a menu that is PARTLY alive, not a broken one, and a later SetControls
// replaces it wholesale.
type Controls struct {
	PauseNow          func() // stop claiming; stop running jobs AT ONCE, keeping up to the last epoch
	PauseAfterCurrent func() // stop claiming; running jobs finish their full epoch count
	Resume            func()
	Restart           func()      // graceful stop; under launchd (KeepAlive) that re-reads config
	OpenSetup         func()      // show the settings sheet (macOS); nil where there is no window
	SetCap            func(n int) // resize the train lane LIVE and persist cap=n to config.toml
}

// PauseState is what the icon and the menu items reflect. Paused-but-draining is "pause after
// current" with a job still finishing — which can be an hour and a half — so "Pause now" stays
// available as the escalation. It no longer KILLS the straggler: it stops it keeping everything up to
// the last completed epoch, so escalating costs seconds of GPU rather than the run.
type PauseState int

// COLOUR MEANS SOMETHING IS WRONG OR HELD, and nothing else. Working and idle-but-
// ready are the SAME normal state: the plain template icon, with the title beside it
// saying whether anything is going on. Red used to mean "paused, nothing running",
// which is a machine quietly waiting — read across a room it says failure, and it was
// the state a trainer sat in most of the time.
const (
	StateActive         PauseState = iota // gate open — working or ready to; template icon
	StatePausedDraining                   // gate closed, a run still finishing; yellow
	StatePaused                           // gate closed, nothing running; orange
	StateNoLibrary                        // the library does not answer; red — the one real fault
)

// DeriveState maps the pool gate + the live running count to the tray state. The
// library being unreachable is not derivable from either — whoever fails to read it
// says so (see the tray loop), and that state outranks these.
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
	SetFault(reason string)                   // one line at the top of the menu saying what is wrong ("" hides it)
	SetTally(t Tally)                         // what this box has computed, ever — the foot of the menu
	SetCap(current int)                       // check-marks the active cap in the submenu
	SetControls(c Controls)                   // wire the menu clicks; the last call wins
}

// noTray is the headless Handle.
type noTray struct{}

func (noTray) Live() bool               { return false }
func (noTray) SetTitle(string)          {}
func (noTray) SetQueue([]QueueRow, int) {}
func (noTray) SetPaused(PauseState)     {}
func (noTray) SetFault(string)          {}
func (noTray) SetTally(Tally)           {}
func (noTray) SetCap(int)               {}
func (noTray) SetControls(Controls)     {}

// Format renders the title: this machine's own load, "running/cap" — 1/1 is a box
// at cap 1 with a run on it, 3/8 is three of eight lanes busy. Idle (nothing of mine
// running) is "" so the menu bar shows just the icon.
//
// AND THAT IS THE WHOLE TITLE. It used to carry two more figures — the clock time this
// box expected to be free, then the moving-average seconds per epoch — and a menu bar is
// read sideways, between two other windows, where three numbers are none. The estimate is
// a guess about an hour and a half away and the rate is of interest once a week; both are
// still in the library, where the app draws them properly. What a person wants from the
// corner of an eye is whether the machine is busy, and how much of it.
//
// (It used to be "running/total" over the WHOLE shared queue, which on a second trainer
// read as this box's own business and was not: 2/4 while this machine held one run.)
func Format(running, cap int) string {
	if running == 0 {
		return ""
	}
	if cap < running {
		cap = running // a cap lowered under what is already going: never draw 2/1
	}
	return fmt.Sprintf("%d/%d", running, cap)
}

// FormatTally renders the line at the FOOT of the menu: "10 485 epochs · 23 probes ·
// 20 h · 6.8 s/ep" — what this machine has done, as against everything above it, which is
// what it is doing. The counts move as the epochs land, so a run adds to it while it goes.
//
// Every part after the count is dropped when the library has no figure for it, and nothing
// at all ("") hides the line rather than opening a fresh install's menu on a row of zeros.
func FormatTally(t Tally) string {
	if t.Epochs <= 0 {
		return ""
	}
	word := " epochs"
	if t.Epochs == 1 {
		word = " epoch"
	}
	line := groupThousands(t.Epochs) + word
	if t.Probes == 1 {
		line += " · 1 probe"
	} else if t.Probes > 1 {
		line += fmt.Sprintf(" · %d probes", t.Probes)
	}
	if t.Hours > 0 {
		line += fmt.Sprintf(" · %d h", t.Hours)
	}
	if t.SPerEpoch > 0 {
		line += fmt.Sprintf(" · %.1f s/ep", t.SPerEpoch)
	}
	return line
}

// groupThousands writes 12480 as "12 480": this is a number to be READ at a glance, not
// one anybody computes with.
func groupThousands(n int64) string {
	s := strconv.FormatInt(n, 10)
	if len(s) <= 3 {
		return s
	}
	var b strings.Builder
	head := len(s) % 3
	if head == 0 {
		head = 3
	}
	b.WriteString(s[:head])
	for i := head; i < len(s); i += 3 {
		b.WriteByte(' ')
		b.WriteString(s[i : i+3])
	}
	return b.String()
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
