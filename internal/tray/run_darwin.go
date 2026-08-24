// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

//go:build darwin && cgo

package tray

/*
#cgo LDFLAGS: -framework CoreGraphics -framework CoreFoundation
#include <CoreGraphics/CoreGraphics.h>
*/
import "C"

import (
	_ "embed"
	"fmt"
	"os"
	"sync"

	"fyne.io/systray"

	"orbit-capture-nam-trainer/internal/buildinfo"
	"orbit-capture-nam-trainer/internal/config"
)

//go:embed icon.png
var icon []byte

// The paused-state plates (non-template, so the colors render in the menu
// bar): orange — gate closed but a job is still draining; red — fully paused.
var (
	//go:embed icon_paused_orange.png
	iconPausedOrange []byte
	//go:embed icon_paused_red.png
	iconPausedRed []byte

	//go:embed icon_paused_yellow.png
	iconPausedYellow []byte
)

// maxQueueRows caps the dropdown list; the overflow shows as "… N more
// queued". systray menu items can only be hidden, never removed, so the slots
// are pre-created once and retitled/hidden on every update.
const maxQueueRows = 12

// statusItem is the live Handle. systray marshals every update onto the AppKit
// main thread, so all methods are goroutine-safe.
type statusItem struct {
	rows [maxQueueRows]*systray.MenuItem
	more *systray.MenuItem

	pauseNow   *systray.MenuItem
	pauseAfter *systray.MenuItem
	resume     *systray.MenuItem

	capParent *systray.MenuItem
	caps      [config.MaxCap]*systray.MenuItem
	capClicks chan int // sub-item clicks, forwarded so clickLoop stays a small select
	restart   *systray.MenuItem

	mu        sync.Mutex
	ctl       Controls
	setup     *systray.MenuItem
	fault     *systray.MenuItem
	faultText string
	stateLine *systray.MenuItem
	busy      bool
	state     PauseState
	cap       int
}

func (s *statusItem) Live() bool { return true }

// SetTitle also decides one word: a title is only drawn while this machine has a run of
// its own, so an empty one means idle.
func (s *statusItem) SetTitle(title string) {
	systray.SetTitle(title)
	s.mu.Lock()
	changed := s.busy != (title != "")
	s.busy = title != ""
	s.mu.Unlock()
	if changed {
		s.say()
	}
}

// say puts the state in WORDS — on the version line, which is always in the menu, and in
// the hover tooltip, which needs no click at all. A colour is a hint and this is the
// sentence: somebody looking at an orange icon should not have to remember what orange
// means, or open an app to find out.
func (s *statusItem) say() {
	s.mu.Lock()
	state, busy := s.state, s.busy
	s.mu.Unlock()
	word := "ready for work"
	switch {
	case state == StateNoLibrary:
		word = "no library — not working"
	case state == StatePaused:
		word = "paused"
	case state == StatePausedDraining:
		word = "pausing — a run is finishing"
	case busy:
		word = "working"
	}
	if s.stateLine != nil {
		s.stateLine.SetTitle(word)
	}
	systray.SetTooltip("OrbitCapture NAM trainer: " + word)
}

func (s *statusItem) SetQueue(rows []QueueRow, moreQueued int) {
	for i, item := range s.rows {
		if i < len(rows) {
			item.SetTitle(FormatRow(rows[i]))
			item.Show()
		} else {
			item.Hide()
		}
	}
	if moreQueued > 0 {
		s.more.SetTitle(fmt.Sprintf("… %d more queued", moreQueued))
		s.more.Show()
	} else {
		s.more.Hide()
	}
}

// SetPaused flips the menu items and swaps the icon plate, so the state is
// visible without opening the menu: orange — paused but a job still draining
// ("Pause now" stays the escalation, and it keeps what is trained); red — fully paused,
// only Resume left. Ticks with an unchanged state are dropped so the icon
// isn't re-set every 3 s.
func (s *statusItem) SetPaused(state PauseState) {
	s.mu.Lock()
	unchanged := s.state == state
	s.state = state
	s.mu.Unlock()
	if unchanged {
		return
	}
	switch state {
	case StateActive:
		systray.SetTemplateIcon(icon, icon)
		s.pauseNow.Enable()
		s.pauseAfter.Enable()
		s.resume.Disable()
	case StatePausedDraining:
		systray.SetIcon(iconPausedYellow)
		s.pauseNow.Enable() // escalate: stop the draining job now — it keeps what it has trained
		s.pauseAfter.Disable()
		s.resume.Enable()
	case StatePaused:
		systray.SetIcon(iconPausedOrange)
		s.pauseNow.Disable()
		s.pauseAfter.Disable()
		s.resume.Enable()
	case StateNoLibrary:
		// The one colour that means a fault. The pause items stay as they were: the gate
		// is a local decision and does not depend on the library answering.
		systray.SetIcon(iconPausedRed)
	}
	s.say()
}

// SetFault shows the reason line, or hides it when the reason is gone.
func (s *statusItem) SetFault(reason string) {
	s.mu.Lock()
	unchanged := s.faultText == reason
	s.faultText = reason
	s.mu.Unlock()
	if unchanged {
		return
	}
	if reason == "" {
		s.fault.Hide()
		return
	}
	s.fault.SetTitle(reason)
	s.fault.Show()
}

func (s *statusItem) SetControls(c Controls) {
	s.mu.Lock()
	s.ctl = c
	s.mu.Unlock()
}

func (s *statusItem) controls() Controls {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ctl
}

// buildMenu creates the dropdown once, at systray-ready: the pause controls on
// top, then the (hidden) queue slots and the overflow line. The list is hard-
// capped at maxQueueRows however deep the queue gets — the rest collapses into
// the overflow count.
func (s *statusItem) buildMenu() {
	// TWO WAYS TO STOP, and both are BOUNDED — this machine is one somebody also works at, so "give me
	// my GPU back" has to have an answer you can wait for. Neither loses the run: the trainer writes a
	// checkpoint after every epoch and the daemon keeps the last one.
	// A COLOUR IS NOT A DIAGNOSIS. Red says the library does not answer; this says what it
	// answered with. Hidden while there is nothing wrong, so the menu keeps its shape.
	s.fault = systray.AddMenuItem("", "")
	s.fault.Disable()
	s.fault.Hide()
	s.pauseNow = systray.AddMenuItem("Pause now",
		"Stop the running job this second, keeping everything up to the last finished epoch, and stop starting new ones. Continue it later — here or on another machine")
	s.pauseAfter = systray.AddMenuItem("Pause after current",
		"Let the running job finish all its epochs, then stop starting new ones")
	s.resume = systray.AddMenuItem("Resume", "Start working the queue again")
	s.resume.Disable()
	systray.AddSeparator()
	for i := range s.rows {
		s.rows[i] = systray.AddMenuItem("", "")
		s.rows[i].Disable()
		s.rows[i].Hide()
	}
	s.more = systray.AddMenuItem("", "")
	s.more.Disable()
	s.more.Hide()
	systray.AddSeparator()
	s.capParent = systray.AddMenuItem("Cap", "Max concurrent training jobs; applies immediately, running jobs finish (the app can ask too: workers.train_cap_wanted)")
	s.capClicks = make(chan int)
	for i := range s.caps {
		s.caps[i] = s.capParent.AddSubMenuItem(fmt.Sprintf("%d", i+1), "")
		go func(n int, ch <-chan struct{}) {
			for range ch {
				s.capClicks <- n
			}
		}(i+1, s.caps[i].ClickedCh)
	}
	s.setup = systray.AddMenuItem("Setup…",
		"Where the shared library is: host, port, database, user, password, schema — and whether this machine may sleep while the queue has work")
	s.restart = systray.AddMenuItem("Restart (re-read config)",
		"Gracefully restart the daemon; running jobs go back in the queue")
	systray.AddSeparator()
	version := systray.AddMenuItem("namtrainerd "+buildinfo.Version, "")
	version.Disable()
	// The state gets its own line UNDER the version: the version is what this is, the state
	// is what it is doing, and reading them as one sentence makes both harder to find.
	s.stateLine = systray.AddMenuItem("", "")
	s.stateLine.Disable()
	s.say()
	go s.clickLoop()
}

// SetCap check-marks the active cap and shows it on the submenu title.
// Unchanged ticks are dropped.
func (s *statusItem) SetCap(current int) {
	s.mu.Lock()
	unchanged := s.cap == current
	s.cap = current
	s.mu.Unlock()
	if unchanged {
		return
	}
	s.capParent.SetTitle(fmt.Sprintf("Cap: %d", current))
	for i, item := range s.caps {
		if i+1 == current {
			item.Check()
		} else {
			item.Uncheck()
		}
	}
}

// clickLoop forwards menu clicks to the wired Controls for the process
// lifetime. Clicks before SetControls land on nil funcs and are ignored.
func (s *statusItem) clickLoop() {
	for {
		var f func()
		select {
		case <-s.pauseNow.ClickedCh:
			f = s.controls().PauseNow
		case <-s.pauseAfter.ClickedCh:
			f = s.controls().PauseAfterCurrent
		case <-s.resume.ClickedCh:
			f = s.controls().Resume
		case <-s.setup.ClickedCh:
			f = s.controls().OpenSetup
		case <-s.restart.ClickedCh:
			f = s.controls().Restart
		case n := <-s.capClicks:
			if set := s.controls().SetCap; set != nil {
				f = func() { set(n) }
			}
		}
		if f != nil {
			f()
		}
	}
}

// Main runs the daemon body. With a window-server session it parks the main OS
// thread in the AppKit run loop (NSStatusItem needs it) and runs the body on a
// goroutine, quitting the loop when the body returns; headless — launched
// before console login, over SSH, or with ONCT_NO_TRAY set — it runs the body
// inline with a no-op Handle so a KeepAlive'd LaunchAgent can never crash-loop
// on a missing GUI.
func Main(run func(Handle)) {
	if os.Getenv("ONCT_NO_TRAY") != "" || !hasGUISession() {
		run(noTray{})
		return
	}
	s := &statusItem{}
	systray.Run(func() {
		systray.SetTemplateIcon(icon, icon)
		s.buildMenu()
		go func() {
			defer systray.Quit()
			run(s)
		}()
	}, func() {})
}

// hasGUISession reports whether a window-server (Aqua) session is reachable —
// the LaunchAgent normally has one, a pre-login or SSH launch does not.
func hasGUISession() bool {
	d := C.CGSessionCopyCurrentDictionary()
	if d == C.CFDictionaryRef(0) {
		return false
	}
	C.CFRelease(C.CFTypeRef(d))
	return true
}
