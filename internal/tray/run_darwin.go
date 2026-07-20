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
	apiCap    *systray.MenuItem
	restart   *systray.MenuItem

	mu         sync.Mutex
	ctl        Controls
	state      PauseState
	cap        int
	apiAllowed bool
	apiInit    bool // first SetAPICapAllowed must render even for false
}

func (s *statusItem) Live() bool { return true }

func (s *statusItem) SetTitle(title string) { systray.SetTitle(title) }

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
// ("Pause now" stays available as the kill escalation); red — fully paused,
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
		systray.SetIcon(iconPausedOrange)
		s.pauseNow.Enable() // escalate: kill the draining job (it requeues)
		s.pauseAfter.Disable()
		s.resume.Enable()
	case StatePaused:
		systray.SetIcon(iconPausedRed)
		s.pauseNow.Disable()
		s.pauseAfter.Disable()
		s.resume.Enable()
	}
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
	s.pauseNow = systray.AddMenuItem("Pause now",
		"Stop the running job (it goes back in the queue) and stop starting new ones")
	s.pauseAfter = systray.AddMenuItem("Pause after current",
		"Let the running job finish, then stop starting new ones")
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
	s.capParent = systray.AddMenuItem("Cap", "Max concurrent training jobs; applies immediately, running jobs finish")
	s.capClicks = make(chan int)
	for i := range s.caps {
		s.caps[i] = s.capParent.AddSubMenuItem(fmt.Sprintf("%d", i+1), "")
		go func(n int, ch <-chan struct{}) {
			for range ch {
				s.capClicks <- n
			}
		}(i+1, s.caps[i].ClickedCh)
	}
	s.capParent.AddSeparator()
	s.apiCap = s.capParent.AddSubMenuItemCheckbox("Allow cap via API",
		"Let clients change cap with PATCH /v1/cap; off = admin-only (403)", false)
	s.restart = systray.AddMenuItem("Restart (re-read config)",
		"Gracefully restart the daemon; running jobs go back in the queue")
	systray.AddSeparator()
	version := systray.AddMenuItem("namtrainerd "+buildinfo.Version, "")
	version.Disable()
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

// SetAPICapAllowed check-marks the API-permission toggle. Unchanged ticks are
// dropped (apiInit forces the very first render, which may be false).
func (s *statusItem) SetAPICapAllowed(allowed bool) {
	s.mu.Lock()
	unchanged := s.apiInit && s.apiAllowed == allowed
	s.apiAllowed = allowed
	s.apiInit = true
	s.mu.Unlock()
	if unchanged {
		return
	}
	if allowed {
		s.apiCap.Check()
	} else {
		s.apiCap.Uncheck()
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
		case <-s.restart.ClickedCh:
			f = s.controls().Restart
		case <-s.apiCap.ClickedCh:
			f = s.controls().ToggleAPICap
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
