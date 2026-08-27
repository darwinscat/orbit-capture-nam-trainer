// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package tray

import "testing"

func TestFormat(t *testing.T) {
	for _, tc := range []struct {
		name         string
		running, cap int
		want         string
	}{
		{"idle is icon-only", 0, 8, ""},
		{"one of one", 1, 1, "1/1"},
		{"three of eight lanes busy", 3, 8, "3/8"},
		// A cap lowered while more runs are already going never draws 2/1.
		{"cap under what is running", 2, 1, "2/2"},
	} {
		if got := Format(tc.running, tc.cap); got != tc.want {
			t.Errorf("%s: Format = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestFormatTally(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   Tally
		want string
	}{
		// A machine that has computed nothing says nothing — the line hides.
		{"nothing yet", Tally{}, ""},
		{"one is singular", Tally{Epochs: 1}, "1 epoch"},
		{"grouped in threes", Tally{Epochs: 12480}, "12 480 epochs"},
		{"the whole line", Tally{Epochs: 10485, Probes: 23, Hours: 20, SPerEpoch: 6.8},
			"10 485 epochs · 23 probes · 20 h · 6.8 s/ep"},
		// Each part is dropped until the library has a figure for it: a box in its first hour
		// has no hours to show, and one that has never self-checked has no probes.
		{"a first hour", Tally{Epochs: 42, SPerEpoch: 5.3}, "42 epochs · 5.3 s/ep"},
		{"probes but no hours yet", Tally{Epochs: 3, Probes: 3}, "3 epochs · 3 probes"},
		// One self-check is one probe, not "1 probes".
		{"a single probe", Tally{Epochs: 6, Probes: 1}, "6 epochs · 1 probe"},
	} {
		if got := FormatTally(tc.in); got != tc.want {
			t.Errorf("%s: FormatTally(%+v) = %q, want %q", tc.name, tc.in, got, tc.want)
		}
	}
}

func TestFormatRow(t *testing.T) {
	ep := int64(41)
	for _, tc := range []struct {
		name string
		row  QueueRow
		want string
	}{
		{"running with epoch", QueueRow{Running: true, Kind: "train", Epochs: 300, Epoch: &ep,
			Label: "RAT 2-0008"}, "▶ train 42/300 RAT 2-0008"},
		{"running before first epoch", QueueRow{Running: true, Kind: "probe_self", Epochs: 1,
			Label: "Big Muff-0003"}, "▶ probe_self –/1 Big Muff-0003"},
		{"queued", QueueRow{Kind: "train_more", Epochs: 300, Label: "TS-0012"}, "train_more 300 ep TS-0012"},
	} {
		if got := FormatRow(tc.row); got != tc.want {
			t.Errorf("%s: FormatRow = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestDeriveState(t *testing.T) {
	for _, tc := range []struct {
		paused  bool
		running int
		want    PauseState
	}{
		{false, 0, StateActive},
		{false, 3, StateActive},
		{true, 1, StatePausedDraining},
		{true, 0, StatePaused},
	} {
		if got := DeriveState(tc.paused, tc.running); got != tc.want {
			t.Errorf("DeriveState(%v, %d) = %v, want %v", tc.paused, tc.running, got, tc.want)
		}
	}
}
