// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package tray

import (
	"testing"
	"time"
)

func f64(v float64) *float64 { return &v }

func TestFormat(t *testing.T) {
	now := time.Date(2026, 7, 19, 8, 22, 0, 0, time.UTC)
	for _, tc := range []struct {
		name         string
		running, cap int
		eta, spe     *float64
		want         string
	}{
		{"idle is icon-only", 0, 8, nil, f64(5.14), ""},
		{"one of one", 1, 1, nil, nil, "1/1"},
		{"three of eight lanes busy", 3, 8, nil, nil, "3/8"},
		{"full title", 2, 8, f64(5*3600 + 14*60), f64(5.14), "2/8 13:36 5.14"},
		{"no rate yet", 1, 1, f64(60), nil, "1/1 08:23"},
		{"clock wraps past midnight stays clock", 1, 2, f64(16 * 3600), f64(4.2), "1/2 00:22 4.20"},
		{"day-plus eta", 1, 4, f64(26 * 3600), f64(9.876), "1/4 24h+ 9.88"},
		// A cap lowered while more runs are already going never draws 2/1.
		{"cap under what is running", 2, 1, nil, nil, "2/2"},
	} {
		if got := Format(now, tc.running, tc.cap, tc.eta, tc.spe); got != tc.want {
			t.Errorf("%s: Format = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestMineSeconds(t *testing.T) {
	// Two of my own runs go on at the same time: the longer one decides when this
	// box is free — 600 epochs × 5 s.
	if got := MineSeconds([]int64{170, 600}, 5); got != 3000 {
		t.Errorf("two of mine = %v, want 3000 (the longer one)", got)
	}
	if got := MineSeconds([]int64{170}, 5); got != 850 {
		t.Errorf("one of mine = %v, want 850", got)
	}
	// Nothing of mine is running: there is nothing to promise, whatever the queue
	// holds — somebody else's rows are somebody else's speed.
	if got := MineSeconds(nil, 5); got != 0 {
		t.Errorf("none of mine = %v, want 0", got)
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
