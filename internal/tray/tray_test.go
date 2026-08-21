// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package tray

import (
	"testing"
	"time"

	"orbit-capture-nam-trainer/internal/jobs"
)

func f64(v float64) *float64 { return &v }

func TestFormat(t *testing.T) {
	now := time.Date(2026, 7, 19, 8, 22, 0, 0, time.UTC)
	for _, tc := range []struct {
		name            string
		running, queued int
		eta, spe        *float64
		want            string
	}{
		{"idle is icon-only", 0, 0, nil, f64(5.14), ""},
		{"running of total", 2, 2, nil, nil, "2/4"},
		{"full title", 2, 18, f64(5*3600 + 14*60), f64(5.14), "2/20 13:36 5.14"},
		{"no rate yet", 1, 0, f64(60), nil, "1/1 08:23"},
		{"clock wraps past midnight stays clock", 1, 2, f64(16 * 3600), f64(4.2), "1/3 00:22 4.20"},
		{"day-plus eta", 1, 40, f64(26 * 3600), f64(9.876), "1/41 24h+ 9.88"},
	} {
		if got := Format(now, tc.running, tc.queued, tc.eta, tc.spe); got != tc.want {
			t.Errorf("%s: Format = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestQueueSeconds(t *testing.T) {
	remaining := map[string]int64{
		jobs.LaneTrain: 600,
		jobs.LaneProbe: 10,
	}
	// Train lane dominates: 600 epochs × 5 s ÷ cap 2 = 1500 s.
	if got := QueueSeconds(remaining, 5, 2, 1); got != 1500 {
		t.Errorf("train-dominated = %v, want 1500", got)
	}
	// With the train lane wide, the probe lane can dominate: 10×5 = 50 vs 600×5÷100 = 30.
	if got := QueueSeconds(remaining, 5, 100, 1); got != 50 {
		t.Errorf("probe-dominated = %v, want 50", got)
	}
	// A zero/absurd cap is clamped to 1, never a divide-by-zero.
	if got := QueueSeconds(map[string]int64{jobs.LaneTrain: 10}, 2, 0, 1); got != 20 {
		t.Errorf("clamped cap = %v, want 20", got)
	}
	// PIN: the store keys the remaining map by jobs.lane (the schema's generated
	// column), and QueueSeconds indexes it by jobs.LaneTrain — train_more's lane
	// MUST resolve to that string or the whole train lane vanishes from the ETA.
	if got := QueueSeconds(map[string]int64{jobs.Lane(jobs.KindTrainMore): 10}, 2, 1, 1); got != 20 {
		t.Errorf("train_more lane key = %v, want 20 (Lane(train_more) must equal LaneTrain)", got)
	}
	if got := QueueSeconds(nil, 5, 1, 1); got != 0 {
		t.Errorf("empty = %v, want 0", got)
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
