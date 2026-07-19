// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"strings"
	"testing"
	"time"
)

func TestParseEpoch(t *testing.T) {
	cases := map[string]int{
		"Epoch 0/100":                       0,
		"Epoch 42/100 · 7.3 s/ep":           42,
		"  Epoch 7/10":                      7,
		"DRIVER: training model epochs=100": -1, // must NOT match the banner
		"no epoch here":                     -1,
		"Epoch abc":                         -1,
	}
	for line, want := range cases {
		if got := parseEpoch(line); got != want {
			t.Errorf("parseEpoch(%q) = %d, want %d", line, got, want)
		}
	}
}

func TestParseReplicateESR(t *testing.T) {
	// The real NAM line has a trailing period; scientific notation also occurs.
	v, ok := parseReplicateESR("Replicate ESR is 0.00003540.")
	if !ok || v < 3.5e-5 || v > 3.6e-5 { // value is 3.54e-05
		t.Errorf("trailing-period parse = %v, ok=%v", v, ok)
	}
	v, ok = parseReplicateESR("Replicate ESR is 9.3e-05")
	if !ok || v < 9.2e-5 || v > 9.4e-5 {
		t.Errorf("sci-notation parse = %v, ok=%v", v, ok)
	}
	if _, ok := parseReplicateESR("some other line"); ok {
		t.Error("non-matching line should not parse")
	}
}

func TestParseDriverESR(t *testing.T) {
	v, isNA, ok := parseDriverESR("DRIVER: esr=0.01690431")
	if !ok || isNA || v < 0.016 || v > 0.017 {
		t.Errorf("esr parse = %v na=%v ok=%v", v, isNA, ok)
	}
	_, isNA, ok = parseDriverESR("DRIVER: esr=na")
	if !ok || !isNA {
		t.Errorf("na parse: na=%v ok=%v", isNA, ok)
	}
	if _, _, ok := parseDriverESR("Epoch 3/10"); ok {
		t.Error("non-esr line should not parse")
	}
}

func TestIsProbeFail(t *testing.T) {
	for _, l := range []string{"DRIVER: checkfail", "Failed checks: latency", "it doesn't sound like itself"} {
		if !isProbeFail(l) {
			t.Errorf("isProbeFail(%q) = false, want true", l)
		}
	}
	for _, l := range []string{"Epoch 0/1", "Replicate ESR is 0.01.", "training"} {
		if isProbeFail(l) {
			t.Errorf("isProbeFail(%q) = true, want false", l)
		}
	}
}

func TestEpochTrackerEWMA(t *testing.T) {
	var e epochTracker
	base := time.Unix(1000, 0)
	if !e.observe(0, base) {
		t.Fatal("first epoch should register as changed")
	}
	if e.sPerEpoch != 0 {
		t.Errorf("no s/epoch until a second epoch, got %v", e.sPerEpoch)
	}
	// 2 s later at epoch 1 → delta 2s > 50ms guard.
	if !e.observe(1, base.Add(2*time.Second)) {
		t.Fatal("epoch advance should register")
	}
	if e.sPerEpoch < 1.9 || e.sPerEpoch > 2.1 {
		t.Errorf("s/epoch = %v, want ~2", e.sPerEpoch)
	}
	// Same epoch again → no change.
	if e.observe(1, base.Add(3*time.Second)) {
		t.Error("same epoch should not register as changed")
	}
	// A sub-50ms burst delta is ignored (EWMA unchanged).
	prev := e.sPerEpoch
	e.observe(2, base.Add(2*time.Second+10*time.Millisecond))
	if e.sPerEpoch != prev {
		t.Errorf("sub-50ms delta should be ignored, s/epoch changed %v→%v", prev, e.sPerEpoch)
	}
}

func TestLineReaderSplitsCRandLF(t *testing.T) {
	lr := newLineReader(strings.NewReader("a\r\nb\nc\rd\r\n"))
	var got []string
	for {
		line, err := lr.next()
		if err != nil {
			break
		}
		got = append(got, line)
	}
	want := []string{"a", "b", "c", "d"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("lines = %v, want %v", got, want)
	}
}
