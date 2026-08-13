// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package jobkey

import "testing"

// lat is a *int64 literal helper for the optional latency argument.
func lat(n int64) *int64 { return &n }

func TestComputeIsDeterministicAndKnown(t *testing.T) {
	// A fixed vector pins the exact byte layout of the key formula so any drift
	// from the design notes (which the desktop app also implements) is caught.
	// A nil latency must reproduce the historical preimage byte-for-byte.
	wavHex := SHA256Hex([]byte("hello"))
	got := Compute(wavHex, "train", 100, "standard", "0.13.0", "drvsha", "sigsha", nil)

	// Recompute the expected value by hand-building the exact preimage.
	preimage := wavHex + "\n" +
		"kind=train\n" +
		"epochs=100\n" +
		"arch=standard\n" +
		"nam=0.13.0\n" +
		"driver=drvsha\n" +
		"signal=sigsha\n"
	want := SHA256Hex([]byte(preimage))

	if got != want {
		t.Errorf("Compute = %s, want %s", got, want)
	}
	if len(got) != 64 {
		t.Errorf("key length = %d, want 64 hex chars", len(got))
	}
}

func TestComputeTrainMoreAppendsBaseLine(t *testing.T) {
	// Pin the train_more formula byte-for-byte: it is Compute's preimage with one
	// final "base=<parent key>\n" line appended, and nothing else.
	wavHex := SHA256Hex([]byte("hello"))
	parent := SHA256Hex([]byte("parent"))
	got := ComputeTrainMore(wavHex, 400, "standard", "0.13.0", "drvsha", "sigsha", parent, nil)

	preimage := wavHex + "\n" +
		"kind=train_more\n" +
		"epochs=400\n" +
		"arch=standard\n" +
		"nam=0.13.0\n" +
		"driver=drvsha\n" +
		"signal=sigsha\n" +
		"base=" + parent + "\n"
	want := SHA256Hex([]byte(preimage))

	if got != want {
		t.Errorf("ComputeTrainMore = %s, want %s", got, want)
	}
	if len(got) != 64 {
		t.Errorf("key length = %d, want 64 hex chars", len(got))
	}
	// The base line is load-bearing: the same inputs as a plain train_more without
	// it (i.e. Compute with kind=train_more) must NOT collide.
	if plain := Compute(wavHex, "train_more", 400, "standard", "0.13.0", "drvsha", "sigsha", nil); plain == got {
		t.Error("base line did not change the key — train_more must include its parent")
	}
	// A different parent yields a different key.
	if other := ComputeTrainMore(wavHex, 400, "standard", "0.13.0", "drvsha", "sigsha", SHA256Hex([]byte("other")), nil); other == got {
		t.Error("changing base did not change the key")
	}
}

func TestComputeAppendsLatencyLine(t *testing.T) {
	// Pin the latency line byte-for-byte: it is the LAST line, present only when a
	// latency was supplied — the same shape as base.
	wavHex := SHA256Hex([]byte("hello"))
	got := Compute(wavHex, "train", 100, "standard", "0.13.0", "drvsha", "sigsha", lat(1101))

	preimage := wavHex + "\n" +
		"kind=train\n" +
		"epochs=100\n" +
		"arch=standard\n" +
		"nam=0.13.0\n" +
		"driver=drvsha\n" +
		"signal=sigsha\n" +
		"latency=1101\n"
	if want := SHA256Hex([]byte(preimage)); got != want {
		t.Errorf("Compute with latency = %s, want %s", got, want)
	}

	// Absence, zero, and any other value are three DIFFERENT identities: the same
	// wav trained on a different trim is a different model.
	absent := Compute(wavHex, "train", 100, "standard", "0.13.0", "drvsha", "sigsha", nil)
	zero := Compute(wavHex, "train", 100, "standard", "0.13.0", "drvsha", "sigsha", lat(0))
	other := Compute(wavHex, "train", 100, "standard", "0.13.0", "drvsha", "sigsha", lat(1102))
	for name, k := range map[string]string{"absent": absent, "zero": zero, "other": other} {
		if k == got {
			t.Errorf("latency=1101 collides with %s", name)
		}
	}
	if zero == absent {
		t.Error("latency=0 collides with an absent latency — absence is its own value")
	}
}

func TestComputeTrainMoreOrdersBaseThenLatency(t *testing.T) {
	// With both present the order is base, then latency — the app hashes the same
	// preimage, so a swap here silently breaks every continuation.
	wavHex := SHA256Hex([]byte("hello"))
	parent := SHA256Hex([]byte("parent"))
	got := ComputeTrainMore(wavHex, 400, "standard", "0.13.0", "drvsha", "sigsha", parent, lat(1101))

	preimage := wavHex + "\n" +
		"kind=train_more\n" +
		"epochs=400\n" +
		"arch=standard\n" +
		"nam=0.13.0\n" +
		"driver=drvsha\n" +
		"signal=sigsha\n" +
		"base=" + parent + "\n" +
		"latency=1101\n"
	if want := SHA256Hex([]byte(preimage)); got != want {
		t.Errorf("ComputeTrainMore with latency = %s, want %s", got, want)
	}
	if plain := ComputeTrainMore(wavHex, 400, "standard", "0.13.0", "drvsha", "sigsha", parent, nil); plain == got {
		t.Error("latency line did not change the train_more key")
	}
}

func TestComputeVariesWithEveryField(t *testing.T) {
	base := Compute("wav", "train", 100, "standard", "0.13.0", "drv", "sig", nil)
	variants := map[string]string{
		"wav":     Compute("wav2", "train", 100, "standard", "0.13.0", "drv", "sig", nil),
		"kind":    Compute("wav", "probe_self", 100, "standard", "0.13.0", "drv", "sig", nil),
		"epochs":  Compute("wav", "train", 200, "standard", "0.13.0", "drv", "sig", nil),
		"arch":    Compute("wav", "train", 100, "lite", "0.13.0", "drv", "sig", nil),
		"nam":     Compute("wav", "train", 100, "standard", "0.14.0", "drv", "sig", nil),
		"driver":  Compute("wav", "train", 100, "standard", "0.13.0", "drv2", "sig", nil),
		"signal":  Compute("wav", "train", 100, "standard", "0.13.0", "drv", "sig2", nil),
		"latency": Compute("wav", "train", 100, "standard", "0.13.0", "drv", "sig", lat(1101)),
	}
	for field, v := range variants {
		if v == base {
			t.Errorf("changing %s did not change the key", field)
		}
	}
}
