// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package jobkey

import "testing"

func TestComputeIsDeterministicAndKnown(t *testing.T) {
	// A fixed vector pins the exact byte layout of the key formula so any drift
	// from the design notes (which the desktop app also implements) is caught.
	wavHex := SHA256Hex([]byte("hello"))
	got := Compute(wavHex, "train", 100, "standard", "0.13.0", "drvsha", "sigsha")

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

func TestComputeVariesWithEveryField(t *testing.T) {
	base := Compute("wav", "train", 100, "standard", "0.13.0", "drv", "sig")
	variants := map[string]string{
		"wav":    Compute("wav2", "train", 100, "standard", "0.13.0", "drv", "sig"),
		"kind":   Compute("wav", "probe_self", 100, "standard", "0.13.0", "drv", "sig"),
		"epochs": Compute("wav", "train", 200, "standard", "0.13.0", "drv", "sig"),
		"arch":   Compute("wav", "train", 100, "lite", "0.13.0", "drv", "sig"),
		"nam":    Compute("wav", "train", 100, "standard", "0.14.0", "drv", "sig"),
		"driver": Compute("wav", "train", 100, "standard", "0.13.0", "drv2", "sig"),
		"signal": Compute("wav", "train", 100, "standard", "0.13.0", "drv", "sig2"),
	}
	for field, v := range variants {
		if v == base {
			t.Errorf("changing %s did not change the key", field)
		}
	}
}
