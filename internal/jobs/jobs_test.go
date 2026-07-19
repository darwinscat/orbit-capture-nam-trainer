// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package jobs

import "testing"

func TestNormalizeEpochs(t *testing.T) {
	cases := []struct {
		kind      string
		requested int
		want      int
	}{
		{KindTrain, 100, 100},
		{KindTrain, 800, 800},
		{KindProbeSelf, 999, ProbeSelfEpochs}, // requested ignored
		{KindProbeE10, 999, ProbeE10Epochs},   // requested ignored
	}
	for _, tc := range cases {
		if got := NormalizeEpochs(tc.kind, tc.requested); got != tc.want {
			t.Errorf("NormalizeEpochs(%q, %d) = %d, want %d", tc.kind, tc.requested, got, tc.want)
		}
	}
}

func TestValidKind(t *testing.T) {
	for _, k := range []string{KindTrain, KindProbeSelf, KindProbeE10} {
		if !ValidKind(k) {
			t.Errorf("ValidKind(%q) = false, want true", k)
		}
	}
	for _, k := range []string{"", "TRAIN", "probe", "nam"} {
		if ValidKind(k) {
			t.Errorf("ValidKind(%q) = true, want false", k)
		}
	}
}

func TestIsTerminalAndStoresModel(t *testing.T) {
	if !IsTerminal(StateSucceeded) || !IsTerminal(StateFailed) {
		t.Error("succeeded/failed must be terminal")
	}
	if IsTerminal(StateQueued) || IsTerminal(StateRunning) {
		t.Error("queued/running must not be terminal")
	}
	if !StoresModel(KindTrain) {
		t.Error("train must store a model")
	}
	if StoresModel(KindProbeSelf) || StoresModel(KindProbeE10) {
		t.Error("probes must not store a model")
	}
}
