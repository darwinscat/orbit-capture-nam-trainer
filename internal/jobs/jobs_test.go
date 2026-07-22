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
	for _, k := range []string{KindTrain, KindTrainMore, KindProbeSelf, KindProbeE10} {
		if !ValidKind(k) {
			t.Errorf("ValidKind(%q) = false, want true", k)
		}
	}
	for _, k := range []string{"", "TRAIN", "probe", "nam", "trainmore"} {
		if ValidKind(k) {
			t.Errorf("ValidKind(%q) = true, want false", k)
		}
	}
}

func TestLaneAndLaneKinds(t *testing.T) {
	// train_more shares the train lane; every other kind is its own lane.
	if Lane(KindTrainMore) != KindTrain {
		t.Errorf("Lane(train_more) = %q, want %q", Lane(KindTrainMore), KindTrain)
	}
	for _, k := range []string{KindTrain, KindProbeSelf, KindProbeE10} {
		if Lane(k) != k {
			t.Errorf("Lane(%q) = %q, want itself", k, Lane(k))
		}
	}
	// LaneKinds groups train + train_more; probes stand alone.
	for _, k := range []string{KindTrain, KindTrainMore} {
		got := LaneKinds(k)
		if len(got) != 2 || got[0] != KindTrain || got[1] != KindTrainMore {
			t.Errorf("LaneKinds(%q) = %v, want [train train_more]", k, got)
		}
	}
	if got := LaneKinds(KindProbeE10); len(got) != 1 || got[0] != KindProbeE10 {
		t.Errorf("LaneKinds(probe_e10) = %v, want [probe_e10]", got)
	}
}

func TestIsTerminalAndStoresModel(t *testing.T) {
	if !IsTerminal(StateSucceeded) || !IsTerminal(StateFailed) {
		t.Error("succeeded/failed must be terminal")
	}
	if IsTerminal(StateQueued) || IsTerminal(StateRunning) {
		t.Error("queued/running must not be terminal")
	}
	if !StoresModel(KindTrain) || !StoresModel(KindTrainMore) {
		t.Error("train and train_more must store a model")
	}
	if StoresModel(KindProbeSelf) || StoresModel(KindProbeE10) {
		t.Error("probes must not store a model")
	}
}
