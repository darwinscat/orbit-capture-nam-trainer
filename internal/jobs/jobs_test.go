// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package jobs

import "testing"

func TestLaneMatchesSchemaRule(t *testing.T) {
	// The schema generates lane = CASE WHEN kind='probe_self' THEN 'probe' ELSE 'train' END.
	for kind, want := range map[string]string{KindTrain: LaneTrain, KindTrainMore: LaneTrain, KindProbeSelf: LaneProbe} {
		if got := Lane(kind); got != want {
			t.Errorf("Lane(%s) = %q, want %q", kind, got, want)
		}
	}
}

func TestLaneKinds(t *testing.T) {
	if k := LaneKinds(LaneTrain); len(k) != 2 || k[0] != KindTrain || k[1] != KindTrainMore {
		t.Errorf("LaneKinds(train) = %v", k)
	}
	if k := LaneKinds(LaneProbe); len(k) != 1 || k[0] != KindProbeSelf {
		t.Errorf("LaneKinds(probe) = %v", k)
	}
}

func TestValidKindAndTerminal(t *testing.T) {
	for _, k := range []string{KindTrain, KindTrainMore, KindProbeSelf} {
		if !ValidKind(k) {
			t.Errorf("ValidKind(%s) = false", k)
		}
	}
	if ValidKind("probe_e10") {
		t.Error("probe_e10 is gone from the schema and must not validate")
	}
	for _, s := range []string{StateSucceeded, StateFailed, StateCancelled} {
		if !IsTerminal(s) {
			t.Errorf("IsTerminal(%s) = false", s)
		}
	}
	if IsTerminal(StateQueued) || IsTerminal(StateRunning) {
		t.Error("queued/running are not terminal")
	}
}
