// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package awake

import (
	"errors"
	"testing"
)

// fakeSpawn counts starts and tracks how many assertions are currently held.
func fakeSpawn(starts, held *int) spawnFunc {
	return func() (func(), error) {
		*starts++
		*held++
		return func() { *held-- }, nil
	}
}

func TestKeeperHoldsOnceAndReleases(t *testing.T) {
	var starts, held int
	k := &Keeper{enabled: true, spawn: fakeSpawn(&starts, &held)}

	k.Set(true)
	k.Set(true) // idempotent: still one hold, no second caffeinate
	if starts != 1 || held != 1 {
		t.Fatalf("after Set(true)×2: starts=%d held=%d, want 1/1", starts, held)
	}

	k.Set(false)
	k.Set(false) // idempotent
	if starts != 1 || held != 0 {
		t.Fatalf("after Set(false)×2: starts=%d held=%d, want 1/0", starts, held)
	}

	k.Set(true) // re-acquire after a release
	if starts != 2 || held != 1 {
		t.Fatalf("after re-Set(true): starts=%d held=%d, want 2/1", starts, held)
	}

	k.Close()
	k.Close() // safe to call twice
	if held != 0 {
		t.Fatalf("after Close: held=%d, want 0", held)
	}
}

func TestKeeperDisabledNeverSpawns(t *testing.T) {
	var starts, held int
	k := &Keeper{enabled: false, spawn: fakeSpawn(&starts, &held)}
	k.Set(true)
	k.Close()
	if starts != 0 {
		t.Fatalf("disabled keeper spawned %d times, want 0", starts)
	}
}

func TestKeeperSpawnErrorStaysReleased(t *testing.T) {
	var starts int
	fail := true
	k := &Keeper{enabled: true, spawn: func() (func(), error) {
		starts++
		if fail {
			return nil, errors.New("boom")
		}
		return func() {}, nil
	}}

	k.Set(true) // spawn fails → not held
	if k.release != nil {
		t.Fatal("failed spawn must leave the keeper released")
	}
	fail = false
	k.Set(true) // retried on the next pending transition
	if starts != 2 || k.release == nil {
		t.Fatalf("keeper did not retry after a spawn error: starts=%d held=%v", starts, k.release != nil)
	}
}

// The real platform hook must be usable: no-op on non-darwin, a live caffeinate
// on darwin. Either way spawn+release must not error or panic.
func TestPlatformSpawnRoundTrips(t *testing.T) {
	release, err := platformSpawn()
	if err != nil {
		t.Fatalf("platformSpawn: %v", err)
	}
	release()
	release() // release must be safe to call twice
}
