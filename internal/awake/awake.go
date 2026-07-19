// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package awake keeps the machine from idle-sleeping while the daemon has
// pending training work, and lets it sleep normally once the queue is empty.
//
// On a laptop a training run that outlives the idle timer is otherwise frozen
// the moment macOS sleeps — the queue then crawls, computing only during brief
// wakes. The Keeper holds a single OS power assertion for as long as work is
// pending and releases it when the queue drains. On platforms without a
// supported assertion the hold is a no-op, so callers need no build tags.
//
// The assertion prevents *idle* sleep; it does not override sleep from closing
// the lid. Keep the lid open (or run clamshell on external power + display).
package awake

import "sync"

// spawnFunc starts an OS "stay awake" assertion and returns a func that releases
// it. It is the platform hook (see hold_*.go) and is overridable in tests.
type spawnFunc func() (release func(), err error)

// Keeper holds at most one power assertion, matching it to whether work is
// pending. Safe for concurrent use.
type Keeper struct {
	enabled bool
	spawn   spawnFunc
	log     func(string, ...any)

	mu      sync.Mutex
	release func() // non-nil exactly while the assertion is held
}

// New returns a Keeper. When enabled is false every method is a no-op and the
// machine keeps its normal sleep behaviour. log may be nil.
func New(enabled bool, log func(string, ...any)) *Keeper {
	return &Keeper{enabled: enabled, spawn: platformSpawn, log: log}
}

// Set makes the assertion match pending: held while true, released while false.
// It is idempotent — repeated calls in the same effective state do nothing, so
// it is cheap to call on every queue-count change.
func (k *Keeper) Set(pending bool) {
	if !k.enabled {
		return
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	switch {
	case pending && k.release == nil:
		rel, err := k.spawn()
		if err != nil {
			k.logf("keep-awake: could not hold assertion: %v", err)
			return
		}
		k.release = rel
		k.logf("keep-awake: holding while work is pending")
	case !pending && k.release != nil:
		k.release()
		k.release = nil
		k.logf("keep-awake: released (queue idle)")
	}
}

// Close releases any held assertion. Safe to call more than once.
func (k *Keeper) Close() {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.release != nil {
		k.release()
		k.release = nil
	}
}

func (k *Keeper) logf(format string, args ...any) {
	if k.log != nil {
		k.log(format, args...)
	}
}
