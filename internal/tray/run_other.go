// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

//go:build !darwin || !cgo

package tray

// Main without a menu bar (Linux, or a CGO_ENABLED=0 build) is a pass-through:
// the daemon body runs inline with a no-op Handle.
func Main(run func(Handle)) { run(noTray{}) }
