// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

//go:build !darwin

package awake

// platformSpawn is a no-op on platforms without a supported power assertion:
// the Keeper "holds" nothing and the machine keeps its normal sleep behaviour.
func platformSpawn() (func(), error) { return func() {}, nil }
