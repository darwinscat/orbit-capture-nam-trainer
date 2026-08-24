// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

//go:build !darwin || !cgo

package tray

// ShowSetup has nowhere to draw on a headless build: there is no menu bar to open
// it from either, and the config file is edited by hand — which is what a Linux
// service does anyway.
func ShowSetup(Setup, func(Setup)) {}
