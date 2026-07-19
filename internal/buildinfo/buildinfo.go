// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package buildinfo carries the daemon's version, reported by GET /v1/health.
package buildinfo

// Version is the daemon version. Overridable at build time with
//
//	go build -ldflags "-X orbit-capture-nam-trainer/internal/buildinfo.Version=1.2.3"
var Version = "0.1.0-dev"

// Protocol is the API contract version (see the design notes). Bump only on a
// breaking change to the wire contract shared with the desktop app.
const Protocol = 1
