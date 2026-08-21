// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package buildinfo carries the daemon's version, reported in workers.version by
// the heartbeat. The contract version it speaks is store.SupportedQueueContract.
package buildinfo

// Version is the daemon version. Overridable at build time with
//
//	go build -ldflags "-X orbit-capture-nam-trainer/internal/buildinfo.Version=1.2.3"
var Version = "0.1.0-dev"
