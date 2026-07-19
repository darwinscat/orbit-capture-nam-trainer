// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package sysutil holds small OS helpers shared across the daemon.
package sysutil

import "syscall"

// FreeBytes returns the bytes available to an unprivileged process on the
// filesystem that holds path. Works on darwin and linux (both expose
// syscall.Statfs with Bavail/Bsize); the daemon targets neither Windows nor
// Plan 9, so no build tags are needed.
func FreeBytes(path string) (uint64, error) {
	var fs syscall.Statfs_t
	if err := syscall.Statfs(path, &fs); err != nil {
		return 0, err
	}
	// Bavail = blocks free to a non-root caller; Bsize = block size. Cast both:
	// the field widths differ between darwin and linux, uint64 covers both.
	return uint64(fs.Bavail) * uint64(fs.Bsize), nil
}
