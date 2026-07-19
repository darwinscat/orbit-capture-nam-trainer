// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package runtime owns the managed python trainer: it provisions a pinned
// python-build-standalone + venv + neural-amp-modeler into the daemon's app-data,
// vendors the trainer driver, and fetches the capture signal. It resolves the
// trainer profile (python/nam versions + driver/signal sha) that feeds
// /v1/health and the content-addressed job key.
package runtime

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	stdruntime "runtime"
)

// Pins for the managed python runtime and the NAM trainer version. The python
// runtime archive is resolved per-platform (see pythonPin); these are the shared
// bits.
const (
	// python-build-standalone release tag and CPython version (same across platforms).
	pyRelease = "20260610"
	pyVersion = "3.12.13"

	NamPin = "0.13.0"

	// The standardized NAM capture signal. It carries NO redistribution license
	// upstream, so it is never shipped or mirrored: it is downloaded from the
	// official source on first run and verified against SignalSHA256 (the transport
	// is untrusted-but-verified).
	SignalName   = "v3_0_0.wav"
	SignalSHA256 = "70f8ec7f25686a1bd77f25973de8e51a6721e957e81eec121822e5e53366bc41"
	SignalURL    = "https://drive.usercontent.google.com/download?id=1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G" +
		"&export=download&confirm=t"
	// SignalPageURL is the human download page, for the manual fallback when the
	// direct fetch is quota-limited (download in a browser, drop the file in place).
	SignalPageURL = "https://drive.google.com/file/d/1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G/view"
)

// pythonTargets maps GOOS/GOARCH → the python-build-standalone target triple and
// its install_only tarball sha256 (from the pyRelease SHA256SUMS). Add a platform
// by dropping in its triple + sha.
var pythonTargets = map[string]struct{ triple, sha256 string }{
	"darwin/arm64": {"aarch64-apple-darwin", "e18ddd4c1e8f4a1d6c4590b37f423d76aec734447edc20ed08e93983d95f2132"},
	"linux/amd64":  {"x86_64-unknown-linux-gnu", "c218f50baeb2c06a30c2f03db5986b2bad6ab7c8a52faad2d5a59bda0677b93a"},
}

// pythonPin resolves the python runtime archive name, download URL, and sha256 for
// the current platform. It errors on an unsupported GOOS/GOARCH rather than
// unpacking a wrong-arch tarball.
func pythonPin() (archive, url, sha string, err error) {
	t, ok := pythonTargets[stdruntime.GOOS+"/"+stdruntime.GOARCH]
	if !ok {
		return "", "", "", fmt.Errorf("no python runtime pinned for %s/%s", stdruntime.GOOS, stdruntime.GOARCH)
	}
	archive = fmt.Sprintf("cpython-%s+%s-%s-install_only.tar.gz", pyVersion, pyRelease, t.triple)
	url = "https://github.com/astral-sh/python-build-standalone/releases/download/" + pyRelease + "/" + archive
	return archive, url, t.sha256, nil
}

// The trainer driver is vendored (AGPL, like this project); this service owns the
// training semantics. Its sha256 feeds the job key.
//
//go:embed assets/trainer_driver.py
var trainerDriverPy []byte

// DriverBytes returns the embedded trainer driver source.
func DriverBytes() []byte { return trainerDriverPy }

// DriverSHA256 is the sha256 hex of the vendored driver (a key input).
func DriverSHA256() string { return sha256hex(trainerDriverPy) }

func sha256hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
