// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package jobkey computes the content-addressed job key from the design notes.
// The formula is shared VERBATIM with the desktop app — do not drift from it. The
// key is identity (same key ⇒ same work); priority is deliberately excluded
// (it's scheduling, not identity).
package jobkey

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
)

// SHA256Hex is the lower-case hex sha256 of b.
func SHA256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// Compute derives the job key.
//
//	key = sha256hex(
//	  sha256hex(wav_bytes) + "\n" +
//	  "kind="   + kind        + "\n" +
//	  "epochs=" + epochs      + "\n" +   // probe_self:1, probe_e10:10, train:requested
//	  "arch="   + arch        + "\n" +
//	  "nam="    + namVersion  + "\n" +   // the RESOLVED nam version, never the pin
//	  "driver=" + driverSHA   + "\n" +
//	  "signal=" + signalSHA   + "\n" +
//	[ "latency="+ latency     + "\n" ] ) // ONLY when the caller sent one
//
// wavHex is the sha256 hex of the raw wav bytes; epochs must already be
// normalized for the kind (see jobs.NormalizeEpochs). latency is the client's
// round-trip in samples, nil when absent (the trainer then auto-detects, as
// before) — a nil latency reproduces the historical key byte-for-byte.
func Compute(wavHex, kind string, epochs int, arch, namVersion, driverSHA, signalSHA string, latency *int64) string {
	return SHA256Hex([]byte(preimage(wavHex, kind, epochs, arch, namVersion, driverSHA, signalSHA, "", latency)))
}

// ComputeTrainMore derives the key of a kind=train_more job: the same preimage as
// Compute, with one line "base=" + baseKey + "\n" appended. The parent is
// thereby part of the child's identity. Callers use this ONLY for train_more —
// the other kinds' formula stays byte-for-byte unchanged (see Compute). An
// optional latency line follows base (see preimage).
func ComputeTrainMore(wavHex string, epochs int, arch, namVersion, driverSHA, signalSHA, baseKey string, latency *int64) string {
	return SHA256Hex([]byte(preimage(wavHex, "train_more", epochs, arch, namVersion, driverSHA, signalSHA, baseKey, latency)))
}

// preimage builds the canonical key preimage. Both trailing lines are optional
// and, when present, always appear in this order: base, when non-empty, appends
// "base=<parent key>\n" — present ONLY for kind=train_more; latency, when
// non-nil, appends "latency=<samples>\n" — present on any kind whose caller
// supplied one. Absence is not the same as zero: latency=0 (train with no trim)
// carries its own line and its own key.
func preimage(wavHex, kind string, epochs int, arch, namVersion, driverSHA, signalSHA, base string, latency *int64) string {
	var sb strings.Builder
	sb.WriteString(wavHex)
	sb.WriteString("\nkind=")
	sb.WriteString(kind)
	sb.WriteString("\nepochs=")
	sb.WriteString(strconv.Itoa(epochs))
	sb.WriteString("\narch=")
	sb.WriteString(arch)
	sb.WriteString("\nnam=")
	sb.WriteString(namVersion)
	sb.WriteString("\ndriver=")
	sb.WriteString(driverSHA)
	sb.WriteString("\nsignal=")
	sb.WriteString(signalSHA)
	sb.WriteString("\n")
	if base != "" {
		sb.WriteString("base=")
		sb.WriteString(base)
		sb.WriteString("\n")
	}
	if latency != nil {
		sb.WriteString("latency=")
		sb.WriteString(strconv.FormatInt(*latency, 10))
		sb.WriteString("\n")
	}
	return sb.String()
}
