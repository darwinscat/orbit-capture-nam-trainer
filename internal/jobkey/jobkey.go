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
//	  "signal=" + signalSHA   + "\n" )
//
// wavHex is the sha256 hex of the raw wav bytes; epochs must already be
// normalized for the kind (see jobs.NormalizeEpochs).
func Compute(wavHex, kind string, epochs int, arch, namVersion, driverSHA, signalSHA string) string {
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
	sum := sha256.Sum256([]byte(sb.String()))
	return hex.EncodeToString(sum[:])
}
