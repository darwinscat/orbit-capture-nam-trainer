// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package testsupport holds helpers shared across test packages. It is imported
// only by _test files, so it never ends up in the daemon binary.
package testsupport

import (
	"bytes"
	"encoding/binary"
)

// WAV builds a canonical mono 16-bit PCM RIFF/WAVE file of the given sample rate
// and duration, filled with silence. It is the minimal valid capture the daemon
// accepts (subject to the sample-rate / duration checks in package wav).
func WAV(sampleRate int, seconds float64) []byte {
	const (
		channels      = 1
		bitsPerSample = 16
	)
	byteRate := sampleRate * channels * bitsPerSample / 8
	dataLen := int(float64(byteRate) * seconds)
	if dataLen%2 == 1 {
		dataLen++ // keep the data chunk word-aligned
	}

	buf := new(bytes.Buffer)
	le := binary.LittleEndian

	buf.WriteString("RIFF")
	_ = binary.Write(buf, le, uint32(36+dataLen))
	buf.WriteString("WAVE")

	buf.WriteString("fmt ")
	_ = binary.Write(buf, le, uint32(16))
	_ = binary.Write(buf, le, uint16(1)) // PCM
	_ = binary.Write(buf, le, uint16(channels))
	_ = binary.Write(buf, le, uint32(sampleRate))
	_ = binary.Write(buf, le, uint32(byteRate))
	_ = binary.Write(buf, le, uint16(channels*bitsPerSample/8)) // block align
	_ = binary.Write(buf, le, uint16(bitsPerSample))

	buf.WriteString("data")
	_ = binary.Write(buf, le, uint32(dataLen))
	buf.Write(make([]byte, dataLen))

	return buf.Bytes()
}

// ValidCapture is the smallest capture the daemon accepts: 48 kHz, exactly 30 s.
func ValidCapture() []byte { return WAV(48000, 30) }

// Distinct returns a copy of b with one data byte tweaked so its sha256 differs,
// while it stays a valid WAV (only silence is perturbed). Used to mint several
// distinct-but-valid captures in one test.
func Distinct(b []byte, seed byte) []byte {
	out := make([]byte, len(b))
	copy(out, b)
	if len(out) > 0 {
		out[len(out)-1] = seed // last byte is in the data (silence) region
	}
	return out
}
