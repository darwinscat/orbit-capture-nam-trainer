// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package wav_test

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"

	"orbit-capture-nam-trainer/internal/testsupport"
	"orbit-capture-nam-trainer/internal/wav"
)

func TestValidateAcceptsCanonicalCapture(t *testing.T) {
	info, err := wav.Validate(testsupport.WAV(48000, 45))
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if info.SampleRate != 48000 {
		t.Errorf("sample rate = %d, want 48000", info.SampleRate)
	}
	if info.DurationSec < 44.9 || info.DurationSec > 45.1 {
		t.Errorf("duration = %.2f, want ~45", info.DurationSec)
	}
}

func TestValidateRejects(t *testing.T) {
	cases := []struct {
		name   string
		input  []byte
		errHas string
	}{
		{"empty", nil, "too short"},
		{"not riff", []byte(strings.Repeat("x", 64)), "RIFF/WAVE"},
		{"wrong sample rate", testsupport.WAV(44100, 45), "sample rate"},
		{"too short duration", testsupport.WAV(48000, 10), "duration"},
		{"too long duration", testsupport.WAV(48000, 20*60+5), "duration"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := wav.Validate(tc.input)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.errHas)
			}
			if !strings.Contains(err.Error(), tc.errHas) {
				t.Errorf("error = %q, want it to contain %q", err.Error(), tc.errHas)
			}
		})
	}
}

// TestValidateRejectsFabricatedDataSize covers the header-only bypass: a 44-byte
// blob whose data chunk CLAIMS minutes of samples that are not actually present.
// The declared-size gate must reject it rather than trusting the claim.
func TestValidateRejectsFabricatedDataSize(t *testing.T) {
	const (
		sampleRate = 48000
		byteRate   = sampleRate * 1 * 16 / 8 // mono, 16-bit
	)
	fakeData := uint32(60 * byteRate) // claims 60 s

	buf := new(bytes.Buffer)
	le := binary.LittleEndian
	buf.WriteString("RIFF")
	_ = binary.Write(buf, le, uint32(36+fakeData))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(buf, le, uint32(16))
	_ = binary.Write(buf, le, uint16(1)) // PCM
	_ = binary.Write(buf, le, uint16(1)) // mono
	_ = binary.Write(buf, le, uint32(sampleRate))
	_ = binary.Write(buf, le, uint32(byteRate))
	_ = binary.Write(buf, le, uint16(2))  // block align
	_ = binary.Write(buf, le, uint16(16)) // bits
	buf.WriteString("data")
	_ = binary.Write(buf, le, fakeData) // header claims 60 s but no bytes follow

	_, err := wav.Validate(buf.Bytes())
	if err == nil {
		t.Fatal("expected rejection of a header-only WAV with a fabricated data size")
	}
	if !strings.Contains(err.Error(), "truncated") {
		t.Errorf("error = %q, want it to mention truncation", err.Error())
	}
}

func TestValidateBoundaryDurations(t *testing.T) {
	// Exactly 30 s and exactly 20 min must pass; the generator is exact.
	if _, err := wav.Validate(testsupport.WAV(48000, 30)); err != nil {
		t.Errorf("30s should pass: %v", err)
	}
	if _, err := wav.Validate(testsupport.WAV(48000, 20*60)); err != nil {
		t.Errorf("20min should pass: %v", err)
	}
}
