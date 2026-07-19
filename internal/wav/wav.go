// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package wav validates an uploaded capture against the daemon's constraints
// (the design notes): a canonical RIFF/WAVE file, 48 kHz, 30 s..20 min, at
// most 200 MB. Any failure is a single error whose message is safe to hand back
// to the client (mapped to 422 wav_invalid). It parses only the header chunks —
// it never walks the multi-megabyte sample data.
package wav

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// Constraints (the design notes).
const (
	RequiredSampleRate = 48000
	MinDurationSec     = 30.0
	MaxDurationSec     = 20 * 60.0 // 20 minutes
	MaxSizeBytes       = 200 << 20 // 200 MiB
)

// Info reports the parsed, validated header facts.
type Info struct {
	SampleRate    int
	Channels      int
	BitsPerSample int
	DurationSec   float64
	DataBytes     int64
}

// Validate parses the header of a full WAV byte slice and enforces the
// constraints. It assumes the canonical layout (fmt chunk before data chunk),
// which every NAM capture uses; an unusual chunk ordering that hides fmt behind
// a huge data chunk is reported as "missing fmt chunk" rather than mis-parsed.
func Validate(b []byte) (Info, error) {
	if int64(len(b)) > MaxSizeBytes {
		return Info{}, fmt.Errorf("file too large: %d bytes (max %d)", len(b), MaxSizeBytes)
	}
	if len(b) < 44 { // smallest possible canonical WAV header
		return Info{}, errors.New("too short to be a WAV file")
	}
	if string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return Info{}, errors.New("not a RIFF/WAVE file")
	}

	var (
		haveFmt, haveData          bool
		sampleRate, channels, bits int
		byteRate                   uint32
		dataBytes                  int64
		dataBodyOff                int
	)

	// Chunks: 4-byte id, 4-byte little-endian size, then size bytes padded to an
	// even boundary. Start after "WAVE" at offset 12.
	for off := 12; off+8 <= len(b); {
		id := string(b[off : off+4])
		size := binary.LittleEndian.Uint32(b[off+4 : off+8])
		body := off + 8

		switch id {
		case "fmt ":
			if body+16 > len(b) {
				return Info{}, errors.New("truncated fmt chunk")
			}
			channels = int(binary.LittleEndian.Uint16(b[body+2 : body+4]))
			sampleRate = int(binary.LittleEndian.Uint32(b[body+4 : body+8]))
			byteRate = binary.LittleEndian.Uint32(b[body+8 : body+12])
			bits = int(binary.LittleEndian.Uint16(b[body+14 : body+16]))
			haveFmt = true
		case "data":
			dataBytes = int64(size)
			dataBodyOff = body
			haveData = true
		}
		if haveFmt && haveData {
			break
		}

		// Advance past this chunk (word-aligned). Guard against a bogus size that
		// would wrap or stall the scan.
		adv := int64(size)
		if adv%2 == 1 {
			adv++
		}
		next := int64(body) + adv
		if next <= int64(off) || next > int64(len(b)) {
			break
		}
		off = int(next)
	}

	if !haveFmt {
		return Info{}, errors.New("missing fmt chunk")
	}
	if !haveData {
		return Info{}, errors.New("missing data chunk")
	}
	// The declared data size must actually be present. Without this a 44-byte
	// header claiming minutes of samples would pass the duration gate below and
	// waste a training slot; the duration must reflect real bytes, not a claim.
	if dataBytes > int64(len(b))-int64(dataBodyOff) {
		return Info{}, fmt.Errorf("data chunk declares %d bytes but only %d are present (truncated capture)",
			dataBytes, int64(len(b))-int64(dataBodyOff))
	}
	if sampleRate != RequiredSampleRate {
		return Info{}, fmt.Errorf("sample rate %d Hz, require %d Hz", sampleRate, RequiredSampleRate)
	}
	if byteRate == 0 {
		return Info{}, errors.New("invalid byte rate (0)")
	}

	dur := float64(dataBytes) / float64(byteRate)
	if dur < MinDurationSec || dur > MaxDurationSec {
		return Info{}, fmt.Errorf("duration %.1fs out of range [%.0f, %.0f]s",
			dur, MinDurationSec, MaxDurationSec)
	}

	return Info{
		SampleRate:    sampleRate,
		Channels:      channels,
		BitsPerSample: bits,
		DurationSec:   dur,
		DataBytes:     dataBytes,
	}, nil
}
