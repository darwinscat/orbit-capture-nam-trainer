// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"bufio"
	"io"
	"math"
	"strconv"
	"strings"
	"time"
)

// parseEpoch mirrors the desktop app's parseEpoch: find "Epoch " (case-sensitive,
// so it never matches the "DRIVER: training … epochs=" banner) and read the
// leading run of digits after it. Returns the 0-based epoch, or -1 if absent.
func parseEpoch(line string) int {
	at := strings.Index(line, "Epoch ")
	if at < 0 {
		return -1
	}
	rest := line[at+len("Epoch "):]
	end := 0
	for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
		end++
	}
	if end == 0 {
		return -1
	}
	n, err := strconv.Atoi(rest[:end])
	if err != nil {
		return -1
	}
	return n
}

// parseEpochESR reads the driver's own per-epoch validation figure straight off the
// stream: "DRIVER: epoch_esr=<epoch>=<value>". The trailing '=' fences the epoch the
// same way liveESRFromLogOK does, and a non-finite value is refused — a diverged epoch
// must never be pinned as the run's ESR.
func parseEpochESR(line string) (epoch int64, esr float64, ok bool) {
	const marker = "DRIVER: epoch_esr="
	at := strings.Index(line, marker)
	if at < 0 {
		return 0, 0, false
	}
	rest := line[at+len(marker):]
	eq := strings.Index(rest, "=")
	if eq <= 0 {
		return 0, 0, false
	}
	e, err := strconv.ParseInt(rest[:eq], 10, 64)
	if err != nil {
		return 0, 0, false
	}
	v, good := parseFloatLenient(firstField(strings.TrimSpace(rest[eq+1:])))
	if !good || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, 0, false
	}
	return e, v, true
}

const replicateESRPhrase = "Replicate ESR is"

// parseReplicateESR reads the self-ESR value from a "Replicate ESR is X" line
// (NAM's own check output). ParseFloat handles the scientific notation NAM can
// emit (e.g. 9.3e-05).
func parseReplicateESR(line string) (float64, bool) {
	at := strings.Index(line, replicateESRPhrase)
	if at < 0 {
		return 0, false
	}
	// NAM prints "Replicate ESR is 0.00003540." — with a trailing period. Parse
	// leniently (like the app's getDoubleValue) so that period doesn't lose the
	// value; also tolerates scientific notation (e.g. 9.3e-05).
	return parseFloatLenient(firstField(strings.TrimSpace(line[at+len(replicateESRPhrase):])))
}

// parseFloatLenient parses the leading float in tok, ignoring trailing
// punctuation such as a sentence period or comma.
func parseFloatLenient(tok string) (float64, bool) {
	tok = strings.TrimRight(tok, ".,;")
	v, err := strconv.ParseFloat(tok, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// isProbeFail reports whether a line is a self-ESR FAIL verdict marker.
func isProbeFail(line string) bool {
	return strings.Contains(line, "DRIVER: checkfail") ||
		strings.Contains(line, "Failed checks") ||
		strings.Contains(line, "doesn't sound like itself")
}

const driverESRPrefix = "DRIVER: esr="

// parseDriverESR reads a "DRIVER: esr=<float|na>" line. isNA is true for the
// literal "na" token (→ failed/no_esr); ok is false if the line isn't an esr line
// or the value doesn't parse.
func parseDriverESR(line string) (value float64, isNA, ok bool) {
	at := strings.Index(line, driverESRPrefix)
	if at < 0 {
		return 0, false, false
	}
	tok := firstField(strings.TrimSpace(line[at+len(driverESRPrefix):]))
	if strings.HasPrefix(tok, "n") { // "na"
		return 0, true, true
	}
	v, ok := parseFloatLenient(tok)
	if !ok {
		return 0, false, false
	}
	return v, false, true
}

func firstField(s string) string {
	if i := strings.IndexAny(s, " \t"); i >= 0 {
		return s[:i]
	}
	return s
}

// epochTracker turns a stream of parsed epoch numbers into a throttled
// s/epoch EWMA. The baseline is the FIRST epoch line (never spawn time), so the
// minutes-long silent torch import is not folded into the first estimate. A delta
// is taken only on a strict increase and small (<50 ms) buffered bursts are
// ignored — both mirror the desktop app's trainer.
type epochTracker struct {
	have      bool
	lastEpoch int
	lastAt    time.Time
	sPerEpoch float64
}

// observe records an epoch reading. changed is true when the epoch value advanced
// (so the caller should persist epoch + s/epoch).
func (e *epochTracker) observe(ep int, now time.Time) (changed bool) {
	if !e.have {
		e.have = true
		e.lastEpoch = ep
		e.lastAt = now
		return true // first epoch line: baseline set, epoch is new
	}
	if ep <= e.lastEpoch {
		return false
	}
	d := now.Sub(e.lastAt).Seconds() / float64(ep-e.lastEpoch)
	if d > 0.05 { // ignore sub-50ms buffered bursts (they zero the ETA)
		if e.sPerEpoch <= 0 {
			e.sPerEpoch = d
		} else {
			e.sPerEpoch = 0.7*e.sPerEpoch + 0.3*d
		}
	}
	e.lastEpoch = ep
	e.lastAt = now
	return true
}

// lineReader yields output lines split on \n, \r, or \r\n, tolerating unbounded
// lines (excess beyond max is dropped rather than failing the read loop). It
// returns io.EOF once the stream is drained.
type lineReader struct {
	r   *bufio.Reader
	buf []byte
	max int
}

func newLineReader(r io.Reader) *lineReader {
	return &lineReader{r: bufio.NewReaderSize(r, 64<<10), max: 1 << 20}
}

func (lr *lineReader) next() (string, error) {
	for {
		b, err := lr.r.ReadByte()
		if err != nil {
			if len(lr.buf) > 0 {
				line := string(lr.buf)
				lr.buf = lr.buf[:0]
				return line, nil
			}
			return "", err
		}
		if b == '\n' || b == '\r' {
			if b == '\r' { // collapse a following \n of a \r\n pair
				if nb, e := lr.r.ReadByte(); e == nil && nb != '\n' {
					_ = lr.r.UnreadByte()
				}
			}
			line := string(lr.buf)
			lr.buf = lr.buf[:0]
			return line, nil
		}
		if len(lr.buf) < lr.max {
			lr.buf = append(lr.buf, b)
		}
	}
}
