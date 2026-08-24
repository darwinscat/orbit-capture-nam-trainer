// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// The best-checkpoint filename shape (PL 2.6.1 auto_insert_metric_name, nam 0.13
// ModelCheckpoint), e.g.
//
//	checkpoint_best_epoch=0031_step=1984_ESR=0.00125_MSE=1.389e-03.ckpt
//
// The ESR renders scientific below 1e-4 (e.g. 3.5e-05) and is "nan" for a diverged
// run. reCkptESR stops at the underscore before _MSE, so it captures exactly the
// ESR token; reCkptEpoch reads the absolute epoch (train_more numbers from
// start_epoch). These parse the REAL rendered name — not the brief's unrendered
// {epoch:04d} template (crew F3).
var (
	reCkptEpoch = regexp.MustCompile(`epoch=(\d+)`)
	reCkptESR   = regexp.MustCompile(`_ESR=([^_]+)`)
)

// ckptChoice is the winner of a best-checkpoint scan.
type ckptChoice struct {
	path  string  // absolute path to the .ckpt
	name  string  // its basename — the cache identity
	epoch int64   // absolute epoch parsed from the name
	esr   float64 // ESR parsed from the name (the fallback if the log has none)
}

// parseCkptName extracts the absolute epoch and validation ESR from a checkpoint
// filename. ok is false when either token is missing/unparseable OR the ESR is
// non-finite ("nan"/inf) — a diverged run must never win the min-ESR scan.
func parseCkptName(name string) (epoch int64, esr float64, ok bool) {
	em := reCkptEpoch.FindStringSubmatch(name)
	if em == nil {
		return 0, 0, false
	}
	ep, err := strconv.ParseInt(em[1], 10, 64)
	if err != nil {
		return 0, 0, false
	}
	sm := reCkptESR.FindStringSubmatch(name)
	if sm == nil {
		return 0, 0, false
	}
	// ParseFloat("nan") succeeds with a NaN value, so the IsNaN/IsInf guard — not the
	// parse error — is what skips a diverged run's checkpoint.
	v, err := strconv.ParseFloat(sm[1], 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, 0, false
	}
	return ep, v, true
}

// selectBestCkpt is the pure selection over an attempt's scratch dir: it globs
// <scratch>/out/**/checkpoints/checkpoint_best_*.ckpt, parses each name, skips the
// unparseable and the nan-ESR ones, and returns the minimum-ESR checkpoint. A tie on
// equal ESR breaks toward the LOWER epoch — matching ModelCheckpoint's first-best
// wins. ok is false when the dir is absent/empty or holds no parseable best ckpt.
//
// Only *.ckpt files are candidates, so the same-stem .nam sibling ModelCheckpoint
// writes beside each checkpoint is never mistaken for one.
func selectBestCkpt(scratch string) (ckptChoice, bool) {
	root := filepath.Join(scratch, "out")
	var best ckptChoice
	found := false
	// WalkDir errors (an unreadable subtree, a dir vanishing mid-walk during teardown)
	// are swallowed per-entry so a partial tree still yields the best visible ckpt;
	// an absent root simply visits nothing.
	_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		name := d.Name()
		if !strings.HasPrefix(name, "checkpoint_best_") || !strings.HasSuffix(name, ".ckpt") {
			return nil
		}
		if filepath.Base(filepath.Dir(path)) != "checkpoints" {
			return nil
		}
		ep, esr, ok := parseCkptName(name)
		if !ok {
			return nil
		}
		if !found || esr < best.esr || (esr == best.esr && ep < best.epoch) {
			best = ckptChoice{path: path, name: name, epoch: ep, esr: esr}
			found = true
		}
		return nil
	})
	return best, found
}

// selectLastCkpt is selectBestCkpt's sibling for the early-stop harvest: it globs
// <scratch>/out/**/checkpoints/checkpoint_last_*.ckpt and returns EVERY candidate
// sorted by epoch DESC. Unlike the best names these carry NO _ESR= token (the real
// shape is checkpoint_last_epoch=0039_step=2480.ckpt), so only the epoch is parsed and
// ckptChoice.esr is left zero — the stop's ESR comes from job_log, never a last-name
// token. Usually one candidate; transiently TWO mid-rotation, because PL saves the new
// checkpoint_last BEFORE removing the previous one — returning ALL is exactly what lets
// the harvest walk them newest-first and take the first pair that is intact (a SIGKILL
// can freeze the newest ckpt torn while the previous pair sits whole one epoch back).
//
// Same guards as selectBestCkpt: only *.ckpt files whose parent dir is "checkpoints",
// WalkDir errors swallowed per-entry, an absent root simply visits nothing.
func selectLastCkpt(scratch string) []ckptChoice {
	root := filepath.Join(scratch, "out")
	var out []ckptChoice
	_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		name := d.Name()
		if !strings.HasPrefix(name, "checkpoint_last_") || !strings.HasSuffix(name, ".ckpt") {
			return nil
		}
		if filepath.Base(filepath.Dir(path)) != "checkpoints" {
			return nil
		}
		em := reCkptEpoch.FindStringSubmatch(name)
		if em == nil {
			return nil
		}
		ep, perr := strconv.ParseInt(em[1], 10, 64)
		if perr != nil {
			return nil
		}
		out = append(out, ckptChoice{path: path, name: name, epoch: ep})
		return nil
	})
	// Newest epoch first; the name (which carries the step) is a deterministic
	// tiebreak for the vanishingly rare same-epoch collision.
	sort.Slice(out, func(i, j int) bool {
		if out[i].epoch != out[j].epoch {
			return out[i].epoch > out[j].epoch
		}
		return out[i].name > out[j].name
	})
	return out
}

// (ExportLive, its one-snapshot cache, harvestStop and the log-scanning ESR helpers used to live
// here, and all of them are gone.
//
// ExportLive answered "let me hear this run as it stands" by exporting a .nam on demand and storing
// it as a row of its own — a conversation that existed because the weights lived on one machine's
// disk. They live in the library now, refreshed every epoch, with the best pair kept beside the last
// one for exactly this listener, so the answer is a SELECT and nobody has to be asked.
//
// harvestStop went hunting through the scratch directory at the moment of the kill for the newest
// pair that was not half-written. That selection still matters and still lives below — but it belongs
// to the saver, once per epoch, where being wrong costs seconds instead of a run. Stopping is reading
// a row.
//
// And the ESR for a stopped epoch was found by reading the job's entire log. It comes from one
// indexed row of job_epochs now — see store.EpochESR.)

// qualifyPair validates one checkpoint pair for the harvest, returning the exact bytes
// on success. The .ckpt must open as a zip: torch checkpoints ARE zip archives whose
// end-of-central-directory record is written LAST, so a SIGKILL that froze a partial
// write has no EOCD and fails to open — that is how a torn newest-last is rejected in
// favour of the intact previous pair. The same-stem .nam must be valid json: nam writes
// the .ckpt first and its .nam sibling second, so a kill can leave a good ckpt beside a
// missing/torn sibling.
func qualifyPair(ckptPath string) (nam, ckpt []byte, ok bool) {
	ckpt, err := os.ReadFile(ckptPath)
	if err != nil {
		return nil, nil, false
	}
	if _, err := zip.NewReader(bytes.NewReader(ckpt), int64(len(ckpt))); err != nil {
		return nil, nil, false
	}
	nam, err = os.ReadFile(strings.TrimSuffix(ckptPath, ".ckpt") + ".nam")
	if err != nil || !json.Valid(nam) {
		return nil, nil, false
	}
	return nam, ckpt, true
}
