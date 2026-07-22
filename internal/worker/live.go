// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// Sentinel errors ExportLive returns so the HTTP layer can map them (errors.Is)
// without importing any worker internals. Everything else is a real read error.
var (
	// ErrNoLiveJob: the key has no registered running attempt (queued, unknown, or
	// already terminal → the entry has been unregistered). The HTTP layer treats a
	// running-but-not-live-exportable job like an old daemon would (404 not_found).
	ErrNoLiveJob = errors.New("worker: no live job for key")
	// ErrNoCheckpoint: a live attempt exists but has produced no best-so-far
	// checkpoint yet — before the first completed epoch, a probe_self (never
	// checkpoints), or the run's final teardown seconds. HTTP 404 no_checkpoint.
	ErrNoCheckpoint = errors.New("worker: no live checkpoint yet")
	// ErrLiveTransient: the best checkpoint's .nam sibling was missing or torn
	// mid-read (a rotation/torn-write race) and there is no prior good snapshot to
	// fall back to. HTTP 500 internal — the client may retry.
	ErrLiveTransient = errors.New("worker: live snapshot transient read failure")
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

// ExportLive serves the best-so-far checkpoint's .nam for a RUNNING train-lane job:
// the same best-checkpoint rule the trainer tracks, for auditioning a live run. It
// is read-only — file reads plus one DB read, no process ever touched.
//
// It captures the attempt entry ONCE (nil → ErrNoLiveJob), scans for the best
// checkpoint (none → ErrNoCheckpoint), and:
//   - identity unchanged since the last export → serves the entry's one-snapshot
//     cache, with no file read at all;
//   - otherwise reads the checkpoint's same-stem <stem>.nam and json.Valid-guards it
//     against a torn write; valid → fills the cache through the CAPTURED entry and
//     serves it; missing/invalid (a rotation/torn-write race) → serves the cached
//     last-good if any, else ErrLiveTransient.
//
// The returned epoch is the snapshot's absolute epoch. The returned ESR prefers the
// driver's higher-precision epoch_esr from job_log (reverse-scanned — a requeued
// attempt appends duplicate epochs, so the LAST occurrence wins, crew F5), falling
// back to the filename ESR.
func (p *Pool) ExportLive(ctx context.Context, key string) (nam []byte, epoch int64, esr float64, err error) {
	p.mu.Lock()
	e := p.procs[key]
	p.mu.Unlock()
	if e == nil {
		return nil, 0, 0, ErrNoLiveJob
	}
	// An entry without a scratch dir (a test-constructed procEntry) must never walk
	// a CWD-relative "out" — enforce the documented contract (crew F-C).
	if e.scratch == "" {
		return nil, 0, 0, ErrNoCheckpoint
	}
	// NB (crew F-A, accepted + documented): at cap>=2 a delete+resubmit of the same
	// content key has a ~ms window where this lookup still sees the OLD attempt's
	// entry (its worker is in the teardown tail) while a NEW attempt already runs —
	// one poll can serve the old snapshot with old headers. At the default cap=1
	// the window cannot exist (one train worker), and content-addressing means both
	// attempts trained the same input; accepted over adding a per-request row fence.

	best, ok := selectBestCkpt(e.scratch)
	if !ok {
		return nil, 0, 0, ErrNoCheckpoint
	}

	e.snapMu.Lock()
	cached := e.snap
	e.snapMu.Unlock()

	// Same best checkpoint as last time → serve the cache; no re-read, no log scan.
	if cached != nil && cached.identity == best.name {
		return cached.nam, cached.epoch, cached.esr, nil
	}

	raw, rerr := os.ReadFile(strings.TrimSuffix(best.path, ".ckpt") + ".nam")
	if rerr != nil || !json.Valid(raw) {
		// The sibling rotated out from under us or is a torn write: fall back to the
		// last good snapshot if we have one, else report a transient failure.
		if cached != nil {
			return cached.nam, cached.epoch, cached.esr, nil
		}
		return nil, 0, 0, ErrLiveTransient
	}

	snap := &liveSnapshot{
		identity: best.name,
		nam:      raw,
		epoch:    best.epoch,
		esr:      p.liveESRFromLog(ctx, key, best.epoch, best.esr),
	}
	e.snapMu.Lock()
	e.snap = snap
	e.snapMu.Unlock()
	return snap.nam, snap.epoch, snap.esr, nil
}

// liveESRFromLog returns the driver's reported ESR for the snapshot's epoch, read
// from job_log in REVERSE so a requeued attempt's duplicate epoch lines resolve to
// the LAST-appended value (crew F5). It falls back to the caller's filename ESR when
// the log is unreadable or has no epoch_esr line for that epoch.
func (p *Pool) liveESRFromLog(ctx context.Context, key string, epoch int64, fallback float64) float64 {
	lines, err := p.store.JobLog(ctx, key)
	if err != nil {
		return fallback
	}
	// The trailing '=' fences the epoch: "epoch_esr=3=" never matches "epoch_esr=30=".
	marker := fmt.Sprintf("DRIVER: epoch_esr=%d=", epoch)
	for i := len(lines) - 1; i >= 0; i-- {
		at := strings.Index(lines[i], marker)
		if at < 0 {
			continue
		}
		// ParseFloat("nan") succeeds, and a diverged earlier ATTEMPT can have logged
		// `epoch_esr=<k>=nan` for the same epoch number — a non-finite value must
		// never be pinned into a served header; keep scanning (crew F-B).
		if v, ok := parseFloatLenient(firstField(strings.TrimSpace(lines[i][at+len(marker):]))); ok &&
			!math.IsNaN(v) && !math.IsInf(v, 0) {
			return v
		}
	}
	return fallback
}
