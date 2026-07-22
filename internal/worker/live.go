// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
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

// liveESRFromLogOK returns the driver's reported ESR for an epoch, read from job_log
// in REVERSE so a requeued attempt's duplicate epoch lines resolve to the
// LAST-appended value (crew F5). ok is false when the log is unreadable or has no
// finite epoch_esr line for that epoch — there is NO filename fallback, since a
// checkpoint_last name carries no ESR token (crew F10); the stop then stores a NULL
// esr, which is honest.
//
// A non-finite value ("nan" from a diverged earlier ATTEMPT of the same epoch number)
// is skipped, so after a requeue this scan can legitimately return an EARLIER
// attempt's finite value for the same epoch — accepted at cap=1 (documented, not
// engineered around).
func (p *Pool) liveESRFromLogOK(ctx context.Context, key string, epoch int64) (float64, bool) {
	lines, err := p.store.JobLog(ctx, key)
	if err != nil {
		return 0, false
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
		// never be pinned; keep scanning (crew F-B).
		if v, ok := parseFloatLenient(firstField(strings.TrimSpace(lines[i][at+len(marker):]))); ok &&
			!math.IsNaN(v) && !math.IsInf(v, 0) {
			return v, true
		}
	}
	return 0, false
}

// liveESRFromLog is the ExportLive form of the above: the log value when there is
// one, else the caller's filename ESR fallback (best-name ESR tokens always exist).
func (p *Pool) liveESRFromLog(ctx context.Context, key string, epoch int64, fallback float64) float64 {
	if v, ok := p.liveESRFromLogOK(ctx, key, epoch); ok {
		return v
	}
	return fallback
}

// harvestStop selects the checkpoint pair an early-stopped train keeps and finishes as
// a NORMAL succeeded run (crew F6). "What you keep is what you hear and where you
// resume." The decision tree, in order:
//
//  1. the LAST pair, newest epoch first — a candidate qualifies when its .ckpt opens
//     as a zip AND its same-stem .nam is valid json (see qualifyPair). The first
//     qualifying last-pair wins.
//  2. no last pair qualified → the BEST pair (selectBestCkpt's min-ESR winner), same
//     zip+json gates.
//  3. neither → ok=false; classify then falls through to the outdir-completed fallback,
//     else stop_failed.
//
// reached = the chosen pair's epoch + 1 in every branch (a ckpt named epoch=K resumes
// at K+1, so reached is exactly the child's first computed epoch). esr = that epoch's
// job_log value, nil when the line is unavailable (harvestStop passes *float64 straight
// to FinishStopped, which stores NULL). The returned nam/ckpt are the exact bytes
// qualifyPair read, so no second read can race a torn rotation.
func (p *Pool) harvestStop(ctx context.Context, key, scratch string) (nam, ckpt []byte, esr *float64, reached int64, ok bool) {
	for _, c := range selectLastCkpt(scratch) {
		if n, ck, good := qualifyPair(c.path); good {
			return n, ck, p.stopESR(ctx, key, c.epoch), c.epoch + 1, true
		}
	}
	if best, found := selectBestCkpt(scratch); found {
		if n, ck, good := qualifyPair(best.path); good {
			return n, ck, p.stopESR(ctx, key, best.epoch), best.epoch + 1, true
		}
	}
	return nil, nil, nil, 0, false
}

// stopESR resolves a stopped epoch's validation ESR, nil when its log line is
// unavailable (FinishStopped then stores NULL — see liveESRFromLogOK's honesty note).
func (p *Pool) stopESR(ctx context.Context, key string, epoch int64) *float64 {
	if v, ok := p.liveESRFromLogOK(ctx, key, epoch); ok {
		return &v
	}
	return nil
}

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
