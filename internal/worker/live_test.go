// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"orbit-capture-nam-trainer/internal/jobs"
)

// ckptName renders a REAL best-checkpoint filename (PL 2.6.1 auto_insert_metric_name
// shape). esr is passed verbatim so tests can exercise decimal, scientific, and nan.
func ckptName(epoch int, esr string) string {
	return fmt.Sprintf("checkpoint_best_epoch=%04d_step=64_ESR=%s_MSE=1.0e-03.ckpt", epoch, esr)
}

// mkCkpt writes a fabricated checkpoint under <scratch>/out/<sub>/checkpoints/ (the
// **-depth the real trainer nests under .train-work-*). A non-empty nam writes the
// same-stem .nam sibling with that content; "" writes no sibling (the pre-sibling /
// rotation-ENOENT case).
func mkCkpt(t *testing.T, scratch, sub, name, nam string) {
	t.Helper()
	dir := filepath.Join(scratch, "out", sub, "checkpoints")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte("CKPT-"+name), 0o644); err != nil {
		t.Fatal(err)
	}
	if nam != "" {
		stem := strings.TrimSuffix(name, ".ckpt")
		if err := os.WriteFile(filepath.Join(dir, stem+".nam"), []byte(nam), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func approx(a, b float64) bool { return math.Abs(a-b) <= math.Abs(b)*1e-9+1e-18 }

// --- selection: parsing REAL rendered names ---

func TestParseCkptName(t *testing.T) {
	cases := []struct {
		name    string
		wantEp  int64
		wantESR float64
		wantOK  bool
	}{
		{"checkpoint_best_epoch=0031_step=1984_ESR=0.00125_MSE=1.389e-03.ckpt", 31, 0.00125, true},
		{"checkpoint_best_epoch=0007_step=448_ESR=3.5e-05_MSE=2.0e-06.ckpt", 7, 3.5e-05, true}, // scientific
		{"checkpoint_best_epoch=0000_step=64_ESR=nan_MSE=nan.ckpt", 0, 0, false},               // diverged → skipped
		{"checkpoint_best_epoch=0123_step=64_ESR=NaN_MSE=1.0e-03.ckpt", 0, 0, false},           // case-insensitive nan
		{"checkpoint_last_epoch=0031_step=64.ckpt", 0, 0, false},                               // no _ESR= token
		{"random_file.ckpt", 0, 0, false},                                                      // neither token
	}
	for _, c := range cases {
		ep, esr, ok := parseCkptName(c.name)
		if ok != c.wantOK {
			t.Errorf("parseCkptName(%q) ok=%v, want %v", c.name, ok, c.wantOK)
			continue
		}
		if !ok {
			continue
		}
		if ep != c.wantEp {
			t.Errorf("parseCkptName(%q) epoch=%d, want %d", c.name, ep, c.wantEp)
		}
		if !approx(esr, c.wantESR) {
			t.Errorf("parseCkptName(%q) esr=%v, want %v", c.name, esr, c.wantESR)
		}
	}
}

// --- selection: min-ESR scan over fabricated dirs ---

func TestSelectBestCkpt(t *testing.T) {
	t.Run("min esr wins including scientific and nested depth", func(t *testing.T) {
		s := t.TempDir()
		// Nested arbitrary depth under out/, matching out/.train-work-*/version_0/checkpoints.
		mkCkpt(t, s, ".train-work-abc/version_0", ckptName(10, "0.00125"), "")
		mkCkpt(t, s, ".train-work-abc/version_0", ckptName(31, "3.5e-05"), "") // scientific min → wins
		mkCkpt(t, s, ".train-work-abc/version_0", ckptName(20, "0.5"), "")
		best, ok := selectBestCkpt(s)
		if !ok || best.epoch != 31 {
			t.Fatalf("best=%+v ok=%v, want epoch 31 (min ESR 3.5e-05)", best, ok)
		}
		if !approx(best.esr, 3.5e-05) {
			t.Errorf("best esr=%v, want 3.5e-05", best.esr)
		}
	})

	t.Run("nan is skipped", func(t *testing.T) {
		s := t.TempDir()
		mkCkpt(t, s, "v0", ckptName(5, "nan"), "") // diverged, must not win the min-scan
		mkCkpt(t, s, "v0", ckptName(9, "0.02"), "")
		best, ok := selectBestCkpt(s)
		if !ok || best.epoch != 9 {
			t.Fatalf("best=%+v ok=%v, want epoch 9 (nan skipped)", best, ok)
		}
	})

	t.Run("all nan yields none", func(t *testing.T) {
		s := t.TempDir()
		mkCkpt(t, s, "v0", ckptName(3, "nan"), "")
		if _, ok := selectBestCkpt(s); ok {
			t.Error("an all-nan dir must yield no best ckpt")
		}
	})

	t.Run("tie breaks to lower epoch", func(t *testing.T) {
		s := t.TempDir()
		mkCkpt(t, s, "v0", ckptName(40, "0.00100"), "")
		mkCkpt(t, s, "v0", ckptName(12, "0.00100"), "") // equal ESR, lower epoch → wins
		best, ok := selectBestCkpt(s)
		if !ok || best.epoch != 12 {
			t.Fatalf("best=%+v ok=%v, want epoch 12 (tie → lower epoch)", best, ok)
		}
	})

	t.Run("nam sibling is never a candidate", func(t *testing.T) {
		s := t.TempDir()
		mkCkpt(t, s, "v0", ckptName(30, "0.5"), `{"ok":true}`) // the real .ckpt (higher ESR)
		// A trap: a bare best-shaped .nam with a much LOWER ESR and no .ckpt beside it.
		// If the scan matched .nam files this would wrongly win.
		trap := "checkpoint_best_epoch=0001_step=64_ESR=0.00001_MSE=1.0e-03.nam"
		if err := os.WriteFile(filepath.Join(s, "out", "v0", "checkpoints", trap), []byte(`{"trap":true}`), 0o644); err != nil {
			t.Fatal(err)
		}
		best, ok := selectBestCkpt(s)
		if !ok || best.epoch != 30 {
			t.Fatalf("best=%+v ok=%v, want epoch 30 (.nam trap excluded)", best, ok)
		}
		if !strings.HasSuffix(best.name, ".ckpt") {
			t.Errorf("best.name=%q, want a .ckpt", best.name)
		}
	})

	t.Run("stray best-shaped file outside checkpoints is ignored", func(t *testing.T) {
		s := t.TempDir()
		// Directly under out/, NOT under a checkpoints/ dir → excluded by the glob shape.
		if err := os.MkdirAll(filepath.Join(s, "out"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(s, "out", ckptName(1, "0.00001")), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		mkCkpt(t, s, "v0", ckptName(7, "0.02"), "")
		best, ok := selectBestCkpt(s)
		if !ok || best.epoch != 7 {
			t.Fatalf("best=%+v ok=%v, want epoch 7 (stray out/ file ignored)", best, ok)
		}
	})

	t.Run("absent scratch yields none", func(t *testing.T) {
		if _, ok := selectBestCkpt(filepath.Join(t.TempDir(), "does-not-exist")); ok {
			t.Error("absent scratch must yield no best ckpt")
		}
	})

	t.Run("empty out dir yields none", func(t *testing.T) {
		s := t.TempDir()
		if err := os.MkdirAll(filepath.Join(s, "out"), 0o755); err != nil {
			t.Fatal(err)
		}
		if _, ok := selectBestCkpt(s); ok {
			t.Error("an empty out dir must yield no best ckpt")
		}
	})
}

// --- selection: last-ckpt scan over REAL rendered names (no _ESR= token) ---

func TestSelectLastCkpt(t *testing.T) {
	t.Run("real name parses the epoch, no ESR token needed", func(t *testing.T) {
		s := t.TempDir()
		mkCkpt(t, s, ".train-work-x/version_0", "checkpoint_last_epoch=0039_step=2480.ckpt", "{}")
		got := selectLastCkpt(s)
		if len(got) != 1 || got[0].epoch != 39 {
			t.Fatalf("got %+v, want one candidate at epoch 39", got)
		}
	})

	t.Run("candidates sorted epoch DESC (mid-rotation pair)", func(t *testing.T) {
		s := t.TempDir()
		mkCkpt(t, s, "v0", "checkpoint_last_epoch=0004_step=248.ckpt", "{}")
		mkCkpt(t, s, "v0", "checkpoint_last_epoch=0005_step=310.ckpt", "{}")
		got := selectLastCkpt(s)
		if len(got) != 2 || got[0].epoch != 5 || got[1].epoch != 4 {
			t.Fatalf("got %+v, want [5, 4] (newest first)", got)
		}
	})

	t.Run("best-shaped names are never last candidates", func(t *testing.T) {
		s := t.TempDir()
		mkCkpt(t, s, "v0", ckptName(7, "0.02"), "{}") // a checkpoint_best_* name
		if got := selectLastCkpt(s); len(got) != 0 {
			t.Errorf("got %+v, want none (best names excluded from the last scan)", got)
		}
	})

	t.Run("last-shaped file outside a checkpoints dir is ignored", func(t *testing.T) {
		s := t.TempDir()
		if err := os.MkdirAll(filepath.Join(s, "out"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(s, "out", "checkpoint_last_epoch=0002_step=1.ckpt"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		if got := selectLastCkpt(s); len(got) != 0 {
			t.Errorf("got %+v, want none (stray out/ file ignored)", got)
		}
	})

	t.Run("absent scratch yields none", func(t *testing.T) {
		if got := selectLastCkpt(filepath.Join(t.TempDir(), "nope")); len(got) != 0 {
			t.Errorf("got %+v, want none", got)
		}
	})
}

// --- ExportLive ---

// A key with no registered running attempt (queued/unknown, or terminal → the entry
// has been unregistered) → ErrNoLiveJob.
func TestExportLiveNoLiveJob(t *testing.T) {
	h := newHarness(t, "", 0)
	if _, _, _, err := h.pool.ExportLive(context.Background(), "k"); !errors.Is(err, ErrNoLiveJob) {
		t.Fatalf("unknown key err=%v, want ErrNoLiveJob", err)
	}
	// Register then unregister (the terminal-job teardown) → still ErrNoLiveJob.
	e := &procEntry{scratch: t.TempDir()}
	h.pool.register("k", e)
	h.pool.unregister("k", e)
	if _, _, _, err := h.pool.ExportLive(context.Background(), "k"); !errors.Is(err, ErrNoLiveJob) {
		t.Fatalf("after unregister err=%v, want ErrNoLiveJob", err)
	}
}

// A live attempt with no best checkpoint yet (before the first completed epoch) →
// ErrNoCheckpoint.
func TestExportLiveNoCheckpoint(t *testing.T) {
	h := newHarness(t, "", 0)
	s := t.TempDir()
	if err := os.MkdirAll(filepath.Join(s, "out"), 0o755); err != nil {
		t.Fatal(err)
	}
	h.pool.register("k", &procEntry{scratch: s})
	if _, _, _, err := h.pool.ExportLive(context.Background(), "k"); !errors.Is(err, ErrNoCheckpoint) {
		t.Fatalf("err=%v, want ErrNoCheckpoint", err)
	}
}

// The best snapshot is served, and when a strictly-better checkpoint appears the
// bytes, identity, and epoch all advance. Absent a log, the ESR is the filename value.
func TestExportLiveServesBestAndAdvances(t *testing.T) {
	h := newHarness(t, "", 0)
	s := t.TempDir()
	h.pool.register("k", &procEntry{scratch: s})

	mkCkpt(t, s, "v0", ckptName(10, "0.00200"), `{"gen":"A"}`)
	nam, ep, esr, err := h.pool.ExportLive(context.Background(), "k")
	if err != nil {
		t.Fatalf("export A: %v", err)
	}
	if string(nam) != `{"gen":"A"}` || ep != 10 {
		t.Fatalf("A: nam=%q epoch=%d, want gen A / 10", nam, ep)
	}
	if !approx(esr, 0.002) {
		t.Errorf("A esr=%v, want 0.002 (filename fallback, no log)", esr)
	}

	mkCkpt(t, s, "v0", ckptName(25, "0.00050"), `{"gen":"B"}`) // strictly better ESR
	nam, ep, _, err = h.pool.ExportLive(context.Background(), "k")
	if err != nil {
		t.Fatalf("export B: %v", err)
	}
	if string(nam) != `{"gen":"B"}` || ep != 25 {
		t.Fatalf("B: nam=%q epoch=%d, want gen B / 25", nam, ep)
	}
}

// A worse (higher-ESR) checkpoint appears so the best identity is UNCHANGED. Rewriting
// the best's .nam on disk and still getting the ORIGINAL bytes proves the cache hit
// re-reads nothing.
func TestExportLiveCacheServesStaleOnUnchangedIdentity(t *testing.T) {
	h := newHarness(t, "", 0)
	s := t.TempDir()
	h.pool.register("k", &procEntry{scratch: s})

	best := ckptName(10, "0.00100")
	mkCkpt(t, s, "v0", best, `{"gen":"orig"}`)
	if nam, _, _, err := h.pool.ExportLive(context.Background(), "k"); err != nil || string(nam) != `{"gen":"orig"}` {
		t.Fatalf("first export nam=%q err=%v", nam, err)
	}

	// A worse epoch appears (does not become the best) and the best's sibling is
	// rewritten underneath us.
	mkCkpt(t, s, "v0", ckptName(40, "0.9"), `{"gen":"worse"}`)
	stem := strings.TrimSuffix(best, ".ckpt")
	if err := os.WriteFile(filepath.Join(s, "out", "v0", "checkpoints", stem+".nam"), []byte(`{"gen":"REWRITTEN"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	nam, ep, _, err := h.pool.ExportLive(context.Background(), "k")
	if err != nil {
		t.Fatalf("second export: %v", err)
	}
	if string(nam) != `{"gen":"orig"}` {
		t.Errorf("nam=%q, want the cached orig bytes (unchanged identity → no re-read)", nam)
	}
	if ep != 10 {
		t.Errorf("epoch=%d, want 10 (best unchanged)", ep)
	}
}

// A torn (invalid-json) sibling: with no prior good snapshot → ErrLiveTransient; with
// one → serve the cached last-good.
func TestExportLiveTornSiblingFallsBackElseTransient(t *testing.T) {
	h := newHarness(t, "", 0)
	s := t.TempDir()
	h.pool.register("k", &procEntry{scratch: s})
	mkCkpt(t, s, "v0", ckptName(10, "0.001"), "this is not json")
	if _, _, _, err := h.pool.ExportLive(context.Background(), "k"); !errors.Is(err, ErrLiveTransient) {
		t.Fatalf("torn/no-cache err=%v, want ErrLiveTransient", err)
	}

	h2 := newHarness(t, "", 0)
	s2 := t.TempDir()
	h2.pool.register("k", &procEntry{scratch: s2})
	mkCkpt(t, s2, "v0", ckptName(10, "0.010"), `{"gen":"good"}`)
	if _, _, _, err := h2.pool.ExportLive(context.Background(), "k"); err != nil {
		t.Fatalf("prime cache: %v", err)
	}
	mkCkpt(t, s2, "v0", ckptName(20, "0.001"), "torn-not-json") // better ESR, torn sibling
	nam, ep, _, err := h2.pool.ExportLive(context.Background(), "k")
	if err != nil {
		t.Fatalf("torn/with-cache err=%v, want last-good served", err)
	}
	if string(nam) != `{"gen":"good"}` || ep != 10 {
		t.Errorf("nam=%q epoch=%d, want cached last-good good/10", nam, ep)
	}
}

// A missing sibling (rotation ENOENT): same policy — last-good if any, else transient.
func TestExportLiveRotationENOENTFallsBackElseTransient(t *testing.T) {
	h := newHarness(t, "", 0)
	s := t.TempDir()
	h.pool.register("k", &procEntry{scratch: s})
	mkCkpt(t, s, "v0", ckptName(10, "0.001"), "") // ckpt present, .nam sibling missing
	if _, _, _, err := h.pool.ExportLive(context.Background(), "k"); !errors.Is(err, ErrLiveTransient) {
		t.Fatalf("enoent/no-cache err=%v, want ErrLiveTransient", err)
	}

	h2 := newHarness(t, "", 0)
	s2 := t.TempDir()
	h2.pool.register("k", &procEntry{scratch: s2})
	mkCkpt(t, s2, "v0", ckptName(10, "0.010"), `{"gen":"good"}`)
	if _, _, _, err := h2.pool.ExportLive(context.Background(), "k"); err != nil {
		t.Fatalf("prime cache: %v", err)
	}
	mkCkpt(t, s2, "v0", ckptName(20, "0.001"), "") // better ESR, sibling missing
	nam, ep, _, err := h2.pool.ExportLive(context.Background(), "k")
	if err != nil {
		t.Fatalf("enoent/with-cache err=%v, want last-good served", err)
	}
	if string(nam) != `{"gen":"good"}` || ep != 10 {
		t.Errorf("nam=%q epoch=%d, want cached last-good good/10", nam, ep)
	}
}

// F4 (structural): the cache is owned by the attempt entry, so a delete+resubmit that
// reuses the content key gets a FRESH entry and can never be served the previous
// attempt's bytes. Fill attempt 1's cache, unregister it, register a new attempt with
// no checkpoint yet — ExportLive must see only the new (empty) attempt.
func TestExportLiveCacheCannotLeakAcrossAttempts(t *testing.T) {
	h := newHarness(t, "", 0)

	s1 := t.TempDir()
	e1 := &procEntry{scratch: s1}
	h.pool.register("k", e1)
	mkCkpt(t, s1, "v0", ckptName(10, "0.001"), `{"gen":"attempt1"}`)
	if nam, _, _, err := h.pool.ExportLive(context.Background(), "k"); err != nil || string(nam) != `{"gen":"attempt1"}` {
		t.Fatalf("attempt1 export nam=%q err=%v", nam, err)
	}
	if e1.snap == nil {
		t.Fatal("attempt1 cache should be filled")
	}

	// Attempt 1 finishes (compare-and-delete unregister); a resubmit reuses the key.
	h.pool.unregister("k", e1)
	s2 := t.TempDir()
	if err := os.MkdirAll(filepath.Join(s2, "out"), 0o755); err != nil {
		t.Fatal(err)
	}
	e2 := &procEntry{scratch: s2}
	h.pool.register("k", e2)

	if _, _, _, err := h.pool.ExportLive(context.Background(), "k"); !errors.Is(err, ErrNoCheckpoint) {
		t.Fatalf("new attempt err=%v, want ErrNoCheckpoint (no stale cache leak)", err)
	}
	if e2.snap != nil {
		t.Error("new attempt's cache should still be empty")
	}
}

// F5: the driver's epoch_esr line is read from job_log in REVERSE, so a requeued
// attempt that appends epoch 3 again resolves to the LAST-appended value, and it
// overrides the coarser filename ESR.
func TestExportLiveESRFromLogReverseWins(t *testing.T) {
	h := newHarness(t, "", 0)
	h.seed(t, "k", jobs.KindTrain, 100)
	ctx := context.Background()
	for _, line := range []string{
		"Epoch 3/100",
		"DRIVER: epoch_esr=3=0.5", // first attempt's value
		"noise",
		"Epoch 30/100",
		"DRIVER: epoch_esr=30=0.9", // a DIFFERENT epoch: must not be picked for epoch 3
		"Epoch 3/100",
		"DRIVER: epoch_esr=3=0.2", // requeued attempt re-logged epoch 3 → LAST wins
	} {
		if _, err := h.store.DB().ExecContext(ctx,
			`INSERT INTO job_log(job_key, line) VALUES(?, ?)`, "k", line); err != nil {
			t.Fatal(err)
		}
	}
	s := t.TempDir()
	h.pool.register("k", &procEntry{scratch: s})
	mkCkpt(t, s, "v0", ckptName(3, "0.9"), `{"gen":"e3"}`) // filename ESR 0.9 must be overridden

	_, ep, esr, err := h.pool.ExportLive(ctx, "k")
	if err != nil {
		t.Fatalf("export: %v", err)
	}
	if ep != 3 {
		t.Errorf("epoch=%d, want 3", ep)
	}
	if !approx(esr, 0.2) {
		t.Errorf("esr=%v, want 0.2 (last epoch_esr=3= in log, not filename 0.9)", esr)
	}
}

// The immutable-snapshot / unlocked-read claim is load-bearing (crew F-F): hammer
// ExportLive from several readers while a writer churns attempts (register → grow
// checkpoints → unregister → RemoveAll), asserting every successful serve is
// internally coherent — the served .nam bytes name the same epoch the header
// reports. Run under -race, this pins both memory safety and cache coherence.
func TestExportLiveConcurrentReadersVsAttemptChurn(t *testing.T) {
	h := newHarness(t, "", 0)
	ctx := context.Background()

	done := make(chan struct{})
	var wg sync.WaitGroup
	for r := 0; r < 3; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-done:
					return
				default:
				}
				nam, epoch, _, err := h.pool.ExportLive(ctx, "k")
				if err != nil {
					continue // no attempt / no ckpt / transient — all legal mid-churn
				}
				var body struct {
					Epoch int64 `json:"epoch"`
				}
				if jerr := json.Unmarshal(nam, &body); jerr != nil {
					t.Errorf("served bytes are not the fabricated json: %v", jerr)
					return
				}
				if body.Epoch != epoch {
					t.Errorf("served bytes epoch=%d but header epoch=%d (incoherent snapshot)", body.Epoch, epoch)
					return
				}
			}
		}()
	}

	for round := 0; round < 25 && !t.Failed(); round++ {
		scratch := t.TempDir()
		e := &procEntry{scratch: scratch}
		h.pool.register("k", e)
		dir := filepath.Join(scratch, "out", "w", "checkpoints")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		for ep := 0; ep < 8; ep++ {
			name := ckptName(ep, fmt.Sprintf("0.%03d", 900-ep)) // each epoch is a new best
			stem := strings.TrimSuffix(name, ".ckpt")
			// nam first, then ckpt — the reader can never select a ckpt whose
			// sibling has not been fully written (mirrors a torn-guard-safe order;
			// the reverse order is exercised by the torn/ENOENT tests).
			if err := os.WriteFile(filepath.Join(dir, stem+".nam"),
				[]byte(fmt.Sprintf(`{"epoch":%d}`, ep)), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, name), []byte("CKPT"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		h.pool.unregister("k", e)
		if err := os.RemoveAll(scratch); err != nil {
			t.Fatal(err)
		}
	}
	close(done)
	wg.Wait()
}
