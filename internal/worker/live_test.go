// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
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

// (Every ExportLive test lived here and is gone with it: "let me hear this run as it stands" is a
// SELECT on the take's own row now, not a request the daemon answers by exporting a .nam on demand.
// What remains above — the checkpoint-name parsing and the newest/best selection — is what the saver
// uses to decide which pair to put in the library, and it is where those subtleties belong.)
