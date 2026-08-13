// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"context"
	"slices"
	"testing"
	"time"

	"orbit-capture-nam-trainer/internal/jobs"
)

// baseSpec is one plain train invocation — no resume, no latency.
func baseSpec() Spec {
	return Spec{
		Signal: "/sig/v3_0_0.wav", Capture: "/scratch/capture.wav",
		Outdir: "/scratch/out", Name: "model", Epochs: 100, Arch: "standard",
	}
}

// TestDriverArgsUnchangedWithoutOptions pins the historical argv: a job with
// neither a resume ckpt nor a latency must spawn exactly what it always did.
func TestDriverArgsUnchangedWithoutOptions(t *testing.T) {
	want := []string{
		"-u", "/drv/trainer_driver.py",
		"--input", "/sig/v3_0_0.wav",
		"--output", "/scratch/capture.wav",
		"--outdir", "/scratch/out",
		"--name", "model",
		"--epochs", "100",
		"--arch", "standard",
	}
	if got := driverArgs("/drv/trainer_driver.py", baseSpec()); !slices.Equal(got, want) {
		t.Errorf("driverArgs = %q,\nwant %q", got, want)
	}
}

// TestDriverArgsAppendsLatency: a supplied latency rides through as --latency, and
// zero is passed as zero (it means "trim nothing", not "no value").
func TestDriverArgsAppendsLatency(t *testing.T) {
	for _, tc := range []struct {
		latency int64
		want    string
	}{{1101, "1101"}, {0, "0"}} {
		spec := baseSpec()
		spec.Latency = &tc.latency
		got := driverArgs("/drv/trainer_driver.py", spec)
		if n := len(got); n < 2 || got[n-2] != "--latency" || got[n-1] != tc.want {
			t.Errorf("latency=%d: argv tail = %q, want --latency %s", tc.latency, got, tc.want)
		}
	}
}

// TestDriverArgsOrdersResumeThenLatency: a train_more that also carries a latency
// gets both flags, resume first — the order the driver's argparse and the manual
// re-run instructions in the story log both assume.
func TestDriverArgsOrdersResumeThenLatency(t *testing.T) {
	spec := baseSpec()
	spec.ResumeCkpt = "/scratch/parent.ckpt"
	latency := int64(1101)
	spec.Latency = &latency

	got := driverArgs("/drv/trainer_driver.py", spec)
	want := append(driverArgs("/drv/trainer_driver.py", baseSpec()),
		"--resume-from", "/scratch/parent.ckpt", "--latency", "1101")
	if !slices.Equal(got, want) {
		t.Errorf("driverArgs = %q,\nwant %q", got, want)
	}
}

// TestPoolCarriesRowLatencyIntoTheSpec closes the last link: the value stored on the
// job row is what the pool hands the runner. A row without one hands nil, so the
// child spawns with no --latency at all.
func TestPoolCarriesRowLatencyIntoTheSpec(t *testing.T) {
	for _, tc := range []struct {
		name    string
		latency *int64
	}{
		{"with latency", func() *int64 { n := int64(1101); return &n }()},
		{"without latency", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, "train-ok", time.Minute)
			cr := &capturingRunner{inner: h.pool.runner}
			h.pool.runner = cr

			if err := h.store.InsertJob(context.Background(), jobs.Job{
				Key: "k", Kind: jobs.KindTrain, State: jobs.StateQueued,
				Priority: 1, Epochs: 5, Arch: "standard", CreatedAt: 1, Latency: tc.latency,
			}, []byte("capture-bytes")); err != nil {
				t.Fatalf("seed: %v", err)
			}
			h.start(t)
			h.waitState(t, "k", jobs.StateSucceeded, 15*time.Second)

			spec, spawns := cr.spec()
			if spawns != 1 {
				t.Fatalf("spawns = %d, want 1", spawns)
			}
			if !jobs.SameLatency(spec.Latency, tc.latency) {
				t.Errorf("Spec.Latency = %v, want %v", spec.Latency, tc.latency)
			}
		})
	}
}
