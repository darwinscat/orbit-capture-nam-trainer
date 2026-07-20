// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Command stubdriver is a TEST-ONLY stand-in for trainer_driver.py. It emits the
// exact stdout contract the worker parses (DRIVER lines, "Epoch k/n", the probe
// verdict lines) under a selectable MODE, so the supervision tests can drive
// real child processes — real process groups, real SIGKILL, real stalls — without
// python or torch.
//
// The worker spawns it exactly as it would the real driver:
//
//	<python> -u <driver> --input .. --output .. --outdir .. --name .. --epochs N --arch A
//
// so the tests set the "python" override to this binary and the "driver" override
// to the MODE string. This binary therefore reads argv as:
//
//	[-u] <mode> --input .. --outdir DIR --name NAME --epochs N ...
//
// It is not shipped with the daemon; only the test suite builds and runs it.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

func main() {
	args := os.Args[1:]
	if len(args) > 0 && args[0] == "-u" { // python's unbuffered flag; ignore
		args = args[1:]
	}
	// The behaviour MODE is taken from ONCT_STUB_MODE when set. This keeps the
	// "driver" argv slot a stable path (basename trainer_driver.py) so the worker's
	// restart-recovery argv guard matches identically for every mode. As a CLI
	// convenience (manual smoke tests) the first non-flag arg is used when the env
	// is unset.
	mode := os.Getenv("ONCT_STUB_MODE")
	if len(args) > 0 && args[0] != "" && args[0][:1] != "-" {
		if mode == "" {
			mode = args[0] // CLI form: mode is the driver slot
		}
		args = args[1:] // drop the driver slot either way
	}
	if mode == "" {
		fmt.Fprintln(os.Stderr, "stubdriver: missing mode (set ONCT_STUB_MODE)")
		os.Exit(2)
	}
	flags := parseFlags(args)
	outdir := flags["outdir"]
	name := flags["name"]
	if name == "" {
		name = "model"
	}
	epochs := atoiOr(flags["epochs"], 100)

	// "auto" lets one pool serve all lanes at once (for the concurrency test): pick
	// behaviour from the epoch count the worker passes per kind.
	if mode == "auto" {
		switch epochs {
		case 1:
			mode = "probe-self-pass"
		case 5:
			mode = "train-ok" // a short train that completes (live-cap resize tests)
		case 10:
			mode = "probe-e10-ok"
		default:
			mode = "train-hang" // a long-running train that occupies its lane
		}
	}

	switch mode {
	case "train-ok":
		banner(name, epochs)
		runEpochs(epochs, 60*time.Millisecond) // >50ms so the EWMA s/epoch is exercised
		fmt.Println("DRIVER: esr=0.00123456")
		writeModel(outdir, name)
		fmt.Printf("DRIVER: exported %s.nam\n", name)
		os.Exit(0)

	case "train-fail":
		banner(name, epochs)
		runEpochs(min(epochs, 2), 10*time.Millisecond)
		fmt.Fprintln(os.Stderr, "stubdriver: simulated training failure")
		os.Exit(1) // nonzero, no model written

	case "train-hang":
		// Prints a little, then goes silent forever — exercises the stall watchdog.
		banner(name, epochs)
		fmt.Println("Epoch 0/" + strconv.Itoa(epochs))
		sleepForever()

	case "silent-hang":
		// Never prints anything (the torch-import-silent analog) — stall watchdog.
		sleepForever()

	case "probe-self-pass":
		// V3 self-ESR check passes, then training starts: the first "Epoch " line
		// is the PASS verdict. Then hang forever so the worker must kill-on-verdict.
		// NAM prints the value with a trailing period — keep it, the parser must cope.
		fmt.Println("Replicate ESR is 0.0012.")
		fmt.Println("Epoch 0/1")
		sleepForever()

	case "probe-self-fail":
		// The check fails: a fail marker is the verdict. Hang so the worker kills it.
		fmt.Println("Replicate ESR is 0.4211.")
		fmt.Println("DRIVER: checkfail")
		sleepForever()

	case "probe-self-crash":
		// Dies with no verdict at all → the worker must report no_verdict, never fail.
		fmt.Println("some unrelated output")
		os.Exit(1)

	case "probe-e10-ok":
		banner(name, 10)
		runEpochs(10, 10*time.Millisecond)
		fmt.Println("DRIVER: esr=0.04710000")
		os.Exit(0)

	case "probe-e10-na":
		banner(name, 10)
		runEpochs(10, 5*time.Millisecond)
		fmt.Println("DRIVER: esr=na")
		os.Exit(0)

	default:
		fmt.Fprintf(os.Stderr, "stubdriver: unknown mode %q\n", mode)
		os.Exit(2)
	}
}

func banner(name string, epochs int) {
	fmt.Printf("DRIVER: training %s epochs=%d\n", name, epochs)
}

func runEpochs(n int, pace time.Duration) {
	for k := 0; k < n; k++ {
		fmt.Printf("Epoch %d/%d\n", k, n)
		time.Sleep(pace)
	}
}

// writeModel creates a non-trivial <outdir>/<name>.nam and <name>.train.json,
// mirroring what the real driver leaves behind on success.
func writeModel(outdir, name string) {
	if outdir == "" {
		return
	}
	_ = os.MkdirAll(outdir, 0o755)
	_ = os.WriteFile(filepath.Join(outdir, name+".nam"),
		[]byte(`{"version":"0.5.4","architecture":"stub","config":{},"weights":[0,0,0]}`), 0o644)
	_ = os.WriteFile(filepath.Join(outdir, name+".train.json"),
		[]byte(`{"esr":0.00123456,"epochs":1,"trainer":"stub"}`), 0o644)
}

func sleepForever() {
	for {
		time.Sleep(time.Hour) // killed by the worker (SIGKILL to the process group)
	}
}

func parseFlags(args []string) map[string]string {
	out := map[string]string{}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if len(a) > 2 && a[:2] == "--" && i+1 < len(args) {
			out[a[2:]] = args[i+1]
			i++
		}
	}
	return out
}

func atoiOr(s string, def int) int {
	if n, err := strconv.Atoi(s); err == nil {
		return n
	}
	return def
}
