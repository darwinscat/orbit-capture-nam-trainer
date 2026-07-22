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
//	[-u] <mode> --input .. --outdir DIR --name NAME --epochs N [--resume-from CKPT]
//
// Successful train/probe_e10 modes leave a <outdir>/model.ckpt (its content is the
// total epoch count as text) exactly as the real driver leaves a resumable ckpt; the
// resume_* modes exercise the train_more path off it.
//
// It is not shipped with the daemon; only the test suite builds and runs it.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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
		writeCkpt(outdir, epochs) // every successful train leaves a resumable ckpt
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
		writeCkpt(outdir, 10) // a probe_e10 also leaves a ckpt — it can seed a train_more
		os.Exit(0)

	case "probe-e10-na":
		banner(name, 10)
		runEpochs(10, 5*time.Millisecond)
		fmt.Println("DRIVER: esr=na")
		writeCkpt(outdir, 10) // the run completed and exported its ckpt (esr=na is a metadata gap)
		os.Exit(0)

	case "resume_ok":
		// Continued training (train_more). Reads the parent's epoch count back from
		// the materialized ckpt to know where to start numbering, mirroring the real
		// driver's Lightning restore, then runs start..epochs-1 and leaves a fresh
		// nam + ckpt (the new total) so a chain can continue off this run.
		rf := flags["resume-from"]
		if rf == "" {
			fmt.Fprintln(os.Stderr, "stubdriver: resume_ok requires --resume-from")
			os.Exit(2)
		}
		start, ok := readCkptEpoch(rf)
		if !ok {
			fmt.Fprintf(os.Stderr, "stubdriver: resume_ok cannot read ckpt %q\n", rf)
			os.Exit(1)
		}
		banner(name, epochs)
		fmt.Printf("DRIVER: resuming from epoch %d\n", start)
		runEpochsFrom(start, epochs, 60*time.Millisecond)
		fmt.Println("DRIVER: esr=0.00098765")
		writeModel(outdir, name)
		writeCkpt(outdir, epochs)
		fmt.Printf("DRIVER: exported %s.nam\n", name)
		os.Exit(0)

	case "resume_badckpt":
		// The ckpt restore blows up inside Trainer.fit, BEFORE any Epoch line — the
		// signal the worker keys `resume_failed` on (no epoch >= start_epoch ever seen).
		// A python-style traceback to stderr mirrors the real failure surface.
		banner(name, epochs)
		printResumeTraceback(flags["resume-from"])
		os.Exit(1)

	case "probe_kill_after_esr":
		// The kill-after-ESR window: emit the verdict ESR, then hang WITHOUT ever
		// writing model.ckpt. The worker kills it (stall/shutdown) after the ESR is
		// banked, so the finished probe row must store NO ckpt — the crew F1 regression.
		banner(name, 10)
		runEpochs(10, 10*time.Millisecond)
		fmt.Println("DRIVER: esr=0.03100000")
		sleepForever()

	default:
		fmt.Fprintf(os.Stderr, "stubdriver: unknown mode %q\n", mode)
		os.Exit(2)
	}
}

func banner(name string, epochs int) {
	fmt.Printf("DRIVER: training %s epochs=%d\n", name, epochs)
}

func runEpochs(n int, pace time.Duration) { runEpochsFrom(0, n, pace) }

// runEpochsFrom prints "Epoch k/total" for k in [start, total) — the resume form
// keeps Lightning's absolute numbering, so a continued run's first epoch is start.
func runEpochsFrom(start, total int, pace time.Duration) {
	for k := start; k < total; k++ {
		fmt.Printf("Epoch %d/%d\n", k, total)
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

// writeCkpt writes <outdir>/model.ckpt ATOMICALLY (tmp file + rename), mirroring
// the real driver's os.replace export so a stall-kill mid-write can never leave a
// torn file. The stub's ckpt CONTENT is just the total epoch count as decimal text
// — resume_ok reads it back to know where to continue numbering.
func writeCkpt(outdir string, epochs int) {
	if outdir == "" {
		return
	}
	_ = os.MkdirAll(outdir, 0o755)
	tmp := filepath.Join(outdir, "model.ckpt.tmp")
	if err := os.WriteFile(tmp, []byte(strconv.Itoa(epochs)), 0o644); err != nil {
		return
	}
	_ = os.Rename(tmp, filepath.Join(outdir, "model.ckpt"))
}

// readCkptEpoch reads the decimal epoch count a train-success ckpt was written
// with (see writeCkpt). ok is false if the file is missing or not an integer.
func readCkptEpoch(path string) (int, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	n, err := strconv.Atoi(strings.TrimSpace(string(b)))
	if err != nil {
		return 0, false
	}
	return n, true
}

// printResumeTraceback writes a python-style traceback to stderr, imitating a
// checkpoint that fails to load inside Trainer.fit (a trainer-pin drift / corrupt
// ckpt). It never prints an Epoch line, so the worker keys this as resume_failed.
func printResumeTraceback(ckpt string) {
	fmt.Fprint(os.Stderr, "Traceback (most recent call last):\n"+
		"  File \"trainer_driver.py\", line 205, in <module>\n"+
		"    main()\n"+
		"  File \"trainer_driver.py\", line 190, in main\n"+
		"    result = train(\n"+
		"  File \"pytorch_lightning/trainer/trainer.py\", line 544, in fit\n"+
		"    self._run(model, ckpt_path="+strconv.Quote(ckpt)+")\n"+
		"RuntimeError: Error(s) in loading state_dict for the trainer: "+
		"unexpected key(s) in state_dict.\n")
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
