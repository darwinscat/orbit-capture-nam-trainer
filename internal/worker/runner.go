// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"syscall"
)

// Spec describes one trainer invocation. Signal is the standard NAM input wav
// (--input); Capture is the reamped take (--output); Outdir receives <name>.nam.
type Spec struct {
	Signal  string
	Capture string
	Outdir  string
	Name    string
	Epochs  int
	Arch    string
}

// Proc is a spawned trainer child: its merged stdout+stderr stream and the pgid
// of its process group (which the worker SIGKILLs as a unit).
type Proc struct {
	cmd  *exec.Cmd
	Out  io.ReadCloser
	Pgid int
}

// Wait reaps the child. It must be called exactly once, and only after the group
// has been (or is about to be) SIGKILLed — see the worker's kill/Wait ordering.
func (p *Proc) Wait() error { return p.cmd.Wait() }

// Close releases the read end of the output pipe. os/exec never closes a
// caller-supplied *os.File (it only closes pipes it creates itself), so the
// worker MUST close it once per spawned Proc — otherwise every job leaks a file
// descriptor until the finalizer runs at GC, and a long-lived daemon draining
// many jobs marches toward the fd rlimit.
func (p *Proc) Close() error {
	if p.Out != nil {
		return p.Out.Close()
	}
	return nil
}

// Runner spawns trainer children. Abstracted so tests inject the stub binary.
type Runner interface {
	Spawn(spec Spec) (*Proc, error)
	// DriverBase is the basename of the driver argv token, used by restart
	// recovery to argv-match a recorded pgid before killing it.
	DriverBase() string
}

// ProcessRunner runs the real (or stub) interpreter + driver as a child process
// in its own group, with stdout and stderr merged into one pipe.
type ProcessRunner struct {
	Python string   // interpreter (venv python3, or the test stub binary)
	Driver string   // driver path (trainer_driver.py) — a stable argv token
	Env    []string // extra environment appended to os.Environ() (test mode selector)
}

// DriverBase returns the driver basename for the recovery argv guard.
func (r ProcessRunner) DriverBase() string { return baseName(r.Driver) }

// Spawn starts the child in a fresh process group. stdout and stderr share one
// *os.File pipe end, so both are captured (the driver's tracebacks and the
// train() failure message go to stderr) and EOF arrives only when the WHOLE
// group — child and any torch grandchildren — has closed it.
func (r ProcessRunner) Spawn(spec Spec) (*Proc, error) {
	args := []string{
		"-u", r.Driver,
		"--input", spec.Signal,
		"--output", spec.Capture,
		"--outdir", spec.Outdir,
		"--name", spec.Name,
		"--epochs", strconv.Itoa(spec.Epochs),
		"--arch", spec.Arch,
	}
	cmd := exec.Command(r.Python, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} // new group; leader pid == pgid
	if len(r.Env) > 0 {
		cmd.Env = append(os.Environ(), r.Env...)
	}

	pr, pw, err := os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("pipe: %w", err)
	}
	// Both are the same *os.File, so exec passes the fd directly (no copy
	// goroutine) and wires child fds 1 and 2 to it.
	cmd.Stdout = pw
	cmd.Stderr = pw

	if err := cmd.Start(); err != nil {
		_ = pw.Close()
		_ = pr.Close()
		return nil, fmt.Errorf("start: %w", err)
	}
	_ = pw.Close() // parent's write end; the child holds its own → we see EOF at group exit

	return &Proc{cmd: cmd, Out: pr, Pgid: cmd.Process.Pid}, nil
}

// killGroup SIGKILLs an entire process group. A missing group (ESRCH) is fine.
func killGroup(pgid int) { _ = syscall.Kill(-pgid, syscall.SIGKILL) }

// baseName returns the last path element without importing path/filepath into
// every call site (kept tiny and dependency-light).
func baseName(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' {
			return p[i+1:]
		}
	}
	return p
}
