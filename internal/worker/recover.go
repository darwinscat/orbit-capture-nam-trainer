// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

import (
	"context"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// recover is the restart-recovery pass (the design notes), run before any
// worker starts. It requeues every job left running by a previous process and
// kills that process's children two ways:
//
//   - the recorded pgid of each running row, but only after argv-confirming it is
//     still OUR trainer (a bare pgid could have been recycled after a reboot into
//     an innocent process group — launchd RunAtLoad makes that the common case);
//   - a pkill sweep matching the scratch path, which catches a child that was
//     spawned but crashed before its pgid was recorded (that path is unique to
//     this daemon, so the sweep can never hit an unrelated process).
//
// Then it wipes all scratch dirs. Doing this fully before workers start means no
// freshly-claimed job's scratch is swept out from under it.
func (p *Pool) recover(ctx context.Context) error {
	pids, err := p.store.RecoverRunning(ctx)
	if err != nil {
		return err
	}
	if len(pids) > 0 {
		p.log.Printf("recovery: requeued %d running job(s) from a previous run", len(pids))
	}

	driverBase := ""
	if p.runner != nil {
		driverBase = p.runner.DriverBase()
	}
	for _, pid := range pids {
		guardedKillGroup(pid, driverBase)
	}
	sweepOrphans(p.scratchRoot)

	// Wipe and recreate the scratch root.
	if p.scratchRoot != "" {
		if err := os.RemoveAll(p.scratchRoot); err != nil {
			p.log.Printf("recovery: wipe scratch: %v", err)
		}
		if err := os.MkdirAll(p.scratchRoot, 0o755); err != nil {
			return err
		}
	}
	return nil
}

// guardedKillGroup SIGKILLs process group pgid only if pgid is still a group
// leader (pgid == its own pid) whose command line contains driverBase — i.e. it
// is still our trainer and not a recycled pid. A gone process (ps error) is a
// no-op. Uses /bin/ps, present on darwin and linux.
func guardedKillGroup(pgid int, driverBase string) {
	out, err := exec.Command("/bin/ps", "-p", strconv.Itoa(pgid), "-o", "pgid=,command=").Output()
	if err != nil {
		return // process gone
	}
	line := strings.TrimSpace(string(out))
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return
	}
	lead, err := strconv.Atoi(fields[0])
	if err != nil || lead != pgid {
		return // not a group leader with this pgid → recycled, leave it alone
	}
	if driverBase != "" && !strings.Contains(line, driverBase) {
		return // command isn't our trainer → recycled, leave it alone
	}
	killGroup(pgid)
}

// sweepOrphans SIGKILLs any process whose argv contains the scratch root — trainer
// children that were spawned but never recorded (crash between spawn and pid
// write). The scratch path is unique to this daemon, so this never hits an
// unrelated process. pkill exits 1 when nothing matches; that is fine.
func sweepOrphans(scratchRoot string) {
	if scratchRoot == "" {
		return
	}
	// pkill -f matches an extended regex; escape the path so a data_dir containing
	// regex metacharacters still matches literally instead of silently no-op'ing.
	pattern := regexp.QuoteMeta(scratchRoot) + "/"
	_ = exec.Command("/usr/bin/pkill", "-9", "-f", pattern).Run()
}
