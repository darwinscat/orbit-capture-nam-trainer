// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

//go:build darwin

package awake

import (
	"os"
	"os/exec"
	"strconv"
	"sync"
)

// platformSpawn holds a macOS power assertion via caffeinate for its own
// lifetime: -i no idle sleep, -m no disk idle sleep, -s no system sleep on AC.
// -w <pid> makes caffeinate exit on its own if this daemon dies without calling
// the release func, so a crash can never leak a stuck assertion. The absolute
// path avoids depending on launchd's PATH.
func platformSpawn() (func(), error) {
	cmd := exec.Command("/usr/bin/caffeinate",
		"-i", "-m", "-s", "-w", strconv.Itoa(os.Getpid()))
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		})
	}, nil
}
