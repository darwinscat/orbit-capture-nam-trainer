// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Command keytool computes the content-addressed job key for a capture wav,
// resolving the runtime profile (nam version + driver/signal sha) from a running
// daemon's /v1/health. It is a convenience for manual testing and verification —
// the desktop app computes the same key from the same formula (the design notes).
//
//	keytool -token <tok> -wav take.wav -kind train -epochs 100
//	keytool -token <tok> -wav take.wav -kind train_more -epochs 200 -base <parent key>
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"

	"orbit-capture-nam-trainer/internal/jobkey"
	"orbit-capture-nam-trainer/internal/jobs"
)

func main() {
	url := flag.String("url", "http://127.0.0.1:8626", "daemon base URL")
	token := flag.String("token", "", "bearer token")
	wavPath := flag.String("wav", "", "path to the capture wav")
	kind := flag.String("kind", "train", "train | train_more | probe_self | probe_e10")
	epochs := flag.Int("epochs", 100, "epochs (train/train_more only; probes are fixed)")
	arch := flag.String("arch", "standard", "arch")
	base := flag.String("base", "", "parent job key (train_more only; 64-hex)")
	flag.Parse()

	if *token == "" || *wavPath == "" {
		fmt.Fprintln(os.Stderr, "keytool: -token and -wav are required")
		os.Exit(2)
	}
	if !jobs.ValidKind(*kind) {
		fmt.Fprintf(os.Stderr, "keytool: invalid kind %q\n", *kind)
		os.Exit(2)
	}
	if *kind == jobs.KindTrainMore {
		if !isHex64(*base) {
			fmt.Fprintln(os.Stderr, "keytool: -base (64-hex parent key) is required for kind=train_more")
			os.Exit(2)
		}
	} else if *base != "" {
		fmt.Fprintln(os.Stderr, "keytool: -base is only valid for kind=train_more")
		os.Exit(2)
	}

	prof, err := fetchProfile(*url, *token)
	if err != nil {
		fmt.Fprintln(os.Stderr, "keytool:", err)
		os.Exit(1)
	}
	if !prof.Ready {
		fmt.Fprintln(os.Stderr, "keytool: daemon not ready (still provisioning)")
		os.Exit(1)
	}

	wav, err := os.ReadFile(*wavPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "keytool:", err)
		os.Exit(1)
	}

	normEpochs := jobs.NormalizeEpochs(*kind, *epochs)
	var key string
	if *kind == jobs.KindTrainMore {
		key = jobkey.ComputeTrainMore(jobkey.SHA256Hex(wav), normEpochs, *arch,
			prof.Nam, prof.DriverSHA256, prof.SignalSHA256, *base)
	} else {
		key = jobkey.Compute(jobkey.SHA256Hex(wav), *kind, normEpochs, *arch,
			prof.Nam, prof.DriverSHA256, prof.SignalSHA256)
	}
	fmt.Println(key)
}

// isHex64 reports whether s is exactly 64 lower-case hex characters — the shape of a
// job key, matching the daemon's own base-param check.
func isHex64(s string) bool {
	if len(s) != 64 {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

type profile struct {
	Ready        bool   `json:"ready"`
	Nam          string `json:"nam"`
	DriverSHA256 string `json:"driver_sha256"`
	SignalSHA256 string `json:"signal_sha256"`
}

func fetchProfile(baseURL, token string) (profile, error) {
	req, err := http.NewRequest(http.MethodGet, baseURL+"/v1/health", nil)
	if err != nil {
		return profile{}, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return profile{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return profile{}, fmt.Errorf("health %d: %s", resp.StatusCode, body)
	}
	var p profile
	if err := json.Unmarshal(body, &p); err != nil {
		return profile{}, err
	}
	return p, nil
}
