// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package runtime

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	stdruntime "runtime"
	"strings"
	"syscall"
	"time"
)

// Profile is the resolved trainer profile. Nam is the RESOLVED version from the
// venv (`import nam`), never the configured pin — a half-built venv must be
// visible as such rather than reported healthy.
type Profile struct {
	Python       string
	Nam          string
	GPU          string // training device: "mps" | "cuda" | "cpu"
	DriverSHA256 string
	SignalSHA256 string
}

// Provision brings the managed runtime up to the pinned state and returns the
// resolved profile. It is idempotent: on a warm runtime it only verifies + writes
// the driver + resolves versions (~a second), so it is safe to run at every start.
// An flock serializes two daemons that might race the same runtime dir.
func Provision(ctx context.Context, runtimeDir string, onStatus func(string)) (Profile, error) {
	if onStatus == nil {
		onStatus = func(string) {}
	}
	if err := os.MkdirAll(runtimeDir, 0o755); err != nil {
		return Profile{}, fmt.Errorf("create runtime dir: %w", err)
	}

	unlock, err := flockExclusive(ctx, filepath.Join(runtimeDir, ".provision.lock"), onStatus)
	if err != nil {
		return Profile{}, err
	}
	defer unlock()

	pythonBin := filepath.Join(runtimeDir, "python", "bin", "python3.12")
	// Gate on a LIVE interpreter, not mere existence: tar extraction is not atomic,
	// so an interrupted unpack can leave python3.12 present but the stdlib tree
	// incomplete. Without the liveness probe that broken tree would be trusted
	// forever and every venv creation would fail — a permanent provisioning wedge.
	if !fileExists(pythonBin) || runQuiet(ctx, pythonBin, "-c", "import sys") != nil {
		_ = os.RemoveAll(filepath.Join(runtimeDir, "python")) // drop any partial tree
		pyArchiveName, pyURL, pySHA, err := pythonPin()
		if err != nil {
			return Profile{}, err
		}
		archive := filepath.Join(runtimeDir, pyArchiveName)
		if !fileExists(archive) {
			onStatus("downloading python runtime (~25 MB, one time)")
			if err := downloadVerify(ctx, pyURL, archive, pySHA, onStatus); err != nil {
				return Profile{}, fmt.Errorf("download python: %w", err)
			}
		}
		onStatus("unpacking python runtime")
		if err := runQuiet(ctx, "/usr/bin/tar", "-xzf", archive, "-C", runtimeDir); err != nil {
			return Profile{}, fmt.Errorf("unpack python: %w", err)
		}
		if !fileExists(pythonBin) || runQuiet(ctx, pythonBin, "-c", "import sys") != nil {
			return Profile{}, errors.New("python interpreter unusable after unpack")
		}
	}

	venvPy := filepath.Join(runtimeDir, "venv", "bin", "python3")
	marker := filepath.Join(runtimeDir, "venv-nam-"+NamPin+".ok")
	// Gate on the marker AND a live `import nam`: the marker is written only after
	// pip succeeds, and the import catches a venv that was half-built or removed.
	venvOK := fileExists(marker) && runQuiet(ctx, venvPy, "-c", "import nam") == nil
	if !venvOK {
		_ = os.Remove(marker)
		onStatus("creating python environment")
		if err := runQuiet(ctx, pythonBin, "-m", "venv", "--clear", filepath.Join(runtimeDir, "venv")); err != nil {
			return Profile{}, fmt.Errorf("create venv: %w", err)
		}
		onStatus("installing the trainer (one time, this is the big one)")
		pipLog := filepath.Join(runtimeDir, "pip.log")
		// Keep pip's build/unpack temp off a possibly-tiny /tmp (a small tmpfs is a
		// common Linux default) by pointing TMPDIR at the roomy runtime volume.
		pipTmp := filepath.Join(runtimeDir, "piptmp")
		if err := os.MkdirAll(pipTmp, 0o755); err != nil {
			return Profile{}, fmt.Errorf("create pip tmp: %w", err)
		}
		pipEnv := []string{"TMPDIR=" + pipTmp}
		// On Linux the default PyPI torch is the CUDA build — ~2.5 GB of NVIDIA
		// wheels a CPU box never uses (and enough to overflow a small /tmp). Install
		// the CPU build from the pytorch index first; nam then resolves against the
		// already-satisfied torch. macOS PyPI torch is the right (MPS) build, so it
		// needs no extra step.
		if stdruntime.GOOS == "linux" {
			if err := runToLog(ctx, pipLog, pipEnv, venvPy, "-m", "pip", "install", "--no-input",
				"torch", "--index-url", "https://download.pytorch.org/whl/cpu"); err != nil {
				return Profile{}, fmt.Errorf("pip install torch (see runtime/pip.log): %w", err)
			}
		}
		if err := runToLog(ctx, pipLog, pipEnv, venvPy, "-m", "pip", "install", "--no-input",
			"neural-amp-modeler=="+NamPin); err != nil {
			return Profile{}, fmt.Errorf("pip install (see runtime/pip.log): %w", err)
		}
		if err := os.WriteFile(marker, []byte(nowStamp()), 0o644); err != nil {
			return Profile{}, fmt.Errorf("write venv marker: %w", err)
		}
	}

	// Vendor the driver every start (this service owns it; refresh on upgrade).
	driverPath := filepath.Join(runtimeDir, "trainer_driver.py")
	if err := os.WriteFile(driverPath, DriverBytes(), 0o644); err != nil {
		return Profile{}, fmt.Errorf("write driver: %w", err)
	}

	// Fetch the capture signal (never shipped; downloaded + sha-verified). Under
	// the same flock as the rest of provisioning.
	if err := EnsureSignal(ctx, runtimeDir, onStatus); err != nil {
		return Profile{}, err
	}

	// Resolve python + nam versions and the training device in one call. nam pulls
	// torch in anyway, so the mps/cuda check rides along for free.
	info, err := captureLine(ctx, venvPy, "-c",
		"import platform, nam, torch;"+
			"print(platform.python_version());"+
			"print(nam.__version__);"+
			"print('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))")
	if err != nil {
		return Profile{}, fmt.Errorf("resolve trainer profile: %w", err)
	}
	f := strings.Split(strings.TrimSpace(info), "\n")
	if len(f) < 3 {
		return Profile{}, fmt.Errorf("resolve trainer profile: unexpected output %q", info)
	}
	// Our three prints are always the last lines; take them from the end so any
	// import-time stdout noise (a deprecation notice, a first-run banner) prepends
	// harmlessly instead of shifting the fields.
	f = f[len(f)-3:]

	return Profile{
		Python:       strings.TrimSpace(f[0]),
		Nam:          strings.TrimSpace(f[1]),
		GPU:          strings.TrimSpace(f[2]),
		DriverSHA256: DriverSHA256(),
		SignalSHA256: SignalSHA256,
	}, nil
}

// EnsureSignal makes the capture signal present at SignalPath(runtimeDir),
// downloading it from the official source (sha-verified) if absent or corrupt. The
// signal carries no redistribution license, so it is fetched, never shipped.
func EnsureSignal(ctx context.Context, runtimeDir string, onStatus func(string)) error {
	if onStatus == nil {
		onStatus = func(string) {}
	}
	return ensureSignalFrom(ctx, SignalPath(runtimeDir), SignalURL, SignalSHA256, onStatus)
}

func ensureSignalFrom(ctx context.Context, path, url, wantSHA string, onStatus func(string)) error {
	if fileHasSHA(path, wantSHA) {
		return nil // already fetched and valid
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	onStatus("downloading capture signal (" + SignalName + ", ~27 MB, one time)")
	if err := downloadVerify(ctx, url, path, wantSHA, onStatus); err != nil {
		return fmt.Errorf("fetch capture signal failed — download it manually from %s and drop it at %s: %w",
			SignalPageURL, path, err)
	}
	return nil
}

// fileHasSHA reports whether the file at path exists and its sha256 matches wantSHA.
func fileHasSHA(path, wantSHA string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false
	}
	return strings.EqualFold(hex.EncodeToString(h.Sum(nil)), wantSHA)
}

// VenvPython / DriverPath / SignalPath give the canonical on-disk locations under
// a runtime dir, shared by main's runner wiring and by Provision.
func VenvPython(runtimeDir string) string { return filepath.Join(runtimeDir, "venv", "bin", "python3") }
func DriverPath(runtimeDir string) string { return filepath.Join(runtimeDir, "trainer_driver.py") }
func SignalPath(runtimeDir string) string { return filepath.Join(runtimeDir, SignalName) }

// ---- helpers ---------------------------------------------------------------

// flockExclusive takes an exclusive advisory lock on path, polling (so ctx
// cancellation is honored) and announcing once if another instance holds it.
func flockExclusive(ctx context.Context, path string, onStatus func(string)) (func(), error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open lock: %w", err)
	}
	announced := false
	for {
		err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			return func() {
				_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
				_ = f.Close()
			}, nil
		}
		if !errors.Is(err, syscall.EWOULDBLOCK) {
			_ = f.Close()
			return nil, fmt.Errorf("flock: %w", err)
		}
		if !announced {
			onStatus("another instance is preparing the runtime — waiting")
			announced = true
		}
		select {
		case <-ctx.Done():
			_ = f.Close()
			return nil, ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
}

// downloadStall is how long the download may make zero progress before it is
// abandoned. http.DefaultClient has no whole-body timeout, so without this a
// connection that stalls mid-body (or never sends the body after a 200) would
// block forever and wedge provisioning; a stall watchdog lets the capped-backoff
// retry recover. It resets on every read, so a slow-but-steady download is fine.
var downloadStall = 90 * time.Second // var (not const) so tests can shorten it

// downloadVerify streams url to dest (via a .part temp), hashing as it goes, and
// only installs it if the sha256 matches — the transport is untrusted-but-verified.
func downloadVerify(ctx context.Context, url, dest, wantSHA string, onStatus func(string)) error {
	// A per-download context a stall watchdog can cancel; armed before Do so a
	// hung header phase is caught too, then reset on each body read.
	dlCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	watchdog := time.AfterFunc(downloadStall, cancel)
	defer watchdog.Stop()

	stalled := func(err error) error {
		if dlCtx.Err() != nil && ctx.Err() == nil {
			return fmt.Errorf("download stalled (no data for %s)", downloadStall)
		}
		return err
	}

	req, err := http.NewRequestWithContext(dlCtx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return stalled(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("http status %d", resp.StatusCode)
	}

	tmp := dest + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	h := sha256.New()
	total := resp.ContentLength
	var done int64
	buf := make([]byte, 256<<10)
	lastReport := time.Time{}
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			watchdog.Reset(downloadStall)
			if _, werr := f.Write(buf[:n]); werr != nil {
				f.Close()
				_ = os.Remove(tmp)
				return werr
			}
			h.Write(buf[:n])
			done += int64(n)
			if total > 0 && time.Since(lastReport) > 1500*time.Millisecond {
				lastReport = time.Now()
				onStatus(fmt.Sprintf("downloading python %d%%", done*100/total))
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			f.Close()
			_ = os.Remove(tmp)
			return stalled(rerr)
		}
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); !strings.EqualFold(got, wantSHA) {
		_ = os.Remove(tmp)
		return fmt.Errorf("sha256 mismatch: got %s want %s", got, wantSHA)
	}
	return os.Rename(tmp, dest)
}

func runQuiet(ctx context.Context, name string, args ...string) error {
	return exec.CommandContext(ctx, name, args...).Run() // nil Stdout/Stderr → /dev/null
}

func runToLog(ctx context.Context, logPath string, env []string, name string, args ...string) error {
	f, err := os.Create(logPath)
	if err != nil {
		return err
	}
	defer f.Close()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdout = f
	cmd.Stderr = f
	if len(env) > 0 {
		cmd.Env = append(os.Environ(), env...)
	}
	return cmd.Run()
}

func captureLine(ctx context.Context, name string, args ...string) (string, error) {
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func fileExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

// nowStamp avoids Date.now-style nondeterminism concerns by being the only clock
// read here; a plain marker timestamp is fine.
func nowStamp() string { return time.Now().UTC().Format(time.RFC3339) }
