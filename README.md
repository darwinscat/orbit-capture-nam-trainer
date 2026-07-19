# orbit-capture-nam-trainer

A single-binary daemon that trains [NAM](https://github.com/sdatkinson/neural-amp-modeler)
(Neural Amp Modeler) `.nam` models. It takes a reamped capture over HTTP, runs a self-provisioned
python trainer, and serves back the `.nam` — so the
[OrbitCapture NAM](https://github.com/darwinscat/orbit-capture-nam) desktop app (or any client) can
train captures without shipping python itself.

By Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. macOS (Apple Silicon / MPS) and Linux
(x86_64 / arm64, CPU).

## Build & run

```sh
go build -o namtrainerd ./cmd/namtrainerd
./namtrainerd
```

Easiest on macOS: download the signed + notarized **`.pkg`** from the
[Releases](https://github.com/darwinscat/orbit-capture-nam-trainer/releases) page and double-click —
it installs `namtrainerd` to `/usr/local/bin` and starts it as a per-user LaunchAgent (the daemon
must run in your login session, since NAM trains on your GPU). Or grab the bare signed binary from
the same page and run it directly.

**Linux** (x86_64 / arm64, manual): download `namtrainerd-<version>-linux-<amd64|arm64>.tar.gz` from
Releases (check its `.sha256`), and install as a systemd service:

```sh
tar -xzf namtrainerd-*-linux-*.tar.gz
sudo install -m0755 namtrainerd /usr/local/bin/namtrainerd
sudo curl -fsSL https://raw.githubusercontent.com/darwinscat/orbit-capture-nam-trainer/main/deploy/systemd/namtrainerd.service \
  -o /etc/systemd/system/namtrainerd.service
sudo sed -i "s/^User=CHANGEME/User=$USER/" /etc/systemd/system/namtrainerd.service
sudo systemctl daemon-reload && sudo systemctl enable --now namtrainerd
```

On Linux, config + token live under the service user's `~/.config/OrbitCaptureNamTrainer/`, and
training runs on **CPU** (no GPU needed — slower per epoch than Apple Silicon). The runtime
self-provisions under the user's home, so give it a roomy home volume; a small `/tmp` is fine
(pip's temp is redirected onto the home volume).

First run provisions its own python (python-build-standalone + a venv + `neural-amp-modeler`) and
fetches the capture signal, one time. `GET /v1/health` reports `ready:false` until it is up. Config
and the bearer token live in `~/Library/Application Support/OrbitCaptureNamTrainer/config.toml` —
`port` (8626), `bind` (127.0.0.1; set it to a Tailscale IP for remote access), `cap` (1–4
concurrent trains), `keep_awake` (hold the machine awake while the queue has work),
`retention_days`, `min_free_gb`, `data_dir`. Auto-start under launchd: `deploy/launchd/` (macOS) or
`deploy/systemd/` (Linux).

On macOS the daemon also puts a small status item in the menu bar: a waveform icon and, while the
queue has work, `2/20 13:36 5.14` — jobs **running/queued**, the clock-time **ETA** estimate for
the queue to drain (`24h+` past a day), and the moving-average **seconds per epoch** (the same
number `/v1/health` reports). Idle shows just the icon. The dropdown menu has **Pause now** (the
running job is stopped and goes back in the queue), **Pause after current**, **Resume**, and the
head of the queue (up to 12 jobs, the rest collapse into a "… N more queued" line). While a pause
drains the current job the icon turns **orange** (Pause now stays available to cut it short); once
nothing is running it turns **red** and the keep-awake hold is released — a fully paused Mac may
sleep. Pause is in-memory: a daemon restart resumes. Set `ONCT_NO_TRAY` (any value) to disable the
tray; without a GUI session (SSH, pre-login) it is skipped automatically, and Linux never shows
one.

## API (v1)

Every request needs `Authorization: Bearer <token>`. Jobs are content-addressed — the key is a sha256
the client computes and the server re-verifies:

```
key = sha256hex(
  sha256hex(wav) + "\n" + "kind="   + <train|probe_self|probe_e10> + "\n" +
                         "epochs=" + <n>   + "\n" + "arch="   + <s>   + "\n" +
                         "nam="    + <v>   + "\n" + "driver=" + <sha> + "\n" +
                         "signal=" + <sha> + "\n" )
```

`nam` / `driver` / `signal` come from `GET /v1/health`.

| Method & path | |
| --- | --- |
| `GET /v1/health` | liveness + resolved profile |
| `PUT /v1/jobs/{key}?kind=…&epochs=N&arch=standard` | enqueue a capture (`audio/wav`) |
| `GET /v1/jobs/{key}` | poll raw progress (epoch, s_per_epoch, esr, verdict, …) |
| `PATCH /v1/jobs/{key}?priority=P` | reorder a queued job |
| `DELETE /v1/jobs/{key}` | free the key (kills a running trainer) |
| `GET /v1/jobs/{key}/model` | download the `.nam` |
| `GET /v1/jobs/{key}/log` | training output |
| `POST /v1/queue` | batch: status + `position`/`epochs_ahead` for a list of the caller's keys |

Kinds: `train` (produces a `.nam`), `probe_self` (self-ESR verdict in seconds), `probe_e10`
(10-epoch ESR probe).

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE).
