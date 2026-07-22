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
`port` (8626), `bind` (127.0.0.1; set it to a Tailscale IP for remote access), `allow_api_cap`
(default false — clients may not resize the training lane until the admin allows it), `cap` (1–8
concurrent trains), `keep_awake` (hold the machine awake while the queue has work),
`retention_days` (default **0 = keep forever** — a finished job's `.nam` and its `train_more`
continuation checkpoint never expire; set a positive N to free them, and end continuation, N days
after a job finishes. This default reaches only a fresh config — an existing config.toml keeps its
written value (an older daemon wrote `7` or `90`), so hand-edit it to `0` for keep-forever),
`min_free_gb`, `data_dir`. Auto-start under launchd: `deploy/launchd/` (macOS) or
`deploy/systemd/` (Linux).

On macOS the daemon also puts a small status item in the menu bar: a waveform icon and, while the
queue has work, `2/20 13:36 5.14` — jobs **running / total in queue** (2 of 20 are running), the
clock-time **ETA** estimate for the queue to drain (`24h+` past a day), and the moving-average
**seconds per epoch** (the same number `/v1/health` reports). Idle shows just the icon. The dropdown menu has **Pause now** (the
running job is stopped and goes back in the queue), **Pause after current**, **Resume**, and the
head of the queue (up to 12 jobs, the rest collapse into a "… N more queued" line). While a pause
drains the current job the icon turns **orange** (Pause now stays available to cut it short); once
nothing is running it turns **red** and the keep-awake hold is released — a fully paused Mac may
sleep. Pause is in-memory: a daemon restart resumes. Below the queue sit **Cap: N** (pick 1–8
concurrent trains — 1 is the safe default, an Ultra-class GPU or a beefy CPU box can win with
more; applied immediately — raising starts idle workers, lowering lets running jobs finish — and
written to config.toml; the same control is `PATCH /v1/cap` on the API, allowed only when the
**Allow cap via API** toggle in this submenu — or `allow_api_cap` in config — says so) and
**Restart (re-read config)** — both restart gracefully, so a running job goes back in the queue
(its progress restarts). Under launchd the agent relaunches in seconds; run by hand, Restart just
stops the daemon. Set `ONCT_NO_TRAY` (any value) to disable the
tray; without a GUI session (SSH, pre-login) it is skipped automatically, and Linux never shows
one.

## API (v1)

Every request needs `Authorization: Bearer <token>`. Jobs are content-addressed — the key is a sha256
the client computes and the server re-verifies:

```
key = sha256hex(
  sha256hex(wav) + "\n" + "kind="   + <train|train_more|probe_self|probe_e10> + "\n" +
                         "epochs=" + <n>   + "\n" + "arch="   + <s>   + "\n" +
                         "nam="    + <v>   + "\n" + "driver=" + <sha> + "\n" +
                         "signal=" + <sha> + "\n"
                         [+ "base=" + <parent job key> + "\n"   -- train_more only] )
```

`nam` / `driver` / `signal` come from `GET /v1/health`.

| Method & path | |
| --- | --- |
| `GET /v1/health` | liveness + resolved profile |
| `PUT /v1/jobs/{key}?kind=…&epochs=N&arch=standard` | enqueue a capture (`audio/wav`) |
| `GET /v1/jobs/{key}` | poll raw progress (epoch, s_per_epoch, esr, verdict, …) |
| `PATCH /v1/jobs/{key}?priority=P` | reorder a queued job |
| `DELETE /v1/jobs/{key}` | free the key (kills a running trainer) |
| `POST /v1/jobs/{key}/stop` | stop a running train early, keeping the last completed epoch (`202` then poll; `409 no_checkpoint` before the first epoch) |
| `GET /v1/jobs/{key}/model` | download the `.nam` |
| `GET /v1/jobs/{key}/model?live=1` | audition a running job's best-so-far snapshot (`.nam`); `404 no_checkpoint` until there is one |
| `GET /v1/jobs/{key}/log` | training output |
| `POST /v1/queue` | batch: status + `position`/`epochs_ahead` for a list of the caller's keys |
| `PATCH /v1/cap?cap=N` | set the live training-lane width (1–8); admin-gated — 403 until `allow_api_cap` is on |

Kinds: `train` (produces a `.nam`), `train_more` (continue a finished job's training from its
stored checkpoint), `probe_self` (self-ESR verdict in seconds), `probe_e10` (10-epoch ESR probe).

**Listening to a running job.** While a `train`/`train_more`/`probe_e10` is still running,
`GET /v1/jobs/{key}/model?live=1` serves its best-so-far checkpoint as a playable `.nam` (with
`X-Live-Epoch` and `X-Live-Esr` headers) so you can audition the model mid-run. It is the same
best-checkpoint rule the trainer tracks; the finished run's deliverable additionally composes
per-submodel bests and may differ slightly — the live artifact is for listening, the terminal
model is the product. When no snapshot exists right now — before the first
completed epoch, a queued job, a `probe_self`, or the run's final teardown seconds — it answers
`404 no_checkpoint`: just poll again. An old daemon that doesn't know `live=1` (or a poll landing
in the moment between claim and attempt registration) answers `404 not_found`; re-detect per poll
rather than latching off one response. The snapshot is ephemeral — never stored, never touches `has_model`
or the plain `/model` download.

**Continued training.** Every successful `train`/`train_more`/`probe_e10` also keeps its last
training checkpoint (server-side only, never downloadable) for `retention_days` (forever by
default). To push the same capture further — say 200 epochs weren't enough — re-upload the SAME
wav as
`PUT /v1/jobs/{key}?kind=train_more&base=<the finished job's key>&epochs=400`: training resumes at
epoch 200 (optimizer state and learning-rate schedule intact) and only computes the difference.
`epochs` is the new TOTAL and must exceed the parent's `reached` epoch count (equal to its
`epochs` for a run that finished on its own — a run stopped at 250/400 continues from 250, so
300 is legal); chains (200→400→600) and continuing a
`probe_e10`'s 10 epochs into a full train both work the same way. If the parent can't seed a
continuation (deleted, failed, checkpoint expired, different wav) the daemon answers
`409 base_unavailable` — retrain from scratch with a normal `kind=train`.

**Stopping a training early.** `POST /v1/jobs/{key}/stop` on a running `train`/`train_more` pauses
it: the daemon stops the trainer at its last completed epoch and finishes the job as a NORMAL
`succeeded` run whose downloadable `.nam` — and retained checkpoint — are exactly that epoch's
state. What you keep is what you hear, and exactly where a continuation picks up: to go further,
`PUT` a `kind=train_more&base=<this key>&epochs=<higher>` and it resumes at the epoch it stopped on
(the job's `reached` count — stop a 400-epoch run at 250, continue to 300). The verb answers
`202 {"state":"stopping"}` and the terminal state lands a moment later, so keep polling `GET` until
it reads `succeeded`. Before the first completed epoch (nothing to keep yet), on a queued job, or on
a probe it answers `409 no_checkpoint`; the same `409` can appear in the run's final teardown seconds
while it is already about to succeed on its own, so re-`GET` the job before falling back to `DELETE`.
A terminal job is `204` (idempotent — also the answer when a stop raced the finish). Honest corner:
a daemon restart racing a stop FORGETS it — the job requeues and trains to its full target, so
re-`POST /stop` if you still want to pause it. An old daemon has no such route and answers a plain
`404` (fall back to `DELETE`).

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE).
