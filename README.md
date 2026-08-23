# orbit-capture-nam-trainer

A single-binary daemon that trains [NAM](https://github.com/sdatkinson/neural-amp-modeler)
(Neural Amp Modeler) `.nam` models for the
[OrbitCapture NAM](https://github.com/darwinscat/orbit-capture-nam) desktop app. It is a **worker on
the app's shared PostgreSQL library**: the app queues a job row against a take it has recorded, the
daemon claims it, runs a self-provisioned python trainer on the take's audio, and writes progress, the
log, live snapshots and the finished `.nam` (plus the torch checkpoint a continuation resumes from)
back into the same database. There is no HTTP API and no private database: the schema
(`app/assets/migrations/0001_init.sql` in the app repository) is the whole contract.

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

On Linux the config lives under the service user's `~/.config/OrbitCaptureNamTrainer/`, and
training runs on **CPU** (no GPU needed — slower per epoch than Apple Silicon). The runtime
self-provisions under the user's home, so give it a roomy home volume; a small `/tmp` is fine
(pip's temp is redirected onto the home volume).

## Configuration

`config.toml` (macOS: `~/Library/Application Support/OrbitCaptureNamTrainer/config.toml`, mode
0600) is created with defaults on first start:

| key | |
| --- | --- |
| `dsn` | **required** — a libpq connection string for the shared library, e.g. `"host=studio.local port=5432 dbname=orbitnam user=orbitnam"` (add `password=…` if the role needs one). The environment variable `ORBITNAM_DSN` overrides it for one run. The daemon refuses to start without one. |
| `cap` | concurrent training jobs (1–8; default 1). Applied live; the app can ask for another value through `workers.train_cap_wanted`, and the menu bar has the same control — both write the new value back here. |
| `keep_awake` | hold the machine awake while the queue has work (macOS idle-sleep assertion; default on). |
| `data_dir` | where per-job scratch dirs live (the take's wav and the trainer's checkpoints while it runs). Default `<base>/data`. |

`ONCT_BASE_DIR` relocates the whole base directory (config, logs, runtime, data) — used by tests and
verification runs. A database whose `queue_contract` is newer than this build knows makes the
daemon report `ready = false` with a note in `workers.note`; it keeps heartbeating and never claims a
job until the library is one it understands. A database the app has not migrated **at all** is a
different thing: there is no `workers` table to write that note into, so the daemon cannot report
anything — it logs `waiting for the library` and retries every five seconds until the app opens it
(the app is what migrates). Installing the daemon before ever running the app is therefore harmless,
just quiet: the app's lamp will say "no trainer has reported" until you Connect once.

First run provisions its own python (python-build-standalone + a venv + `neural-amp-modeler`) and
fetches the capture signal, one time; `workers.ready` is false until it is up.

## How it works (the contract, in one page)

Everything goes through the app's tables; the daemon never touches the library's own tables
(devices, takes, models, …) beyond READING `take_audio` and `signals` of the take it is pointed at.

* **Heartbeat** — every 5 s the daemon UPSERTs its `workers` row (name = hostname, a fresh `instance`
  per start, version, nam/driver/signal provenance, gpu, python, `schema_version` = the
  `library.queue_contract` it was built for, `train_cap`/`probe_cap`, `running`, `paused`, `ready`,
  `avg_s_per_epoch` over its last 30 computed epochs, `disk_free_bytes`, `last_seen_at`). The app
  treats a worker as usable while `ready` and `last_seen_at` is within 15 s. A pending
  `train_cap_wanted` is applied live and cleared.
* **Claim** — per lane (`train` = train + train_more, `probe` = probe_self): the oldest queued row
  in drain order (`priority`, `queued_at`, `id`) that carries no stop/cancel request and pins the nam
  version this daemon runs, taken with `FOR UPDATE SKIP LOCKED` and stamped `worker`,
  `worker_instance`, a fresh `claim_token`, `claimed_at`/`started_at`. **Every later write is fenced
  on `(id, claim_token, state = 'running')`** — a straggler of an earlier attempt, or a row the app
  took away, simply affects 0 rows.
* **Materialize** — the take's wav from `take_audio` (sha-verified against the bytes the job pins,
  header-validated: 48 kHz, 30 s..20 min, ≤ 200 MB), the stimulus sha checked against the daemon's
  own (`signal_mismatch`), and for a `train_more` the app's `job_resume` snapshot as
  `--resume-from`; a missing snapshot fails `base_unavailable` — a continuation is never run from
  scratch. Then the trainer is spawned in its own process group and `pgid` recorded.
* **Progress** — `epoch` / `s_per_epoch` (≤ 1/s) and one `job_log` row per stdout line.
* **Commands on the row** (polled every 2 s):
  * `cancel_requested_at` → the group is killed, the row ends `cancelled` (error_code `cancelled`),
    nothing kept. Cancel beats stop.
  * `stop_requested_at` → acknowledged at once (`stop_seen_at`, `stop_state = pending`) and, as soon
    as a whole checkpoint pair exists, `armed` + killed; the last completed epoch's pair is
    harvested (a torn newest pair falls back to the previous one, then to the best-ESR pair) and the
    run finishes as a NORMAL `succeeded` row with `reached` = that epoch + 1, `stop_state = done` —
    what you keep is what you hear and where a continuation picks up. Nothing usable →
    `failed` / `stop_failed`, `stop_state = refused`. A probe is never stopped (`refused`, it runs to
    its verdict in seconds).
  * `live_requested_at` (newer than `live_served_at`) → the best-so-far checkpoint's `.nam` lands in
    `job_snapshots` (epoch, esr, sha, bytes) and `live_served_at` is stamped; with nothing to serve
    yet, `live_error = no_checkpoint` (or `transient` for a torn read — ask again).
* **Terminal write, one transaction** — `state`, `finished_at`, `reached` (passed explicitly from
  what the trainer was spawned with or what the stop harvested — never read back), `esr`,
  `verdict` (probe_self: `pass`/`fail`), `error_code` from the closed `job_error` list,
  `error_message`, the run's provenance (`nam_version`, `driver_sha256`, `signal_sha256`) and, for a
  success, the `job_result` row (`.nam` bytes + sha/size, `epochs` = reached, `esr`, the torch
  checkpoint when the run left one). A stall (no output for 15 minutes — torch import is silent for
  minutes, so this slack is deliberate) fails `stalled`; a `train_more` that dies before its first
  epoch line fails `resume_failed`, after it `train_failed`.
* **Restart recovery** — at start, the daemon requeues ONLY ITS OWN running rows (`worker` =
  hostname; claim released, progress cleared), argv-checks each recorded `pgid` before killing it,
  sweeps orphans by scratch path, and wipes scratch. A graceful shutdown or a menu-bar pause requeues
  in-flight jobs the same way — never `failed`. Stop/cancel flags stay on a requeued row (the claim
  filter skips them; the app resolves).

On macOS the daemon also puts a small status item in the menu bar: a waveform icon and, while the
queue has work, `2/20 13:36 5.14` — jobs **running / total in queue**, the clock-time **ETA** estimate
for the queue to drain at this machine's speed (`24h+` past a day), and the moving-average **seconds
per epoch** (the same number the heartbeat reports). The dropdown has **Pause now** (running jobs stop this
second and KEEP every epoch they finished — a `Continue` in the app resumes from there), **Pause
after current** (they run to their full epoch count), **Resume**, the head of the queue (take labels,
up to 12 rows), **Cap: N** and **Restart (re-read config)**. A pause is REMEMBERED: it is written to `<data_dir>/paused` and a
restart comes up paused, because a restart is what an upgrade, a config re-read and a crash all are —
and whoever paused this trainer wanted the machine, not a relaunch handing it back to the queue. Only
a hand lifts it: Resume here, or Resume from the app. While paused the heartbeat says so
(`workers.paused`). Set `ONCT_NO_TRAY` (any value) to disable
the tray; without a GUI session it is skipped automatically, and Linux never shows one.

## Tests

```sh
go test ./...                                   # the database-backed tests SKIP without a DSN
ORBITNAM_TEST_PG_DSN="host=… dbname=… user=…" go test ./...
```

Every database test creates a PRIVATE schema (`trainertest_<pid>_<n>`) in the named database,
applies the app's `0001_init.sql` inside it, seeds its own rows, and drops the schema at the end —
the public schema is never touched. The DDL is read from `ORBITNAM_TEST_DDL`, or by default from a
sibling checkout of `orbit-nam-capture` (`../orbit-nam-capture/app/assets/migrations/0001_init.sql`,
also one directory further up for a worktree layout). The supervision tests drive a stub trainer
(`cmd/stubdriver`, built by the test run) through real process groups.

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE).
