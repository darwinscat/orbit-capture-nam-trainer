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

### A second trainer on another Mac

Nothing but this daemon goes on that machine — no app, no database, no python. Two trainers on one
library halve the wall clock of a shoot, and a machine that is asleep or busy simply does not claim.

1. **Install.** Take `namtrainerd-<version>-macos-arm64.pkg` from
   [Releases](https://github.com/darwinscat/orbit-capture-nam-trainer/releases) and open it. It is
   signed and notarized, so Gatekeeper lets it through without ceremony. The installer puts the binary
   in `/usr/local/bin` and loads a LaunchAgent, so it starts at login and stays up.
2. **Point it at the library.** Click the waveform in the menu bar → **Setup…** and fill in host, port,
   database, user, password and **schema** (`public` is the real library; anything else is a scratch
   one). Save & restart. That writes `~/Library/Application Support/OrbitCaptureNamTrainer/config.toml`
   — the same file you could edit by hand, if you would rather.
3. **Let it through the local network.** The library is on another machine, and macOS gates that per
   application. Expect the first minute to look broken — the agent starts, cannot reach the database,
   exits, and `KeepAlive` restarts it until the system lets it through. If it never does: System
   Settings → Privacy & Security → **Local Network**, and switch `namtrainerd` on.
4. **Wait for the first run to provision.** Python, torch and the NAM trainer are downloaded and
   sha-verified into `~/Library/Application Support/OrbitCaptureNamTrainer/runtime` — several minutes
   and a few gigabytes, once. The menu bar says `ready for work` when it is over.

**The library must already exist.** The app owns the schema and migrates it; this daemon only reads
what it finds, and refuses a library whose queue contract it does not know. Open the library with the
app once, from any machine, before pointing a trainer at it — a trainer aimed at an empty or older
database shows red and says why on the line above the menu.

**One name per machine.** The worker name is the hostname unless `ONCT_WORKER_NAME` says otherwise,
and the database refuses a second daemon under a name already held — so cloning a configured machine
means giving the clone its own name.

### macOS: the library on another machine needs Local Network permission

A daemon whose library is not on this Mac talks to Postgres over the local network, and macOS gates
that per application — including background ones, which cannot show a prompt. **Install a signed
build.** An ad-hoc binary (`go build` alone) has no stable identity: `codesign -dv` shows
`Identifier=a.out`, and under launchd it gets `dial tcp …: connect: no route to host` while the very
same binary run from a terminal connects fine, because the terminal already has the permission.

Releases are signed. If you build your own:

```sh
codesign -f -s "Developer ID Application: …" --identifier net.lafox.namtrainerd \
         --options runtime --timestamp ./namtrainerd
```

Expect the first minute after replacing the binary to look broken: the launchd agent starts, cannot
reach the database, exits, and `KeepAlive` restarts it until the system lets it through. Observed:
four restarts over ~40 s, then a normal connect. Nothing is lost — a job it held goes back to the
queue and resumes from its checkpoint. A library on the SAME machine (`localhost`) is not affected.

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
* **Claim** — per lane (`train` = train + train_more, `probe` = probe_self), in drain order
  (`priority`, `queued_at`, `id`) over rows pinning the nam version this daemon runs, taken with
  `FOR UPDATE SKIP LOCKED` and stamped `worker`, `worker_instance`, a fresh `claim_token`,
  `claimed_at`/`started_at`. **Claimable is two states**: a `queued` row that nobody has asked to stop
  or cancel, or a `running` one that cron marked `paused_at` because its task went silent — the second
  is a handover, and taking it clears the mark so it happens exactly once. A paused row carrying a
  stop or a cancel IS claimable, because it has no holder and the claim is the only hand that can
  carry the ask out: the job is closed there and then, without spawning anything. **Every write that
  keeps weights is fenced on `(id, claim_token, state = 'running', paused_at IS NULL)`**; progress,
  epochs and log lines are fenced on the claim alone, so a paused run's counter may run a little ahead
  of the weights until the next claim resets it.
* **Materialize** — the take's wav from `take_audio` (sha-verified against the bytes the job pins,
  header-validated: 48 kHz, 30 s..20 min, ≤ 200 MB), the stimulus sha checked against the daemon's
  own (`signal_mismatch`), and **the take's own weights** as `--resume-from` when it has any — so
  every run of a take continues from where the take is, whichever machine trained it there. A
  continuation with nothing to continue from fails `base_unavailable`; it is never run from scratch.
  Then the trainer is spawned in its own process group and `pgid` recorded.
* **Progress** — `epoch` / `s_per_epoch` and one `job_log` row per stdout line. This is also the
  job row's ONLY pulse while a run is going: one write per epoch, at the epoch's start. **The cron
  line's quiet window must outlast the slowest epoch in the workshop and the silent start of a run**
  (torch import, dataset build, latency analysis — tens of seconds on a warm Mac, minutes on a cold
  CPU box). Under that, a healthy run is marked claimable, loses its fence at its next write, is
  killed and re-claimed, and starts its import over: the card runs flat out and nothing is trained.
* **Every finished epoch goes into the library** — the newest intact pair on disk, both halves, into
  `take_checkpoint` (the take's row, not the job's): the LAST completed epoch for continuing and the
  BEST by validation ESR for listening. So losing a trainer costs the unfinished epoch instead of the
  run, and a player can hear a take that is training right now.
* **That same write is the only channel.** It answers three things at once: is this run still ours
  (no → stop, say nothing, the row is not ours to write), has it been asked to stop (→ close it with
  what the library holds), has it been cancelled (→ kill, keep nothing). There is no poll: two
  channels answering one question are two things to keep in agreement. The price is latency — a
  cancel lands at the end of the epoch in progress rather than within two seconds.
* **Terminal write, one transaction** — `state`, `finished_at`, `reached` (passed explicitly from
  what the trainer was spawned with, or what the library holds when a stop closes it — never read
  back), `esr`, `verdict` (probe_self: `pass`/`fail`), `error_code` from the closed `job_error` list,
  `error_message`, the run's provenance (`nam_version`, `driver_sha256`, `signal_sha256`) and, for a
  success, the `job_result` row (`.nam` bytes + sha/size, `epochs` = reached, `esr`). The checkpoint
  is not in there: the weights live once, on the take. A stall (no output for 15 minutes — torch
  import is silent for minutes, so this slack is deliberate) fails `stalled`; a `train_more` that dies
  before its first epoch line fails `resume_failed`, after it `train_failed`.
* **Restart recovery kills children and touches nothing else.** At start the daemon argv-checks each
  recorded `pgid` of its own previous life before killing it, sweeps orphans by scratch path, and
  wipes scratch. A child of a process that is gone still holds a GPU, and only the machine it runs on
  can kill it — that is the trainer's job. What happens to the ROWS is the library's: a run whose task
  has gone silent is marked claimable by cron, keeping its worker, its claim, its epoch and its
  weights. A graceful shutdown or a menu-bar pause still puts its own in-flight rows back in the queue
  directly, because there the daemon knows, rather than guesses, that they are free.

On macOS the daemon also puts a small status item in the menu bar, and everything in it is about THIS
machine — the shared queue, with everyone's names on it, is the app's view. While this machine has a
run: `1/1 13:36 5.14` — its own **running / cap**, the clock time it expects to be free (the longest
of its own runs at its own speed; `24h+` past a day), and the moving-average **seconds per epoch**
(the same number the heartbeat reports). The dropdown lists its own runs, then **Pause now** (running jobs stop this
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
