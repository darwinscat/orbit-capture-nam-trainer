# orbit-capture-nam-trainer

A single-binary macOS daemon that trains [NAM](https://github.com/sdatkinson/neural-amp-modeler)
(Neural Amp Modeler) `.nam` models. It takes a reamped capture over HTTP, runs a self-provisioned
python trainer, and serves back the `.nam` — so the
[OrbitCapture NAM](https://github.com/darwinscat/orbit-capture-nam) desktop app (or any client) can
train captures without shipping python itself.

By Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. mac-only (Apple Silicon / MPS).

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

First run provisions its own python (python-build-standalone + a venv + `neural-amp-modeler`) and
fetches the capture signal, one time. `GET /v1/health` reports `ready:false` until it is up. Config
and the bearer token live in `~/Library/Application Support/OrbitCaptureNamTrainer/config.toml` —
`port` (8626), `bind` (127.0.0.1; set it to a Tailscale IP for remote access), `cap`,
`retention_days`. Auto-start under launchd: `deploy/launchd/`.

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

Kinds: `train` (produces a `.nam`), `probe_self` (self-ESR verdict in seconds), `probe_e10`
(10-epoch ESR probe).

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE).
