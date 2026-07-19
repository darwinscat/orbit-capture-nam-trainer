# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.
"""
trainer_driver.py — train ONE .nam from one reamp take, against the PINNED neural-amp-modeler
(0.13.x API: no `architecture` kwarg — the default recipe trains; TrainOutput carries
metadata.validation_esr). Executed by OrbitCapture NAM's hidden managed runtime.

stdout contract (parsed by the app):
    DRIVER: training <name> epochs=<n>
    ... lightning's own progress ("Epoch k/n ...") streams through unmodified ...
    DRIVER: esr=<float>
    DRIVER: exported <name>.nam
Also writes <outdir>/<name>.train.json — {"esr": ..., "epochs": ..., "finished_at": ...} — so the
queue can colour convergence long after the run.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import tempfile
import traceback
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one .nam from one reamp take.")
    ap.add_argument("--input", required=True, help="the standard NAM input wav (v3_0_0.wav)")
    ap.add_argument("--output", required=True, help="the captured take wav")
    ap.add_argument("--outdir", required=True, help="folder to write <name>.nam into")
    ap.add_argument("--name", required=True, help="model base name (= the take name)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--arch", default="standard",
                    help="accepted for forward-compat; 0.13 trains its default recipe")
    a = ap.parse_args()

    import shutil

    from nam.train.core import train  # pinned by the app's runtime manifest

    # Lightning's progress bar is silent when stdout is a pipe (non-TTY), so the host would see no
    # per-epoch progress. train() builds its own Trainer with no callbacks hook, so patch Trainer to
    # ride an extra callback that prints a parseable, flushed "Epoch k/n" at each epoch start — the
    # app turns that into the row's progress bar + ETA.
    import pytorch_lightning as _pl

    class _EpochEcho(_pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            print(f"Epoch {trainer.current_epoch}/{a.epochs}", flush=True)

    class _EsrEcho(_pl.Callback):
        # Emit the per-epoch validation ESR so the host can watch convergence (and
        # study whether a short probe converges early). The default packed recipe
        # logs the aggregate under "ESR" in callback_metrics — the SAME quantity as
        # the final `DRIVER: esr=` line. current_epoch is still k here (validation
        # runs at the end of training epoch k, before the increment), so it lines up
        # 1:1 with the "Epoch k/n" line.
        def on_validation_epoch_end(self, trainer, pl_module):
            if getattr(trainer, "sanity_checking", False):
                return  # skip the pre-train sanity-check pass (meaningless ESR)
            esr = trainer.callback_metrics.get("ESR")
            if esr is None:
                return
            esr = esr.item() if hasattr(esr, "item") else float(esr)
            print(f"DRIVER: epoch_esr={trainer.current_epoch}={esr:.8f}", flush=True)

    _orig_trainer_init = _pl.Trainer.__init__

    def _trainer_init(self, *args, **kw):
        kw["callbacks"] = list(kw.get("callbacks") or []) + [_EpochEcho(), _EsrEcho()]
        _orig_trainer_init(self, *args, **kw)

    _pl.Trainer.__init__ = _trainer_init

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # Per-take work dir (lightning's checkpoints/logs land here, NOT in models/). Parallel takes share
    # `out` (models/), so a single shared .train-work made workers rmtree each other's live logs
    # (FileNotFoundError on lightning_logs/version_0). A unique temp dir per process isolates them.
    work = Path(tempfile.mkdtemp(prefix=".train-work-", dir=str(out)))
    print(f"DRIVER: training {a.name} epochs={a.epochs}", flush=True)

    result = train(
        input_path=a.input,
        output_path=a.output,
        train_path=str(work),
        epochs=a.epochs,
        silent=True,          # no plot windows; lightning still logs epoch progress to the console
        save_plot=False,
        modelname=a.name,
    )
    if result is None or result.model is None:
        # V3/self-ESR data checks failed (or latency detection did): a clean marker so the app's
        # probe lane can flag the take for re-capture without parsing NAM's free-form check text.
        print("DRIVER: checkfail", flush=True)
        sys.exit("DRIVER: train() returned nothing (input/output checks failed)")

    esr = getattr(result.metadata, "validation_esr", None)
    # Fixed-point, never Python's float repr (which can be scientific, e.g. 9.3e-05 — the app's
    # parser would mis-read that). A missing value prints a distinct `na` token, not '0'.
    print(f"DRIVER: esr={esr:.8f}" if esr is not None else "DRIVER: esr=na", flush=True)

    result.model.net.export(out, basename=a.name)
    (out / f"{a.name}.train.json").write_text(json.dumps({
        "esr": esr,
        "epochs": a.epochs,
        "trainer": "neural-amp-modeler",
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, indent=1))
    shutil.rmtree(work, ignore_errors=True)   # models/ holds deliverables only
    print(f"DRIVER: exported {a.name}.nam", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
