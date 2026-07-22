# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.
"""
trainer_driver.py — train ONE .nam from one reamp take, against the PINNED neural-amp-modeler
(0.13.x API: no `architecture` kwarg — the default recipe trains; TrainOutput carries
metadata.validation_esr). Executed by OrbitCapture NAM's hidden managed runtime.

stdout contract (parsed by the app):
    DRIVER: training <name> epochs=<n>
    DRIVER: resuming from epoch <k>   (train_more only: printed once when --resume-from is set,
                                       from the checkpoint's RESTORED epoch — Lightning reads it
                                       out of the pickle, we do not guess it from a filename)
    ... lightning's own progress ("Epoch k/n ...") streams through unmodified ...
    DRIVER: esr=<float>
    DRIVER: no ckpt found             (only when no checkpoint could be found to export)
    DRIVER: exported <name>.nam
Also writes <outdir>/<name>.train.json — {"esr", "epochs", "finished_at", and, on a resume,
"resumed_from_epoch"} — so the queue can colour convergence long after the run.

Every successful run ALSO exports its last checkpoint atomically to <outdir>/model.ckpt (weights +
optimizer/LR state, ~505 KB). That freeze-dried state lets a later `train_more` continue from this
run via `--resume-from`; the ckpt is born, stored and dies inside the daemon and never crosses the
wire.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path


# nam 0.13 keeps TWO ModelCheckpoints in <work>/**/checkpoints/:
# `checkpoint_best_{epoch:04d}_{...}` (top-3) and `checkpoint_last_{epoch:04d}_{step}`
# (keep-latest). The epoch is the zero-padded field right after the prefix; PL's own
# default naming ("epoch=NN-step=..") is matched too as a belt-and-braces fallback.
_CKPT_EPOCH_RE = re.compile(r"checkpoint_(?:best|last)_0*(\d+)|epoch=0*(\d+)")


def _ckpt_epoch(name: str) -> int:
    """Read the training epoch encoded in a checkpoint filename, or -1 if none."""
    m = _CKPT_EPOCH_RE.search(name)
    if not m:
        return -1
    return int(m.group(1) if m.group(1) is not None else m.group(2))


def _find_last_ckpt(work: Path) -> Path | None:
    """Locate the run's final checkpoint under <work>/**/checkpoints/.

    Prefer the explicit `checkpoint_last_*.ckpt` (a true last-epoch ckpt always
    exists); among several, take the highest epoch. Fall back to the highest
    `epoch=`-parse over ALL *.ckpt (best included). NEVER mtime — best and last land
    milliseconds apart, so file times cannot order them. The glob is extension-strict
    (`.ckpt`) because nam writes a `.nam` beside every ckpt.
    """
    lasts = list(work.glob("**/checkpoints/checkpoint_last_*.ckpt"))
    if lasts:
        return max(lasts, key=lambda p: _ckpt_epoch(p.name))
    scored = [(p, _ckpt_epoch(p.name)) for p in work.glob("**/checkpoints/*.ckpt")]
    scored = [(p, e) for (p, e) in scored if e >= 0]
    if not scored:
        return None
    return max(scored, key=lambda pe: pe[1])[0]


def _export_last_ckpt(work: Path, out: Path) -> None:
    """Copy the run's last checkpoint to <out>/model.ckpt ATOMICALLY.

    A stall-kill mid-copy must never leave a torn file that a succeeded probe_e10
    would go on to store as a poisoned parent seed — so write model.ckpt.tmp first
    and os.replace it into place (same filesystem: tmp lives in `out`). Missing
    checkpoint → print a marker and continue; the run still succeeds, it is just not
    continuable.
    """
    src = _find_last_ckpt(work)
    if src is None:
        print("DRIVER: no ckpt found", flush=True)
        return
    tmp = out / "model.ckpt.tmp"
    shutil.copyfile(str(src), str(tmp))
    os.replace(str(tmp), str(out / "model.ckpt"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one .nam from one reamp take.")
    ap.add_argument("--input", required=True, help="the standard NAM input wav (v3_0_0.wav)")
    ap.add_argument("--output", required=True, help="the captured take wav")
    ap.add_argument("--outdir", required=True, help="folder to write <name>.nam into")
    ap.add_argument("--name", required=True, help="model base name (= the take name)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--arch", default="standard",
                    help="accepted for forward-compat; 0.13 trains its default recipe")
    ap.add_argument("--resume-from", dest="resume_from", default=None,
                    help="a .ckpt to continue training from (train_more); Lightning restores the "
                         "epoch counter + optimizer/LR state from it")
    a = ap.parse_args()

    from nam.train.core import train  # pinned by the app's runtime manifest

    # Lightning's progress bar is silent when stdout is a pipe (non-TTY), so the host would see no
    # per-epoch progress. train() builds its own Trainer with no callbacks hook, so patch Trainer to
    # ride an extra callback that prints a parseable, flushed "Epoch k/n" at each epoch start — the
    # app turns that into the row's progress bar + ETA.
    import pytorch_lightning as _pl

    # On a resume run the first training epoch Lightning enters is the checkpoint's
    # restored epoch; capture it once so it can also land in train.json.
    resume_epoch = {"value": None}

    class _EpochEcho(_pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            # Announce the resume point once, from the RESTORED current_epoch (the ckpt's
            # own state, read by Lightning — not guessed from the file name). Guarded on
            # max_epochs so it rides the same trainer the ckpt was injected into.
            if (a.resume_from and resume_epoch["value"] is None
                    and getattr(trainer, "max_epochs", None) == a.epochs):
                resume_epoch["value"] = trainer.current_epoch
                print(f"DRIVER: resuming from epoch {trainer.current_epoch}", flush=True)
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

    # Resume: inject ckpt_path into Trainer.fit so Lightning restores epoch/optimizer/LR
    # from the parent's checkpoint. Guard on the training Trainer only (max_epochs == the
    # requested total) — nam 0.13 core builds exactly one Trainer/fit so this fires once;
    # the guard is belt-and-braces (same spirit as the __init__ patch) and the pending
    # flag makes it inject at most once, so an unrelated fit is never hijacked.
    if a.resume_from:
        _orig_trainer_fit = _pl.Trainer.fit
        _resume_pending = {"on": True}

        def _trainer_fit(self, *args, **kw):
            if (_resume_pending["on"] and "ckpt_path" not in kw
                    and getattr(self, "max_epochs", None) == a.epochs):
                _resume_pending["on"] = False
                kw["ckpt_path"] = a.resume_from
            return _orig_trainer_fit(self, *args, **kw)

        _pl.Trainer.fit = _trainer_fit

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
    train_meta = {
        "esr": esr,
        "epochs": a.epochs,
        "trainer": "neural-amp-modeler",
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if a.resume_from:
        train_meta["resumed_from_epoch"] = resume_epoch["value"]
    (out / f"{a.name}.train.json").write_text(json.dumps(train_meta, indent=1))

    # Keep the freeze-dried trainer state (weights + Adam moments + epoch/LR counters) beside the
    # deliverable so a later train_more can continue from it. ALWAYS — resume or not — and BEFORE the
    # work dir (which holds the checkpoints) is torn down.
    _export_last_ckpt(work, out)
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
