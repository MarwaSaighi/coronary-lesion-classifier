"""Evaluate trained runs: threshold on validation, then score the test patients.

    python -m angio_cls.evaluate --runs runs --data data/cadica --out results

For every run: load `best.pt`, predict validation and test frames, pick the
decision threshold on VALIDATION only, then report test metrics with a 95%
patient-level bootstrap interval and a per-patient breakdown. Model selection
uses validation ROC AUC, never test results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import FrameDataset, eval_transform, load_manifest
from .metrics import patient_bootstrap_auc, per_patient, select_threshold, summary
from .models import LesionClassifier
from .train import predict


def evaluate_run(run_dir: Path, df: pd.DataFrame, out_dir: Path, device) -> dict:
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = LesionClassifier(cfg["arch"], "none").to(device)
    model.load_state_dict(ck["model"])
    preds = {}
    for fold in ("val", "test"):
        part = df[df.fold == fold].reset_index(drop=True)
        y, p, _ = predict(model, DataLoader(FrameDataset(part, eval_transform()), batch_size=64), device)
        preds[fold] = part[["frame_uid", "patient_id", "video_id"]].assign(true_lesion=y, prob_lesion=p)
        preds[fold].to_csv(out_dir / "predictions" / f"{cfg['run_name']}_{fold}.csv", index=False)

    thr = select_threshold(preds["val"].true_lesion, preds["val"].prob_lesion)
    te = preds["test"]
    m = summary(te.true_lesion, te.prob_lesion, thr)
    lo, hi = patient_bootstrap_auc(te)
    per_patient(te, thr).to_csv(out_dir / "per_patient" / f"{cfg['run_name']}_test.csv", index=False)
    return {"run": cfg["run_name"], "arch": cfg["arch"], "pretrain": cfg.get("pretrain", "imagenet"),
            "params_total": model.n_params()["total"], "best_epoch": ck["best"]["epoch"],
            "val_auc": summary(preds["val"].true_lesion, preds["val"].prob_lesion)["roc_auc"],
            "test_auc": m["roc_auc"], "test_auc_ci_low": lo, "test_auc_ci_high": hi,
            "threshold": thr, "test_sensitivity": m["sensitivity"],
            "test_specificity": m["specificity"], "test_balanced_accuracy": m["balanced_accuracy"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data/cadica")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    for sub in ("predictions", "per_patient"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_manifest(a.data)
    rows = [evaluate_run(r, df, out, device) for r in sorted(Path(a.runs).iterdir()) if (r / "best.pt").exists()]
    table = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    table.to_csv(out / "model_comparison.csv", index=False)
    print(table.to_string(index=False))
    json.dump({"selected_by_validation": table.iloc[0]["run"]}, open(out / "selection.json", "w"), indent=2)


if __name__ == "__main__":
    main()
