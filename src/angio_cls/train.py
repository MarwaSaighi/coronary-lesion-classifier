"""Train a frozen-backbone + MLP lesion classifier.

    python -m angio_cls.train --config configs/mobilenet_v2.yaml --data data/cadica --out runs

Protocol: unweighted cross-entropy, AdamW on the head only (lr 1e-3, wd 1e-4),
batch 32, up to 30 epochs, early stopping (patience 8) on validation ROC AUC.
The checkpoint with the highest validation ROC AUC is kept (earliest on ties).
Every epoch writes a resumable checkpoint, so a job killed by a cluster
walltime limit restarts where it stopped.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .data import FrameDataset, eval_transform, load_manifest, train_transform
from .metrics import summary
from .models import LesionClassifier, p_lesion


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: LesionClassifier, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval_mode()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, t in loader:
        x, t = x.to(device), t.to(device)
        logits = model(x)
        loss_sum += ce(logits, t).item(); n += len(t)
        ps.append(p_lesion(logits).cpu().numpy())
        ys.append((t == 0).int().cpu().numpy())          # 1 = lesion
    return np.concatenate(ys), np.concatenate(ps), loss_sum / n


def train(cfg: dict, data_dir: str, out_dir: str, num_workers: int = 4) -> dict:
    name = cfg["run_name"]
    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_manifest(data_dir)
    tr, va = df[df.fold == "train"], df[df.fold == "val"]
    kw = dict(batch_size=cfg.get("batch_size", 32), num_workers=num_workers)
    train_loader = DataLoader(FrameDataset(tr, train_transform()), shuffle=True, pin_memory=torch.cuda.is_available(), **kw)
    val_loader = DataLoader(FrameDataset(va, eval_transform()), **kw)
    train_eval_loader = DataLoader(FrameDataset(tr, eval_transform()), **kw)

    model = LesionClassifier(cfg["arch"], cfg.get("pretrain", "imagenet"), cfg.get("weights_path"),
                             cfg.get("hidden", 128), cfg.get("dropout", 0.3), cfg.get("seed", 0)).to(device)
    opt = torch.optim.AdamW(model.head.parameters(), lr=cfg.get("lr", 1e-3),
                            weight_decay=cfg.get("weight_decay", 1e-4))
    criterion = nn.CrossEntropyLoss()

    history, best, start, stale = [], None, 1, 0
    last = run_dir / "last.pt"
    if last.exists():                                     # resume after walltime kill
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        history, best, stale, start = ck["history"], ck["best"], ck["stale"], ck["epoch"] + 1
        print(f"[{name}] resuming at epoch {start}")

    for epoch in range(start, cfg.get("max_epochs", 30) + 1):
        model.train_mode()
        loss_sum, n = 0.0, 0
        for x, t in train_loader:
            x, t = x.to(device), t.to(device)
            opt.zero_grad()
            loss = criterion(model(x), t)
            loss.backward(); opt.step()
            loss_sum += loss.item() * len(t); n += len(t)

        yv, pv, val_loss = predict(model, val_loader, device)
        yt, pt, _ = predict(model, train_eval_loader, device)
        rec = {"epoch": epoch, "train_loss": loss_sum / n, "val_loss": val_loss,
               "train_auc": summary(yt, pt)["roc_auc"], "val_auc": summary(yv, pv)["roc_auc"]}
        history.append(rec)
        improved = best is None or rec["val_auc"] > best["val_auc"] + 1e-12
        if improved:
            best, stale = {"epoch": epoch, "val_auc": rec["val_auc"]}, 0
        else:
            stale += 1
        print(f"[{name}] ep{epoch:02d} train_auc={rec['train_auc']:.3f} val_auc={rec['val_auc']:.3f}"
              + (" *" if improved else ""), flush=True)

        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                 "history": history, "best": best, "stale": stale, "config": cfg}
        torch.save(state, run_dir / "last.tmp"); shutil.move(run_dir / "last.tmp", last)
        if improved:
            shutil.copy2(last, run_dir / "best.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if stale >= cfg.get("patience", 8):
            print(f"[{name}] early stop; best epoch {best['epoch']} (val AUC {best['val_auc']:.3f})")
            break

    out = {"run_name": name, "best_epoch": best["epoch"], "best_val_auc": best["val_auc"],
           "params": model.n_params(), "feature_dim": model.feature_dim}
    json.dump(out, open(run_dir / "summary.json", "w"), indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/cadica")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    train(yaml.safe_load(open(a.config)), a.data, a.out, a.workers)


if __name__ == "__main__":
    main()
