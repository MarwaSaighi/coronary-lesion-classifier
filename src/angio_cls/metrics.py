"""Metrics, validation-based threshold selection and patient-level bootstrap.

Positive class = lesion. Scores are P(lesion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

# Candidate decision thresholds, searched on the VALIDATION split only.
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"threshold": float(thr), "sensitivity": sens, "specificity": spec,
            "balanced_accuracy": (sens + spec) / 2,
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def summary(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    out = at_threshold(y, p, thr)
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return out


def select_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Threshold maximising validation balanced accuracy (lowest wins ties)."""
    bal = [at_threshold(y_val, p_val, t)["balanced_accuracy"] for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmax(bal))])


def patient_bootstrap_auc(df: pd.DataFrame, n_reps: int = 2000, seed: int = 0,
                          score_col: str = "prob_lesion", label_col: str = "true_lesion") -> tuple[float, float]:
    """95% CI of ROC AUC, resampling whole PATIENTS (never single frames).

    Frames from one patient are strongly correlated, so resampling frames would
    give a falsely narrow interval.
    """
    y, p = df[label_col].to_numpy(), df[score_col].to_numpy()
    idx = {q: np.flatnonzero(df["patient_id"].to_numpy() == q) for q in df["patient_id"].unique()}
    pats = np.array(list(idx))
    rng = np.random.default_rng(seed)
    aucs = []
    while len(aucs) < n_reps:
        ii = np.concatenate([idx[q] for q in rng.choice(pats, len(pats))])
        if len(np.unique(y[ii])) == 2:
            aucs.append(roc_auc_score(y[ii], p[ii]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def per_patient(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("patient_id"):
        y, p = g["true_lesion"].to_numpy(), g["prob_lesion"].to_numpy()
        rows.append({"patient_id": pid, "frames": len(g), "lesion_frames": int(y.sum()),
                     "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
                     "accuracy": float(((p >= thr).astype(int) == y).mean())})
    return pd.DataFrame(rows)
