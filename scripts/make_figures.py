"""Rebuild the README figures from `results/` (and `data/` for example frames).

    python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

R, OUT, DATA = Path("results"), Path("docs/figures"), Path("data/cadica")
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.titleweight": "bold", "axes.titlesize": 11,
})
NAMES = {"mobilenet_v2": "MobileNetV2", "mobilenet_v3_small": "MobileNetV3-Small", "resnet18": "ResNet-18",
         "shufflenet_v2_x0_5": "ShuffleNetV2 0.5x", "squeezenet1_1": "SqueezeNet 1.1",
         "resnet18_radiologynet": "ResNet-18 (RadiologyNET)"}


def overfitting():
    h = pd.read_csv(R / "training_history/resnet18_finetune_vs_frozen.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    panels = [("fine_tuned_layer4", "Fine-tuned backbone", 3), ("frozen_backbone", "Frozen backbone", 1)]
    for ax, (key, title, sel) in zip(axes, panels):
        g = h[h.setting == key]
        for col, color, label in (("train_auc", BLUE, "Train patients"), ("val_auc", ORANGE, "New patients (validation)")):
            ax.plot(g.epoch, g[col], color=color, lw=2, marker="o", ms=4, label=label)
        r = g[g.epoch == sel].iloc[0]
        ax.annotate("", xy=(sel, r.val_auc), xytext=(sel, r.train_auc),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
        ax.text(sel + 0.3, (r.train_auc + r.val_auc) / 2, f"gap {r.train_auc - r.val_auc:+.2f}",
                color=INK2, va="center", fontsize=9)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_ylim(0.5, 1.02)
    axes[0].set_ylabel("ROC AUC")
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("Fine-tuning memorized the training patients; a frozen backbone keeps train and new-patient scores aligned.",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "overfitting.png", dpi=200)


def comparison():
    d = pd.read_csv(R / "model_comparison.csv").sort_values("val_auc")
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    y = range(len(d))
    ax.hlines(y, d.test_auc_ci_low, d.test_auc_ci_high, color=BLUE, lw=2, alpha=0.35)
    ax.scatter(d.test_auc, y, color=BLUE, s=55, zorder=3, label="Test AUC (95% CI over patients)",
               edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(d.val_auc, y, facecolor=SURFACE, edgecolor=ORANGE, s=55, lw=2, zorder=3, label="Validation AUC")
    labels = [NAMES[m] + f"  ({p / 1e6:.1f}M)" for m, p in zip(d.model, d.params)]
    ax.set_yticks(list(y), labels, color=INK2)
    top = len(d) - 1
    ax.text(d.test_auc_ci_high.iloc[top] + 0.01, top, "selected on validation", va="center",
            fontsize=9, color=INK, fontweight="bold")
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("ROC AUC")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    ax.set_title("One protocol, six frozen backbones", loc="left", pad=26)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=200)


def examples():
    if not (DATA / "manifest.csv").exists():
        print("skip examples: run scripts/prepare_data.py first")
        return
    man = pd.read_csv(DATA / "manifest.csv").set_index("frame_uid")
    boxes = pd.read_csv(DATA / "boxes.csv")
    picks = [("p40_v16_00024", "Lesion"), ("p36_v22_00037", "Lesion"),
             ("p4_v4_00008", "No lesion"), ("p40_v14_00034", "No lesion")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (uid, title) in zip(axes, picks):
        ax.imshow(Image.open(DATA / man.loc[uid, "frame_path"]).convert("L"), cmap="gray")
        for b in boxes[boxes.frame_uid == uid].itertuples():
            ax.add_patch(plt.Rectangle((b.x, b.y), b.w, b.h, fill=False, ec="#1baf7a", lw=2))
        ax.set_title(title, loc="left", fontsize=10)
        ax.axis("off")
    fig.suptitle("CADICA angiography frames (green box = expert lesion annotation)", x=0.01, ha="left",
                 fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=150)


if __name__ == "__main__":
    overfitting(); comparison(); examples()
    print("figures ->", OUT)
