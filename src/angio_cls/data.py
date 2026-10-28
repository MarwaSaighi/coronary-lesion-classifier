"""CADICA data preparation, patient-level split, transforms and dataset.

Labelling rule (from the CADICA annotations):
  * only the expert-selected frames of each video are used
    (`selectedVideos/<p>/<v>/<p>_<v>_selectedFrames.txt`);
  * a frame is `lesion` if its video is listed in `lesionVideos.txt`,
    otherwise `non_lesion` (listed in `nonlesionVideos.txt`).
This yields 6,126 frames from 42 patients (3,996 lesion / 2,130 non-lesion).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import CLASS_TO_IDX, EXPECTED_COUNTS, INPUT_SIZE, N_PATIENTS, NORM_MEAN, NORM_STD, fold_of


class _Source:
    """Uniform read access to CADICA as a .zip or as an extracted folder."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.suffix == ".zip":
            self.zip = zipfile.ZipFile(self.path)
            names = self.zip.namelist()
            root = next(n for n in names if n.rstrip("/").endswith("CADICA"))
            self.prefix = root.rstrip("/") + "/"
            self.names = set(names)
        else:
            self.zip = None
            self.prefix = str(self.path / "CADICA") + "/" if (self.path / "CADICA").is_dir() else str(self.path) + "/"

    def exists(self, rel: str) -> bool:
        if self.zip:
            return self.prefix + rel in self.names
        return Path(self.prefix + rel).exists()

    def read(self, rel: str) -> bytes:
        if self.zip:
            return self.zip.read(self.prefix + rel)
        return Path(self.prefix + rel).read_bytes()

    def patients(self) -> list[str]:
        if self.zip:
            pats = {n[len(self.prefix):].split("/")[1] for n in self.names
                    if n.startswith(self.prefix + "selectedVideos/p") and n.count("/") >= 3}
        else:
            pats = {p.name for p in Path(self.prefix + "selectedVideos").iterdir() if p.is_dir()}
        return sorted(pats, key=lambda p: int(p[1:]))


def _parse_boxes(text: str) -> list[tuple[int, int, int, int, str]]:
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5:
            x, y, w, h = map(int, parts[:4])
            boxes.append((x, y, w, h, parts[4]))
    return boxes


def build_manifest(cadica: str | Path, out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the selected frames and write `manifest.csv` + `boxes.csv`."""
    src = _Source(cadica)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    rows, box_rows = [], []
    for pid in src.patients():
        base = f"selectedVideos/{pid}/"
        for list_file, label in (("lesionVideos.txt", "lesion"), ("nonlesionVideos.txt", "non_lesion")):
            if not src.exists(base + list_file):
                continue
            for vid in src.read(base + list_file).decode().split():
                for uid in src.read(f"{base}{vid}/{pid}_{vid}_selectedFrames.txt").decode().split():
                    dst = frames_dir / pid / vid / f"{uid}.png"
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(src.read(f"{base}{vid}/input/{uid}.png"))
                    gt = f"{base}{vid}/groundtruth/{uid}.txt"
                    if label == "lesion" and src.exists(gt):
                        for x, y, w, h, cat in _parse_boxes(src.read(gt).decode()):
                            box_rows.append(dict(frame_uid=uid, x=x, y=y, w=w, h=h, category=cat))
                    rows.append(dict(frame_uid=uid, patient_id=pid, video_id=vid, label=label,
                                     fold=fold_of(pid), frame_path=dst.relative_to(out_dir).as_posix()))
    manifest = pd.DataFrame(rows)
    boxes = pd.DataFrame(box_rows, columns=["frame_uid", "x", "y", "w", "h", "category"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    boxes.to_csv(out_dir / "boxes.csv", index=False)
    return manifest, boxes


def audit_manifest(manifest: pd.DataFrame) -> dict:
    """Leakage and integrity checks. Raises AssertionError on any failure."""
    assert manifest["frame_uid"].is_unique, "duplicate frames"
    assert manifest["patient_id"].nunique() == N_PATIENTS, "unexpected patient count"
    folds_per_patient = manifest.groupby("patient_id")["fold"].nunique()
    assert (folds_per_patient == 1).all(), "a patient appears in more than one split"
    videos = manifest.assign(v=manifest.patient_id + "_" + manifest.video_id)
    assert (videos.groupby("v")["fold"].nunique() == 1).all(), "a video crosses splits"
    counts = manifest.groupby(["fold", "label"]).size().unstack(fill_value=0)
    for fold, exp in EXPECTED_COUNTS.items():
        for label, n in exp.items():
            assert counts.loc[fold, label] == n, f"{fold}/{label}: {counts.loc[fold, label]} != {n}"
    return {"frames": len(manifest), "patients": int(manifest.patient_id.nunique()),
            "counts": counts.to_dict(), "patients_crossing_splits": 0, "videos_crossing_splits": 0}


def load_manifest(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "manifest.csv")
    df["frame_path"] = [str(data_dir / p) for p in df["frame_path"]]
    return df


def train_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


class FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[i]
        img = Image.open(row["frame_path"]).convert("RGB")
        return self.transform(img), CLASS_TO_IDX[row["label"]]
