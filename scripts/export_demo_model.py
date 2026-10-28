"""Strip a training checkpoint down to the weights the demo needs.

    python scripts/export_demo_model.py runs/mobilenet_v2/best.pt demo/model.pt --threshold 0.57
"""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("out")
ap.add_argument("--arch", default="mobilenet_v2")
ap.add_argument("--threshold", type=float, required=True, help="validation-selected threshold")
a = ap.parse_args()

ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
state = ck.get("model", ck.get("model_state_dict"))
torch.save({"arch": a.arch, "threshold": a.threshold, "state_dict": state}, a.out)
print(f"saved {a.out}")
