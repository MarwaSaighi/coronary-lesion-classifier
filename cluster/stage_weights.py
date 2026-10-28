"""Download and cache all ImageNet backbones (run on a node WITH internet).

GPU compute nodes on many HPC clusters are offline; torchvision then reads the
weights from the shared cache ($TORCH_HOME) populated here.
"""
from torchvision import models

from angio_cls.config import IMAGENET_WEIGHTS

for arch, enum in IMAGENET_WEIGHTS.items():
    cls, member = enum.split(".")
    getattr(models, arch)(weights=getattr(getattr(models, cls), member))
    print("cached", arch)
