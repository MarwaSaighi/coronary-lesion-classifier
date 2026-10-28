#!/bin/bash
# Assemble a self-contained Hugging Face Space folder from this repo.
#   bash demo/build_space.sh ../hf-space
set -euo pipefail
OUT=${1:?output dir}
mkdir -p "$OUT/examples"
cp demo/app.py demo/requirements.txt demo/model.pt "$OUT/"
cp demo/examples/*.png "$OUT/examples/"
cp -r src/angio_cls "$OUT/"
rm -rf "$OUT/angio_cls/__pycache__"
cp demo/SPACE_README.md "$OUT/README.md"
echo "Space ready in $OUT"
