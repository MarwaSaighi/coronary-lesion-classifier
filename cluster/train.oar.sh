#!/bin/bash
# GPU training job for an OAR-scheduled cluster (GRICAD Bigfoot in this project).
#
#   CONFIG=configs/mobilenet_v2.yaml oarsub -S ./cluster/train.oar.sh
#
# Training checkpoints every epoch and resumes from runs/<name>/last.pt, so if
# the walltime kills the job, resubmitting the same command picks up where it
# stopped. Compute nodes have no internet: run `python cluster/stage_weights.py`
# once on the login node to cache the ImageNet weights first.
#OAR -n cadica-train
#OAR -l /nodes=1/gpu=1,walltime=02:00:00
#OAR -p gpumodel!='MI210'
#OAR --project YOUR_PROJECT
#OAR -O logs/%jobid%.out
#OAR -E logs/%jobid%.err

set -euo pipefail
cd "$OAR_WORKDIR"
source /applis/environments/conda.sh
conda activate angio-cls

: "${CONFIG:?set CONFIG=configs/<model>.yaml}"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) config=$CONFIG"
python -m angio_cls.train --config "$CONFIG" --data data/cadica --out runs --workers 4
