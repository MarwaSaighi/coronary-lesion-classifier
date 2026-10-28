# Running on an HPC cluster

All experiments in this project were trained on NVIDIA V100/A100 GPUs of the
GRICAD **Bigfoot** cluster (Université Grenoble Alpes), scheduled with
[OAR](https://oar.imag.fr/). These scripts are the portable version of that setup.

| File | Purpose |
|---|---|
| `stage_weights.py` | Cache ImageNet weights on the login node (compute nodes are offline). |
| `train.oar.sh` | One GPU training job; resources are declared with `#OAR` directives. |
| `submit_all.sh` | Submit one job per config and log the job ids to `logs/jobs.csv`. |

```bash
conda create -n angio-cls python=3.10 -y && conda activate angio-cls
pip install -e .
python cluster/stage_weights.py           # login node, once
bash cluster/submit_all.sh                # one job per backbone
oarstat -u $USER                          # monitor
```

Practical lessons from running this on a shared, often saturated cluster:

- **Make every job resumable.** Training writes a full checkpoint (model,
  optimizer, history, early-stopping state) every epoch, written to a temp file
  and atomically renamed. A walltime kill costs at most one epoch, and
  resubmitting the same job continues from there.
- **Short jobs get scheduled first.** When the GPU queue was saturated
  (estimated waits of ~18 h), short walltimes plus widening the GPU filter to
  every type the CUDA build supports (`-p gpumodel!='MI210'` excludes AMD)
  got jobs running. Resumable checkpoints made short walltimes safe.
- **Keep heavy work off the login node.** Data audits and plotting are fine
  there; model inference is not.
- **Verify transfers.** Results were copied off the cluster with a
  SHA-256 manifest and checked file by file.
