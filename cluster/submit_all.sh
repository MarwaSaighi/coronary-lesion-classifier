#!/bin/bash
# Submit one GPU job per backbone and record the job ids.
#   bash cluster/submit_all.sh
set -euo pipefail
mkdir -p logs
for cfg in configs/*.yaml; do
    out=$(CONFIG="$cfg" oarsub -S ./cluster/train.oar.sh)
    jid=$(echo "$out" | sed -n 's/.*OAR_JOB_ID=\([0-9]*\).*/\1/p')
    echo "$(date -Iseconds),$cfg,$jid" | tee -a logs/jobs.csv
done
