#!/bin/bash
# Hold allocation; sleep infinity so that `scontrol update TimeLimit=` actually extends it.
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
echo "ROCR=$ROCR_VISIBLE_DEVICES cpus=$(nproc)"
exec sleep infinity
