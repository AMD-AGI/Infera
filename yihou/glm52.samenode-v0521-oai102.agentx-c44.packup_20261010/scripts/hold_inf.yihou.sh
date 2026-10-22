#!/bin/bash
# Hold the allocation for its whole TimeLimit. `sleep infinity` (not a fixed sleep) so that a
# later `scontrol update TimeLimit=...` actually extends the hold (the 86400 s sleep of the
# previous hold script ended job 200276 at 24 h despite a raised limit).
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
echo "ROCR=$ROCR_VISIBLE_DEVICES cpus=$(nproc)"
exec sleep infinity
