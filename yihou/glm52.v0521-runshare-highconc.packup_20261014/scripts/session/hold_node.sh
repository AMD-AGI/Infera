#!/bin/bash
echo "held: $(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
sleep "${HOLD_SECONDS:-28800}"
