#!/bin/bash
echo "held: $(hostname) job=$SLURM_JOB_ID gpus=$ROCR_VISIBLE_DEVICES at $(date -u)"
sleep 28800   # 8h
