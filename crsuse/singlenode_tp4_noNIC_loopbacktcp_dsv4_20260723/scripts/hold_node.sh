#!/bin/bash
echo "held: $(hostname) job=$SLURM_JOB_ID gpus=$ROCR_VISIBLE_DEVICES"
sleep 28800
