#!/usr/bin/env bash
# Node-side prep: load the S1-S3 image if missing, verify ID, arm the job-bound guard. Usage: prep_node <JOB> <NK>
JOB=$1; NK=$2; LOG=/shared_nfs/yihou_glm52/dcp_pd_20261021/gmu/prep.$NK.log
setsid nohup bash -c "
  docker image inspect --format '{{.Id}}' infera-sglang:v0522-d395c3b3-1020-mc4340 2>/dev/null | grep -q d3d6a9c20ce0 ||
    docker load -i /shared_nfs/yihou_glm52/llying_dcp_20261022/images/infera-sglang_v0522-d395c3b3-1020-mc4340.tar
  docker image inspect --format '{{.Id}}' infera-sglang:v0522-d395c3b3-1020-mc4340 | grep -q d3d6a9c20ce0 && echo IMAGE_OK || echo IMAGE_BAD
  setsid nohup bash /shared_nfs/yihou_glm52/dcp_pd_20261021/pd/tools/job_container_guard.yihou.sh start $JOB $NK > /dev/null 2>&1 < /dev/null &
  echo GUARD_STARTED
" > $LOG 2>&1 < /dev/null &
echo "prep launched $NK log=$LOG"
