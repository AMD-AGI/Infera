#!/usr/bin/env bash
# cleanup_run (task NFS cleanup 2026-10-27, user-approved steps 1-4; image tars untouched). ON 135.
# Each step: re-validate its list (fresh mount check) -> abort on any invalid line -> run the generated literal-path script.
set -u
A=/shared_nfs/yihou_glm52/du_audit.yihou; LOG=$A/cleanup_run.log
ts(){ date -u +%FT%TZ; }
step(){ local st=$1 mode=$2 L=$3 out
  out=$(bash $A/validate.yihou.sh $L 2>&1 | tail -1); echo "$(ts) step$st $out" >> $LOG
  [[ "$out" == *" 0 invalid"* ]] || { echo "$(ts) step$st ABORT" >> $LOG; exit 1; }
  python3 -I $A/gen_cmds.yihou.py $mode $L $A/cmds_step$st.yihou.sh >> $LOG 2>&1 || { echo "$(ts) step$st gen ABORT" >> $LOG; exit 1; }
}
echo "$(ts) START df: $(df -h /shared_nfs | tail -1)" >> $LOG
step 1 del $A/del_step1.yihou.list; bash $A/cmds_step1.yihou.sh >> $LOG 2>&1; echo "$(ts) step1 df: $(df -h /shared_nfs | tail -1)" >> $LOG
step 2 del $A/del_step2.yihou.list; bash $A/cmds_step2.yihou.sh >> $LOG 2>&1; echo "$(ts) step2 df: $(df -h /shared_nfs | tail -1)" >> $LOG
step 3 gzip $A/gzip_step3.yihou.list; bash $A/cmds_step3.yihou.sh >> $LOG 2>&1; echo "$(ts) step3 df: $(df -h /shared_nfs | tail -1)" >> $LOG
step 4 del $A/del_step4_root.yihou.list
docker run --rm --name yihou-cleanup-step4 -v /shared_nfs/yihou_glm52:/shared_nfs/yihou_glm52 --entrypoint bash infera-sglang:v0522-fdbafb60 \
  /shared_nfs/yihou_glm52/du_audit.yihou/cmds_step4.yihou.sh >> $LOG 2>&1
echo "$(ts) step4 df: $(df -h /shared_nfs | tail -1)" >> $LOG
echo "$(ts) CLEANUP_DONE" >> $LOG
