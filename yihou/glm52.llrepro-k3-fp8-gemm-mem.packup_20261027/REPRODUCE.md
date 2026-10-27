# Reproduce

Prerequisites as in `yihou/glm52.llrepro-k3-mtp.packup_20261026/REPRODUCE.md` (image `infera-sglang:v0522-fdbafb60` on
both nodes, kit `/shared_nfs/yihou_glm52/llrepro_20261023/kit/sglang_2p1d_glm5.2` with our `worker.sh`/`common.sh`
hooks in `scripts/`, KF overlay `/shared_nfs/yihou_glm52/kf_e2e_20261023/kf/e2e/overlay`, model
`/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`).

## Current stack (llr35) on crsuse2-m2m-135
```bash
D=/shared_nfs/yihou_glm52/llrepro_20261023
bash $D/run_llr.yihou.sh $D/llr35.site.env llr35-c96 > $D/run_llr.llr35.yihou.log 2>&1 &
bash $D/guard_llr.yihou.sh llr35 > $D/guard.llr35.yihou.log 2>&1 &
# sequence: bash $D/queue_runs.yihou.sh <prev-tag> llr37:$D/llr37.site.env llr38:$D/llr38.site.env
python3 -I $D/llsum.yihou.py $D/work/runs/llr35-c96; python3 -I $D/bneck.yihou.py $D/work/runs/llr35-c96
```
llr35 = llr24 (previous base, see the c96 pack-up) plus:
- both legs: `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`, `SIM_ACCEPT_LEN=2.99`;
- decode env `SGLANG_GLM_NEXTN_MOE_PTPC=1`; decode `--mem-fraction-static 0.95`;
- `DECODE_MOUNT_EXTRA` adds `aiter_retune/merged/yihou_glm52_dpa_bf16_tuned_gemm.csv` and
  `aiter_retune/ptpc/yihou_glm52_ptpc_bf16_tuned_gemm.csv` as
  `/sgl-workspace/aiter/aiter/configs/model_configs/<same name>:ro` (space-separated list; worker.sh line 148).

## Re-tuning the GEMMs for a new config
1. Collect untuned shapes from the decode log: `python3 -I aiter_retune/mk_untuned.yihou.py <decode log> <out dir>`
   (parses aiter's "not found tuned config" lines).
2. Tune inside the image on 4 free GPUs: `aiter_retune/ptpc/tune_ptpc.yihou.sh 0,1,2,3` (gemm_a16w16_tune.py
   `--libtype all --with-hipblaslt`; ~1 h for 50-65 shapes).
3. Check there is no key overlap with the image's `aiter/configs/*bf16_tuned_gemm*.csv` (a duplicate raises at start-up),
   mount, and check the decode log: aiter prints "merge tuned file under model_configs/" with our file name.
