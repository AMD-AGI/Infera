# Reproduce

Prerequisites (same as `yihou/glm52.llrepro-pp4-dpa4.packup_20261024` and later pack-ups): image
`infera-sglang:v0522-fdbafb60` on both nodes, llying 4+4 kit with our hooks at
`/shared_nfs/yihou_glm52/llrepro_20261023/kit/sglang_2p1d_glm5.2` (our `worker.sh` / `common.sh` in `scripts/`), KF
overlay `/shared_nfs/yihou_glm52/kf_e2e_20261023/kf/e2e/overlay`, model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.

## E2E (on crsuse2-m2m-135; decode on 136 started over ssh by the kit)
```bash
D=/shared_nfs/yihou_glm52/llrepro_20261023
# best (llr27): K3
bash $D/run_llr.yihou.sh $D/llr27.site.env llr27-c96 > $D/run_llr.llr27.yihou.log 2>&1 &
bash $D/guard_llr.yihou.sh llr27 > $D/guard.llr27.yihou.log 2>&1 &
# or several in sequence: bash $D/queue_runs.yihou.sh <prev-tag> llr29:$D/llr29.site.env llr31:$D/llr31.site.env
python3 -I $D/llsum.yihou.py $D/work/runs/llr27-c96; python3 -I $D/bneck.yihou.py $D/work/runs/llr27-c96
```
Site files are in `sites/`. The only differences from llr24 are the lines described in each file's header comment
(llr26: `SGLANG_GLM_NEXTN_MOE_PTPC=1` in DECODE_ENV_EXTRA; llr27: `SIM_ACCEPT_LEN=2.99` and
`--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` on both legs; llr28: three
`DECODE_MOUNT_EXTRA` JSON mounts from `mla_bmm/out`; llr25/llr31: `--max-running-requests 320 --cuda-graph-max-bs-decode 80`).

## Real acceptance (any free 4-GPU MI355X node with the image)
`acceptance/start.sh` (server, no fused flags, `--disable-custom-all-reduce`, no SIM), `build.py` / `build_real.py`
(requests; need the AgentX HF cache and the GLM tokenizer), `bench.py`, `runall.sh`, `analyze.py` -> `summary.md`.

## aiter GEMM retune
`aiter_retune/mk_untuned.yihou.py <decode log> <out dir>` -> untuned csv; tune inside the image:
`python3 csrc/gemm_a16w16/gemm_a16w16_tune.py --input_file <untuned> --tuned_file <tuned> -o2 <prof> --mp 4 --libtype all --with-hipblaslt`
(see `aiter_retune/k3/tune_k3.yihou.sh`). Mount the tuned csv as
`/sgl-workspace/aiter/aiter/configs/model_configs/<name>_bf16_tuned_gemm.csv:ro` (aiter merges every
`*bf16_tuned_gemm*.csv` there; a duplicate key raises at start-up).
