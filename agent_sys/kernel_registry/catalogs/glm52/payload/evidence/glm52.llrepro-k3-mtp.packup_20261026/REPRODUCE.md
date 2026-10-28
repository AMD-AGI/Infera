> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Reproduce

Prerequisites (same as `registry/glm52.llrepro-pp4-dpa4.packup_20261024` and later pack-ups): image
`historical-sglang-image-b` on both nodes, contributor 4+4 kit with our hooks at
`REDACTED_LOCAL_PATH` (our `worker.sh` / `common.sh` in `scripts/`), KF
overlay `REDACTED_LOCAL_PATH`, model `REDACTED_LOCAL_PATH`.

## E2E (on compute-node-03; decode on REDACTED_NODE started over ssh by the kit)
```bash
D=REDACTED_LOCAL_PATH
# best (llr27): K3
bash $D/run_llr.registry.sh $D/llr27.site.env llr27-c96 > $D/run_llr.llr27.registry.log 2>&1 &
bash $D/guard_llr.registry.sh llr27 > $D/guard.llr27.registry.log 2>&1 &
# or several in sequence: bash $D/queue_runs.registry.sh <prev-tag> llr29:$D/llr29.site.env llr31:$D/llr31.site.env
python3 -I $D/llsum.registry.py $D/work/runs/llr27-c96; python3 -I $D/bneck.registry.py $D/work/runs/llr27-c96
```
Site files are in `sites/`. The only differences from llr24 are the lines described in each file's header comment
(llr26: `SGLANG_GLM_NEXTN_MOE_PTPC=1` in DECODE_ENV_EXTRA; llr27: `SIM_ACCEPT_LEN=2.99` and
`--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` on both legs; llr28: three
`DECODE_MOUNT_EXTRA` JSON mounts from `mla_bmm/out`; llr25/llr31: `--max-running-requests 320 --cuda-graph-max-bs-decode 80`).

## Real acceptance (any free 4-GPU MI355X node with the image)
`acceptance/start.sh` (server, no fused flags, `--disable-custom-all-reduce`, no SIM), `build.py` / `build_real.py`
(requests; need the AgentX HF cache and the GLM tokenizer), `bench.py`, `runall.sh`, `analyze.py` -> `summary.md`.

## aiter GEMM retune
`aiter_retune/mk_untuned.registry.py <decode log> <out dir>` -> untuned csv; tune inside the image:
`python3 csrc/gemm_a16w16/gemm_a16w16_tune.py --input_file <untuned> --tuned_file <tuned> -o2 <prof> --mp 4 --libtype all --with-hipblaslt`
(see `aiter_retune/k3/tune_k3.registry.sh`). Mount the tuned csv as
`REDACTED_LOCAL_PATH` (aiter merges every
`*bf16_tuned_gemm*.csv` there; a duplicate key raises at start-up).
