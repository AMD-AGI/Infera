# Reproduce

Assumes a granted 8x MI355X node (openai smci355 or crusoe spur) and the external paths in `environment.md`.
`K` = this packup's `scripts/kit` copied to a path the node's docker daemon can read (NOT $HOME NFS on openai — root_squash).

## 0. One-time staging (from a16-19)
```bash
P=<this packup>; NFS=/perf_apps/yihou/dsv41_vllm_pd_20261119          # crusoe: /shared_nfs/yihou/dsv41_vllm_pd_20261119
mkdir -p $NFS/stage && rsync -a $P/scripts/kit/ $NFS/stage/kit/        # kit
rsync -a <InferenceX>/inferencex-e2e/ $NFS/stage/inferencex-e2e/        # InferenceX @ db3fc796
# venv-agentx (agentx-harness 1.0.14, uv CPython 3.11) + hf_home (dataset @23f152f6 + tokenizer) must exist under $NFS:
#   see header of scripts/kit/client_selftest.sh for the build recipe. On another cluster, copy the venv and set
#   KIT_VENV_CTR=/perf_apps/yihou/dsv41_vllm_pd_20261119/venv-agentx (absolute symlinks inside the venv).
```

## 1. Node preflight + image (E0)
```bash
bash $K/pull_images.sh r1-dp2-c128 --router-wheel
bash $K/preflight.sh --in-image-all          # facts.json/md under $NFS/preflight/
bash $K/client_selftest.sh                   # must print SELFTEST PASS
```

## 2. One scored point = fresh server + official 3600 s bench
```bash
# openai (from a16-19): verifies 'launched' + run dir
bash $K/launch_remote.sh <node> <config-id> <gpus> <port-offset> <conc> <tag-prefix>
#   R2 (public bar):  r2-dp2-umbp-c384 4,5 20 384 e2-r2-c384
#   R1:               r1-dp2-c128      0,1  0 128 e1-r1-c128      (c192/c256: same config, conc 192/256)
#   E4c:              e4c-r1-dsmla-c128          6,7 30 128 e4c-r1ds-c128
#   E4b:              e4b-r1-dsmla-nosparse-c128 2,3 10 128 e4b-r1dsnosp-c128
# crusoe (from spur005): spur exec <job> -- bash $NFS/cru_launch.sh <config-id> <gpus> <offset> <conc> <tag-prefix>
```
Each chain = `serve.sh <cfg> <gpus> <tag> --port-offset N --wait` then `run_point.sh <tag> http://127.0.0.1:<8000+N> <conc> 2 official`.
Results: `$NFS/runs/<tag>/bench/run_point.log` (summary; copied here as runs/<tag>/run_point.summary.txt), `inferencex_result.json`, ledger `$NFS/results/ledger.csv`.
If the harness post-processing did not run, recompute: `python3 -I scripts/raw_score_from_export.py <profile_export.jsonl> 2`.
Stop: `bash $K/stop.sh <tag>`.

## 3. PD 1P1D (E7, prepared, not yet measured)
See `scripts/kit/PD_README.md`: `serve_pd.sh e7s-pd-mooncake-1p1d-dp2-smoke 0,1 2,3 <tag> --wait`, reference
`serve.sh e7r-colo-dsmla-ref-smoke 4,5 <ref-tag> --port-offset 40 --wait`, then `pd_smoke.sh <tag> http://127.0.0.1:8040 --docker`.

## Expected (same image/node class; single runs, ~4% replicate spread)
R2 c384 ≈ 407k (CI 427k); R1 c128 ≈ 256-267k (CI 271k); R1 c192 ≈ 336k, c256 ≈ 347k; E4c ≈ 244k; E4b ≈ 221k.
