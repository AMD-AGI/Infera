# Reproduce — tw90, cx90, wts64

```bash
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness      # harness (verbatim copy in scripts/harness/)
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013             # shim, dryruns, correctness tests, metrics
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012            # image tar, metrics sampler
```

## 0. Prerequisites, node, image

These are identical to `../glm52.v0521-offloaderv2.packup_20261015/REPRODUCE.md` §0-1: oai hold, checks, `docker load` of the 9d755dd6 tar, JIT seed.
- **Secrets:** none beyond the spur cluster login.
- **Model:** `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.
- **Node 136:** plain ssh, with its own image build `sha256:9872bbda…`. It uses `config.yihou.mc4340n.sh` and the login-host drivers (`chain_cap2l`, `cont_benchl*`).

## 1. Env files (`scripts/harness/`)

| round | env | CONFIG | difference vs pf90 (dryrun tokdiff, `results/dryruns/`) |
|---|---|---|---|
| cx90 | `env.cx90-089` | `config.yihou.mc4340n4.sh` (via E2-target) | decode drops the three `SGLANG_SIMULATE_ACC_*` envs (sim OFF) |
| tw90 | `env.tw90-089` | `config.yihou.tokw.sh` (→ `TOKW_BASE_CONFIG=config.yihou.mc4340n4.sh`) | decode `--tokenizer-worker-num 4`, nothing else |
| wts64 | `env.wts48-136` + `CONC=64` | see the 10-14 packup (write_through_selective, prefill mem 0.88) | C64 bench on the existing wts48 stack |

## 2. Launch

```bash
# cx90 (driver ON the node): up -> census -> smoke -> audit -> correctness_tests -> stop
spur exec $J bash -c "cd /; setsid nohup bash $H/chain_cx.yihou.sh $H/env.cx90-089.yihou.sh > $R/cx90.chain.log 2>&1 < /dev/null &"
# tw90 (driver ON the node; it ran after cx90 freed the GPUs): fresh A1 C48 -> flush -> scored A2 C48
spur exec $J bash -c "cd /; setsid nohup bash $H/chain_cap2.yihou.sh $H/env.tw90-089.yihou.sh 48 48 > $R/tw90.chain.log 2>&1 < /dev/null &"
# wts64 (from a login host, 136 = plain ssh): one more bench on the live wts48 stack, C64, tag A2 -> out agentx-c64-A2
bash $H/cont_benchl.yihou.sh $H/env.wts48-136.yihou.sh - 64 > $R/wts48.cont.log 2>&1
```

The correctness client runs as `python3 -u $N/correctness_tests.yihou.py http://<ip>:28100 glm5.2-mxfp4 $N/runs/<round>` (copy in `scripts/session/`).
- Sampled tests use T=1.0, top_p=0.95 (the model's generation config) with the chat template.
- `max_tokens` is 4096.

## 3. Read-out

```bash
cd $R && python3 -B $N/tools_hits.yihou.py "tw90 A2=tw90/agentx-c48-A2/agentx_conc48.json" "wts64=wts48/agentx-c64-A2/agentx_conc64.json"
python3 -B scripts/tools/results_csv_v15.yihou.py followups
zcat results/correctness/cx90/results.jsonl.gz | python3 -c 'import sys,json;[print(d["test"],d["sampling"],d["pass"]) for d in map(json.loads,sys.stdin)]'
```

Expected (single replicates):
- tw90: A1 ≈ 22.2k, A2 ≈ 22.6k.
- wts64: ≈ 8.7k.
- cx90: 22/23.
