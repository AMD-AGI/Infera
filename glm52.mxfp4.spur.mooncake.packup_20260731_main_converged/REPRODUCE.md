# Reproduce

Cold start, from nothing to the numbers in `README.md`. Roughly 25 minutes, most of it
model load and tilelang JIT.

Everything here runs from a **login node**. `ssh` to compute nodes is banned on spur;
`spur exec <job> <cmd>` is the only way in.

> **Export `DOCKER_CONFIG=/tmp/dockercfg` before every docker call.** Docker 29's buildx
> plugin discovery fails on the default path. Every command below already does this.

---

## 0. Hold two nodes

```bash
sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00   # ×2
squeue -u "$USER"
```

Expect `JobHoldMaxRequeue` bounces; retry or `--exclude` the bad nodes. Health-gate each
one — spur has nodes that enumerate 8 GPUs but report `False`:

```bash
spur exec <job> bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --entrypoint python3 lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x \
  -c "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())"'
```

Must print `True 8`. Then record each node's `ens3` IP — you will substitute these into
`scripts/boot_conv.sh` and `scripts/router_conv.sh`, which have this run's IPs hard-coded.

## 1. Build the image — on **each** node

The patches are applied at build time, so this step *is* the patch application.

```bash
# from a checkout of the Infera PR branch at commit 0d3e34a (rebased onto main 8692fb4)
cd <infera-checkout>
mkdir -p /shared_nfs/yihou.temp/converged_build
tar czf /shared_nfs/yihou.temp/converged_build/buildctx.tar.gz \
    deploy/docker pyproject.toml README.md infera rust

# then, on EACH of the two nodes:
spur exec <job> bash -c '
export DOCKER_CONFIG=/tmp/dockercfg
rm -rf /tmp/convbuild && mkdir -p /tmp/convbuild
tar xzf /shared_nfs/yihou.temp/converged_build/buildctx.tar.gz -C /tmp/convbuild
cd /tmp/convbuild
setsid nohup docker build -f deploy/docker/Dockerfile.sglang \
  -t infera.yihou.sglang.converged:1.0 . \
  > /shared_nfs/yihou.temp/converged_build/build_<job>.log 2>&1 < /dev/null &
disown'
```

The build context needs more than `deploy/docker/` — `Dockerfile.sglang` also copies
`pyproject.toml`, `infera/` and `rust/`.

Build twice rather than `docker save` + copy: a backgrounded long-running docker client
inside `spur exec` is **killed at namespace teardown**, even under `nohup`/`setsid`. With
base layers cached the build takes ~2 minutes.

> Large artifacts (>500 MB) belong under `/shared_nfs/yihou.temp/`, never `/home` —
> `/home` has filled up and destroyed a 28 GB image tar before.

**The build is self-verifying.** It fails if any patch does not reach the bytecode. Confirm
both the prerequisite and our markers, **in this order**:

```bash
grep -a "glm52-nextn\|PREREQ nextn\|verified in bytecode" \
     /shared_nfs/yihou.temp/converged_build/build_<job>.log
# [glm52-nextn] patched /sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py
#   PREREQ nextn eh_proj      -> src=1 (want 1)
# === all sglang DSA patches verified in bytecode ===
```

The ordering is the point: main's idempotent script makes the nextn edit, then our script
asserts it. If that script ever silently skips, the assert fails the build instead of
letting GLM-5.2 die at draft weight-load.

Then confirm the image carries the patches:

```bash
spur exec <job> bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
docker run --rm --entrypoint bash infera.yihou.sglang.converged:1.0 -c \
  "cd /sgl-workspace/sglang && git status --short --untracked-files=no python/sglang/srt | wc -l"'
# 9
```

## 2. Start the containers

```bash
bash scripts/start_ctr_conv.sh <prefill-job>
bash scripts/start_ctr_conv.sh <decode-job>
```

Each prints `GPUGATE True 8` and `PATCHED_FILES=9`. Both must be right before continuing —
`PATCHED_FILES=0` means you started the stock image.

If a previous round's servers are still up, kill them **by explicit PID** first (a broad
`pkill -f` can match your own shell):

```bash
spur exec <job> bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
docker exec <ctr> bash -c "pgrep -f sglang:: | xargs -r kill -9"'
```

Then check what survives with `ps -eo pid,etimes,cmd | grep sglang::`. Entries marked
`<defunct>` are zombies — VRAM is already released and `docker rm -f` reaps them. A raw
`pgrep -c` cannot tell a zombie from a live server.

## 3. Boot both PD legs

Edit the IPs at the top of `scripts/boot_conv.sh` first.

```bash
bash scripts/boot_conv.sh prefill
sleep 5
bash scripts/boot_conv.sh decode
```

Cold start is ~5–10 minutes each: weights ~15 s (shared-storage cache warm), then tilelang
JIT and DP CUDA graph capture. **Be patient** — 8 live `sglang::scheduler_DP*` processes
means it is working, not hung. Wait for both:

```bash
strings /shared_nfs/yihou_exp3way/conv/prefill.log | grep -a "server is fired up"
strings /shared_nfs/yihou_exp3way/conv/decode.log  | grep -a "server is fired up"
```

The decode leg also logs `Capture draft decode CUDA graph end` — that line is the draft
graph being captured, which is what the whole patch set exists to make usable.

> Server logs contain binary bytes. Plain `grep` reports "binary file matches" and
> `grep -c` returns **0**, which reads as "clean". Always use `strings … | grep` or
> `grep -a`.

If the prefill leg dies with `port_base at 30234 is not available`, a previous server is
still holding it — see step 2.

## 4. Start the router

```bash
bash scripts/router_conv.sh          # port 8180 in this run
```

**Use a fresh `--port` and `--prometheus-port` every restart.** A router whose circuit
breaker is still open returns 503 in ~0.4 s, which looks exactly like a backend failure.
Read the *latency*: ~0.4 s is the breaker, 12–23 s is a real backend fault.

## 5. Correctness probe

```bash
spur exec <prefill-job> bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
docker exec conv python3 /shared_nfs/yihou_exp3way/probe.py http://<PIP>:8180 4 24 180'
```

Expect **4/4 ok** with `acc_len` between 1 and 4 (this run: 2.00–3.43). `acc_len > 1` is
what proves MTP is genuinely active rather than silently bypassed — a 4/4 with
`acc_len == 1` is a *failure* dressed as a pass.

## 6. Stress

```bash
for f in "32 stress_c32" "128 stress_c128" "128 stress_c128_r2"; do
  set -- $f
  spur exec <prefill-job> bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec conv python3 /shared_nfs/yihou_exp3way/stress.py \
    http://<PIP>:8180 $1 512 /shared_nfs/yihou_exp3way/conv/$2.jsonl 600"
done
```

Expect **32/32** and **128/128** twice, all 8 DP ranks serving, 0 retries. Then confirm
neither leg logged an exception:

```bash
for r in prefill decode; do
  strings /shared_nfs/yihou_exp3way/conv/$r.log | grep -acE "Traceback|KVTransferError"
done
# 0
# 0
```

## 7. Confirm the nextn fix came from main's script, not from this patch set

This is the specific thing this run exists to verify.

```bash
spur exec <decode-job> bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
docker exec conv bash -c "cd /sgl-workspace/sglang &&
  grep -n \"num_hidden_layers}.eh_proj\" python/sglang/srt/models/deepseek_nextn.py"'
# 363:            ckpt_prefix = f"model.layers.{config.num_hidden_layers}.eh_proj"
```

There must be **no trailing comment**. The diff removed in this convergence appended
`# GLM-5.2: MTP layer bf16; ...` to that line; its absence is what proves the edit came
from `patch_glm52_nextn_quark_exclude.py`.

## 8. What this does NOT reproduce

The **draft-graph replay count** is not measured by any step above. Everything here passes
identically if the draft path is forced eager — which is a workaround, not a fix. Measuring
it requires adding `instr_graph_usage.py` to the image, i.e. building a different image,
and is done in `..._20260731_final_deliverable` (92.0 %, identical on all 8 ranks).

## 9. Reverting to stock, for A/B

```bash
docker build -f deploy/docker/Dockerfile.sglang \
  --build-arg APPLY_SGLANG_DSA_PATCHES=0 -t sglang-stock:1.0 .
```

Expect the deadlock to return on the first routed request. Note this still applies the
nextn prerequisite (it is in the other patch loop), so the failure you see will be the
deadlock, not a weight-load crash.

## 10. The alternative that needs no patches 2b/4

```bash
# on both legs, and MTP must be ON for the PREFILL leg too
--json-model-override-args '{"index_share_for_mtp_iteration":false}'
```

Validated separately in `..._20260730_exp2_indexshare_off` (4/4, 32/32 ×2, 64/64). Patches
1 and 3 are still required. See `notes.md` §2 for the mechanism and the expiry condition.
