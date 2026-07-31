# Reproduce

Cold start, from nothing to the numbers in `README.md`. Roughly 25 minutes,
most of it model load and tilelang JIT.

Everything here runs from a **login node**. `ssh` to compute nodes is banned on
spur; `spur exec <job> <cmd>` is the only way in.

> **Export `DOCKER_CONFIG=/tmp/dockercfg` before every docker call.** Docker 29's
> buildx plugin discovery fails on the default path. Every command below already
> does this.

---

## 0. Hold two nodes

```bash
sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00   # ×2
squeue -u "$USER"
```

Expect `JobHoldMaxRequeue` bounces; retry or `--exclude` the bad nodes. Health-gate
each one — spur has nodes that enumerate 8 GPUs but report `False`:

```bash
spur exec <job> bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --entrypoint python3 lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x \
  -c "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())"'
```

Must print `True 8`. Then record each node's `ens3` IP — you will substitute
these into `scripts/boot_final.sh` and `scripts/router_final.sh`, which have
this run's IPs hard-coded.

## 1. Build the image — on **each** node

The patches are applied at build time, so this step *is* the patch application.

```bash
# from a checkout of the Infera PR branch, commit 51a7b24
cd <infera-checkout>
tar czf /shared_nfs/yihou.temp/final_build/buildctx.tar.gz deploy/docker/

# then, on EACH of the two nodes:
spur exec <job> bash -c '
export DOCKER_CONFIG=/tmp/dockercfg
rm -rf /tmp/finalbuild && mkdir -p /tmp/finalbuild
tar xzf /shared_nfs/yihou.temp/final_build/buildctx.tar.gz -C /tmp/finalbuild
cd /tmp/finalbuild
setsid nohup docker build -f deploy/docker/Dockerfile.sglang.dmabuf \
  -t infera.yihou.sglang.final:1.0 . \
  > /shared_nfs/yihou.temp/final_build/build_<job>.log 2>&1 < /dev/null &
disown'
```

Build twice rather than `docker save` + copy: a backgrounded long-running docker
client inside `spur exec` is **killed at namespace teardown**, even under
`nohup`/`setsid`. A `docker save` of this image died at ~670 MB of 28 GB this
run. With the base layers cached the build takes ~2 minutes.

> Large artifacts (>500 MB) belong under `/shared_nfs/yihou.temp/`, never
> `/home` — `/home` has filled up and destroyed a 28 GB image tar before.

**The build is self-verifying.** It fails if any patch does not reach the
bytecode. Confirm:

```bash
grep -a "verified in bytecode" /shared_nfs/yihou.temp/final_build/build_<job>.log
# === all sglang DSA patches verified in bytecode ===
```

Then confirm the image really carries the patches:

```bash
spur exec <job> bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
docker run --rm --entrypoint bash infera.yihou.sglang.final:1.0 -c \
  "cd /sgl-workspace/sglang && git status --short --untracked-files=no python/sglang/srt | wc -l"'
# 9
```

## 2. Start the containers

```bash
bash scripts/start_ctr_final.sh <prefill-job>
bash scripts/start_ctr_final.sh <decode-job>
```

Each prints `GPUGATE True 8` and `PATCHED_FILES=9`. Both must be right before
continuing — `PATCHED_FILES=0` means you started the stock image.

## 3. Boot both PD legs

Edit the IPs at the top of `scripts/boot_final.sh` first.

```bash
bash scripts/boot_final.sh prefill
sleep 3
bash scripts/boot_final.sh decode
```

Cold start is ~6–8 minutes each: weights ~2 min, then tilelang JIT and DP CUDA
graph capture. **Be patient** — 8 live `sglang::scheduler_DP*` processes means it
is working, not hung. Wait for both:

```bash
strings /shared_nfs/yihou_exp3way/final/prefill.log | grep -a "server is fired up"
strings /shared_nfs/yihou_exp3way/final/decode.log  | grep -a "server is fired up"
```

> Server logs contain binary bytes. Plain `grep` reports "binary file matches"
> and `grep -c` returns **0**, which reads as "clean". Always use `strings … |
> grep` or `grep -a`.

If the prefill leg dies with `port_base at 30234 is not available`, a previous
server is still holding it. Kill it **by explicit PID** (a broad `pkill -f` can
match your own shell), then re-boot that leg:

```bash
spur exec <job> bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
docker exec <ctr> bash -c "pgrep -f sglang:: | xargs -r kill -9"'
```

All VRAM is released by killing the launcher plus the 8 scheduler PIDs — a
container recreate is *not* needed, and would silently drop anything you patched
in by hand.

## 4. Start the router

```bash
bash scripts/router_final.sh          # port 8170 in this run
```

**Use a fresh `--port` and `--prometheus-port` every restart.** A router whose
circuit breaker is still open returns 503 in ~0.4 s, which looks exactly like a
backend failure. Read the *latency*: ~0.4 s is the breaker, 12–23 s is a real
backend fault.

## 5. Correctness probe

```bash
spur exec <prefill-job> bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
docker exec final python3 /shared_nfs/yihou_exp3way/probe.py http://<PIP>:8170 4 24 180'
```

Expect **4/4 ok** with `acc_len` between 1 and 4. `acc_len > 1` is what proves
MTP is genuinely active rather than silently bypassed — a 4/4 with `acc_len == 1`
is a *failure* dressed as a pass.

## 6. Stress

```bash
for f in "32 stress_c32" "128 stress_c128" "128 stress_c128_r2"; do
  set -- $f
  spur exec <prefill-job> bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec final python3 /shared_nfs/yihou_exp3way/stress.py \
    http://<PIP>:8170 $1 512 /shared_nfs/yihou_exp3way/final/$2.jsonl 600"
done
```

Expect **32/32** and **128/128** twice, all 8 DP ranks serving, 0 retries.
Then confirm neither leg logged an exception:

```bash
strings /shared_nfs/yihou_exp3way/final/decode.log | grep -acE "Traceback|KVTransferError"
# 0
```

## 7. Prove the draft CUDA graph is actually used

**Do not skip this.** Forcing the draft path eager passes steps 5 and 6 while
disabling the feature under test, so everything above is consistent with a
workaround. Only a replay count distinguishes them.

This needs an added probe, so it is a separate run:

```bash
spur exec <decode-job> bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
docker exec final bash -c "cd /sgl-workspace/sglang &&
  python3 <kit>/scripts/instr_graph_usage.py"'
```

Then restart **only the decode leg** (the running server holds the old bytecode),
restart the router on a fresh port, and re-run step 6 at conc=128. Read:

```bash
strings /shared_nfs/yihou_exp3way/final/decode.log | grep -a GLM52_GUSE | tail
```

Expect ~92 % graph usage, **identical on all 8 ranks**. Uniformity across ranks
is the property the patch exists to produce; a per-rank spread would mean it is
not working even if the percentage looks high.

Also check `GLM52_GUSE_WHY` shows `seed_none=0.0%`: the predicate must be reading
`future_dsa_topk_indices_available`, not the stale direct field. A revision that
got this wrong refused the graph 100 % of the time and still passed every
functional test.

## 8. Reverting to stock, for A/B

```bash
docker build -f deploy/docker/Dockerfile.sglang.dmabuf \
  --build-arg APPLY_SGLANG_DSA_PATCHES=0 -t sglang-stock:1.0 .
```

Expect the deadlock to return on the first routed request.
