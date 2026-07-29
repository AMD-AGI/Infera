# Reproduce — Bug 2b deadlock and its fix

Two nodes, one prefill leg + one decode leg, mooncake over mlx5_0 + dmabuf + GID 3.
Every command below was actually run on 2026-07-29; nothing here is reconstructed.

**No credentials are required.** Model and image are on shared NFS / already loaded.

---

## 0. Prerequisites

```bash
export DOCKER_CONFIG=/tmp/dockercfg        # before EVERY docker call (docker 29 buildx)
```

* Two spur nodes with 8× MI355X each, held via
  `sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00`.
  Health-gate each: `torch.cuda.is_available()` must be True.
* Container `dbg2` running image `infera.yihou.sglang.1.0`, with `/home/yihou` and
  `/shared_nfs` bind-mounted.
* **ssh to compute nodes is banned** — use `spur exec <job> <cmd>`.
* **Never background a long client inside `spur exec`** — use `docker exec -d`.

Absolute paths this kit depends on:

| path | what |
|---|---|
| `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` | model (408 GB) |
| `/home/yihou/glm52_fix/` | working dir, bind-mounted into the container |
| `/home/yihou/glm52_fix/{inductor_cache,triton_cache}` | persistent JIT caches (halve boot time) |
| `/sgl-workspace/sglang` | editable sglang install inside the image |

## 1. Apply the prerequisite patches (decode leg)

Order matters, and the nextn patch is not optional — without it the server dies at weight
load with `size of tensor a (3072) must match ... b (6144)`.

```bash
docker exec dbg2 bash -c '
cd /home/yihou/glm52_fix
python3 apply_fix.py                      # Bug 1: HIP padded-vs-real rows
python3 fix_bug5_page_table_rows.py       # Bug 5: page_table vs topk_indices rows
python3 fix_bug6_idle_qoffset.py          # Bug 6: q_offset == 0 on an idle rank
cd /sgl-workspace/sglang
git apply /home/yihou/glm52_fix/deepseek_nextn_glm52_mtp.patch
'
```

Verify the nextn patch by the **exact** post-patch text, not a substring (see PITFALLS P2):

```bash
grep -c 'num_hidden_layers}.eh_proj' python/sglang/srt/models/deepseek_nextn.py   # want 1
```

## 2. Reproduce the deadlock (control arm)

Apply the instrumentation only — **not** the fix:

```bash
docker exec dbg2 python3 /home/yihou/glm52_fix/bug2b/r01_instrument/probe_guard.py
docker exec dbg2 bash /home/yihou/glm52_fix/bug2b/verify_pyc.sh \
    /sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py GLM52_R1
# want: pyc count 3   (a stale .pyc silently reverts the patch -- this check is mandatory)
```

Launch prefill (node A) and decode (node B):

```bash
# node A — prefill
docker exec -d dbg2 bash -c '
export MY_IP=<A_IP> P_IP=<A_IP> ROLE=prefill PORT=30000 DPA=1 MTP=0
export LOG=/home/yihou/glm52_fix/bug2b/r01_instrument/prefill.log
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/home/yihou/glm52_fix/inductor_cache
export TRITON_CACHE_DIR=/home/yihou/glm52_fix/triton_cache
bash /home/yihou/glm52_fix/pd_leg_spur.sh'

# node B — decode, MTP on, probe on
docker exec -d dbg2 bash -c '
export MY_IP=<B_IP> P_IP=<A_IP> ROLE=decode PORT=30001 DPA=1 MTP=1
export GLM52_R1_PROBE=1
export LOG=/home/yihou/glm52_fix/bug2b/r01_instrument/decode.log
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/home/yihou/glm52_fix/inductor_cache
export TRITON_CACHE_DIR=/home/yihou/glm52_fix/triton_cache
bash /home/yihou/glm52_fix/pd_leg_spur.sh'
```

Cold boot is ~7–9 min. Both must reach `health=200`; decode must log `ready to roll`.
**Warmup passing proves nothing** — it loads all 8 ranks, which is the case that works.

Router (fresh port every restart, `docker exec -d`):

```bash
docker exec -d dbg2 bash -c '
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://<A_IP>:30000 8998 --decode http://<B_IP>:30001 \
  --host 0.0.0.0 --port 8100 --prometheus-port 29100 > /tmp/router.log 2>&1'
```

Trigger — a handful of *sequential* requests, which produce the partial occupancy the bug
needs:

```bash
docker exec dbg2 python3 /home/yihou/glm52_fix/bug2b/probe.py http://<A_IP>:8100 4 24 120
```

Expected: `0/4`, first request timing out at ~120 s. Confirm it is a deadlock, not slowness
— sample twice ~8 s apart and require identical output:

```bash
docker exec dbg2 bash /home/yihou/glm52_fix/bug2b/stacks.sh; sleep 8
docker exec dbg2 bash /home/yihou/glm52_fix/bug2b/stacks.sh
```

Expected shape: exactly one rank in `init_forward_metadata (dsa_backend.py:785)`, the rest
split across `all_gather_into_tensor` and `broadcast`.

Now read the decision records:

```bash
docker exec dbg2 python3 /home/yihou/glm52_fix/bug2b/r01_instrument/analyze.py \
    /home/yihou/glm52_fix/bug2b/r01_instrument/decode.log
```

Expected: `final` diverges on **exactly one** iteration — the last one — where a single
busy rank went eager and the idle ranks went graph. That rank is the one in the py-spy
dump. (The victim rank varies between runs: dp2 and dp3 were both observed.)

## 3. Apply the fix

```bash
docker exec dbg2 python3 /home/yihou/glm52_fix/bug2b/r02_fix/fix_uniform_draft_graph.py
# verify with an identifier, NOT the comment marker (PITFALLS P6):
docker exec dbg2 bash -c '
cd /sgl-workspace/sglang/python/sglang/srt/speculative
rm -f __pycache__/eagle_worker_v2.*.pyc
python3 -c "import sglang.srt.speculative.eagle_worker_v2" >/dev/null 2>&1
strings __pycache__/eagle_worker_v2.*.pyc | grep -c _needs_eager_local'   # want 1
```

Restart **only** the decode leg (kill by explicit PID; a broad `pkill -f` matches your own
shell), and confirm VRAM is fully released before relaunching:

```bash
docker exec dbg2 bash -c '
PIDS=$(ps -eo pid,args --no-headers | grep -E "launch_server|sglang::scheduler" \
       | grep -v grep | awk "{print \$1}")
[ -n "$PIDS" ] && kill -9 $PIDS; sleep 10
rocm-smi --showmemuse | grep -c "VRAM%): 0"'      # want 8
```

Relaunch decode as in step 2, **restart the router on a fresh port** (its circuit breaker
is still open from the hang — see PITFALLS P4), then:

```bash
docker exec dbg2 python3 /home/yihou/glm52_fix/bug2b/probe.py  http://<A_IP>:8101 4 24 120
docker exec dbg2 python3 /home/yihou/glm52_fix/bug2b/probe.py  http://<A_IP>:8101 1 512 240
for c in 1 2 4 8 16 64 128 256; do
  docker exec dbg2 python3 /home/yihou/glm52_fix/bug2b/stress.py \
      http://<A_IP>:8101 $c 512 /tmp/stress_c$c.jsonl 700
done
```

Expected: all pass, `spec_accept_length` ≈ 2–3 throughout (proving MTP is active and not
silently bypassed).

## 4. Verify the *mechanism*, not just the outcome

Passing is not sufficient — Variant B also passed by disabling the graph. Add the vote
probe and confirm the graph is still used:

```bash
docker exec dbg2 python3 /home/yihou/glm52_fix/bug2b/r03_verify/probe_voted.py
# relaunch decode with GLM52_VOTE_PROBE=1, run the sweep, then:
docker exec dbg2 python3 /home/yihou/glm52_fix/bug2b/r03_verify/analyze_vote.py \
    /home/yihou/glm52_fix/bug2b/r03_verify/decode.log
```

Expected:

```
(A) iterations where LOCAL diverges  : > 0     (the bug is still latent -- good)
(A) iterations where VOTED diverges  : 0       (mandatory)
(B) all-ranks-GRAPH iterations       : ~98%    (mandatory -- else it is Variant B)
vote changed a rank's decision       : > 0     (each = one averted deadlock)
```

## 5. Differential control — do not skip

Revert **only** the fix, keep everything else identical, and confirm the hang returns:

```bash
docker exec dbg2 python3 /home/yihou/glm52_fix/bug2b/r03_verify/probe_voted.py --revert
docker exec dbg2 python3 /home/yihou/glm52_fix/bug2b/r02_fix/fix_uniform_draft_graph.py --revert
# verify 0 in bytecode, relaunch decode, fresh router, rerun step 2's probe
```

Observed: `0/4` again, within minutes, on the same node — with the victim rank changing
between runs, which is the race behaving as a race.

## 6. Generate the shippable patch

```bash
docker exec dbg2 bash -c '
mkdir -p /tmp/patchgen && cd /sgl-workspace/sglang
git show HEAD:python/sglang/srt/speculative/eagle_worker_v2.py \
    > /tmp/patchgen/eagle_worker_v2.py.pristine'
docker exec dbg2 python3 /home/yihou/glm52_fix/bug2b/gen_patch.py
# -> /home/yihou/glm52_fix/bug2b/bug2b_uniform_draft_graph.patch
```

The generator applies the fix to a **pristine** copy, so the instrumentation can never
leak into the patch.
