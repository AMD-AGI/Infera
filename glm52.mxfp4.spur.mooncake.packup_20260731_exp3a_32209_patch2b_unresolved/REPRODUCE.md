# Reproduction kit — Exp 3a / 3c (reproducing a FAILURE)

Goal: reproduce the **conc=32 crash** of #32209's patch-2b port on a cross-node
PD deployment of GLM-5.2-MXFP4 with DP-attention 8 and MTP, and capture the
instrumentation that eliminated seventeen candidate causes.

This kit reproduces a failure, not a fix. Success here means:

```
conc=32 → 0/32,  503 in 12–23 s
ValueError: output tensor size must be equal to world_size times input tensor size
```

Estimated time: **~45 min** with a warm image, **~55 min** if you must rebuild
it (step 0b).

## 0a. Prerequisites

**Machines.** Two nodes, 8 × MI355X:

```bash
sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00   # twice
```

Expect `JobHoldMaxRequeue`; `scontrol release <jobid>` in a loop until RUNNING.
Substitute your job ids and IPs into `scripts/boot.sh` and `scripts/router.sh`
(hard-coded node tables — see `scripts/NODES.md`).

**External paths:** model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.

**Repo state:** `yihou.dev.glm5.2.mxfp4.experiment` @ `0d3e374`.

## 0b. Image — rebuild if absent

The image is a local build and is **not** in any registry. If
`/shared_nfs/yihou_imgbuild/infera.yihou.sglang.1.0.tar` is gone, rebuild it:

```bash
B=/shared_nfs/yihou_imgbuild
mkdir -p $B/deploy/docker/scripts $B/deploy/docker/patches
cp glm52.mxfp4.spur.mooncake.packup_20260728/patches/Dockerfile.sglang.dmabuf.copy \
   $B/Dockerfile.sglang.dmabuf
git show 9cbed1f:deploy/docker/scripts/build_mooncake_dmabuf.sh \
   > $B/deploy/docker/scripts/build_mooncake_dmabuf.sh
cp deploy/docker/scripts/infera_inject_host_ionic.sh $B/deploy/docker/scripts/
cp -r deploy/docker/patches/mooncake_cpp $B/deploy/docker/patches/
chmod +x $B/deploy/docker/scripts/*.sh

export DOCKER_CONFIG=/tmp/dockercfg
spur exec <JOB> bash -c 'export DOCKER_CONFIG=/tmp/dockercfg; cd /shared_nfs/yihou_imgbuild \
  && docker build -f Dockerfile.sglang.dmabuf -t infera.yihou.sglang.1.0:latest . 2>&1 | tail -40'
```

**The build must print** `DMABUF_COMPILED_IN=yes`. Without it, device-memory
registration falls back to bare `ibv_reg_mr`, which EFAULTs here.

Ship it to the second node:

```bash
spur exec <JOB_A> bash -c 'docker save infera.yihou.sglang.1.0:latest \
  -o /shared_nfs/yihou_imgbuild/infera.yihou.sglang.1.0.tar'
spur exec <JOB_B> bash -c 'docker load -i /shared_nfs/yihou_imgbuild/infera.yihou.sglang.1.0.tar'
```

Keep the tar on `/shared_nfs`, not `/home` — a node-local copy does not survive
a NODE_FAIL, which is exactly how it was lost the first time.

## 1. Stage the workspace

```bash
export W=/shared_nfs/yihou_exp3way
mkdir -p $W/common $W/kit_patches $W/e3 $W/e3a $W/e3c
cp scripts/{probe.py,stress.py,pd_leg_exp.sh}                     $W/
cp scripts/{apply_arm.sh,boot.sh,router.sh,start_ctr.sh,NODES.md} $W/common/
cp scripts/instr_*.py scripts/strip_patch2b_v1.py                 $W/e3/
cp patches/*.diff                                                 $W/kit_patches/
cp patches/patch2b_32209_*.py                                     $W/e3/
```

`boot.sh` already points the JIT caches at `$W` — keep them off `/home`.

## 2. Containers

```bash
export DOCKER_CONFIG=/tmp/dockercfg
bash $W/common/start_ctr.sh <JOB_A>
bash $W/common/start_ctr.sh <JOB_B>
```

Both must print `GPUGATE True 8`. A node printing `False` must be abandoned.

## 3. Apply the arm

Two arms are defined. **e3a** is the minimal failing configuration; **e3c**
adds upstream's slice hunk and fails identically.

```bash
for J in <JOB_A> <JOB_B>; do
  spur exec $J bash -c 'docker exec dbg2 bash /shared_nfs/yihou_exp3way/common/apply_arm.sh e3c'
done
```

Must end with `ARM e3c OK`. The arm is:

| # | what | source |
|---|---|---|
| 1 | HIP/aiter padded rows | `dsa_indexer_hip_dp_padded_rows.diff` |
| 2a | dp sync | `dsa_backend_dp_sync_and_page_table_rows.diff` (ours, 2b stripped) |
| **2b** | **trim/restore + output slice** | `patch2b_32209_style.py` + `patch2b_32209_slice.py` |
| 3 | nextn `eh_proj` bf16 | `deepseek_nextn_glm52_mtp_bf16.diff` |
| 4 | draft-graph vote | `eagle_worker_v2_uniform_draft_graph.diff` (ours) |

with anti-markers that must count **0**:

```
WANT=0  dsa_backend.py :: _glm52_match_page_table_rows   (our patch 2b)
WANT=0  dp_attn.py     :: can_draft_cuda_graph           (#32209 patch 4)
```

Verification is against **bytecode**, using identifiers — a stale `.pyc`
silently reverts a patch, and the compiler discards comments so a comment
marker is a guaranteed false negative.

## 4. Instrumentation

Apply on the **decode** leg only (that is where the crash is):

```bash
spur exec <JOB_B> bash -c 'docker exec dbg2 bash -c "
  python3 /shared_nfs/yihou_exp3way/e3/instr_multi.py
  python3 /shared_nfs/yihou_exp3way/e3/instr_hs_origin.py
  python3 /shared_nfs/yihou_exp3way/e3/instr_pad.py"'
```

Then **verify every touched module still compiles** — two of these probes
originally produced a `SyntaxError` by anchoring on a bare `class` line:

```bash
spur exec <JOB_B> bash -c 'docker exec dbg2 bash -c "cd /sgl-workspace/sglang/python/sglang/srt
  for f in speculative/eagle_info.py speculative/eagle_worker_v2.py \
           layers/dp_attention.py layers/attention/dsa_backend.py \
           model_executor/forward_batch_info.py; do
    python3 -m py_compile \$f || echo BAD \$f; done; echo ALL_OK"'
```

## 5. Boot, router, probe

```bash
bash $W/common/boot.sh e3c prefill
bash $W/common/boot.sh e3c decode
for r in prefill decode; do
  echo -n "$r ready="; strings $W/e3c/$r.log | grep -c 'ready to roll'
done
bash $W/common/router.sh e3c && sleep 25
spur exec <JOB_A> bash -c 'docker exec dbg2 python3 /shared_nfs/yihou_exp3way/probe.py \
  http://<PREFILL_IP>:<PORT> 4 24 180'
```

Logs contain binary bytes — plain `grep -c` returns 0 and reads as "clean".
Always `strings` or `grep -a`.

The 4-prompt probe **passes** (4/4). That is expected and is why conc=32 is the
only meaningful test here.

## 6. Reproduce the failure

```bash
spur exec <JOB_A> bash -c 'docker exec dbg2 python3 /shared_nfs/yihou_exp3way/stress.py \
  http://<PREFILL_IP>:<PORT> 32 512 /shared_nfs/yihou_exp3way/e3c/stress_c32.jsonl 900'
```

Expected: `ok: 0/32`, each request 503 after **12–23 s**.

> **Check the 503 latency.** ~0.4 s means a stale router circuit breaker, not a
> backend fault. Restart the router on a **fresh port** and re-run. This
> happened once during these rounds and would otherwise have been recorded as
> a different failure.

## 7. Read the instrumentation

```bash
D=$W/e3c/decode.log
strings $D | grep -c "output tensor size must be equal"      # the crash
strings $D | grep -a "GLM52_MULTI gather" | grep -v "ratio=8.0" | tail
strings $D | grep -a "GLM52_PAD hs_pad"   | tail
strings $D | grep -a "GLM52_HSO draft_entry" | tail
```

The two lines that carry the result — same rank, same iteration:

```
GLM52_MULTI gather seq=18 rank=0 local_rows=6 global_rows=32
  plan=[4,4,4,4,4,4,4,4] orig=[2,1,3,2,2,2,2,4] bs=4 fwd=2 inp_rows=4
GLM52_PAD   hs_pad rank=0 before=2 target=4 bs=4 backup=2 fwd=2
```

Padding targets 4 and is correct; the all-gather receives 6.

Useful aggregate — the faulting condition, and its insufficiency:

```bash
strings $D | grep -a "GLM52_MULTI gather" | python3 -c "
import sys,re,ast
rows=[]
for L in sys.stdin:
    m=re.search(r'rank=(\d+) local_rows=(\d+)',L)
    m2=re.search(r'plan=(\[[^]]*\]) orig=(\[[^]]*\])',L)
    if not(m and m2): continue
    r,lr=int(m.group(1)),int(m.group(2))
    plan=ast.literal_eval(m2.group(1)); orig=ast.literal_eval(m2.group(2))
    rows.append((orig[r],plan[r],lr))
f=[x for x in rows if x[2]!=x[1]]; ok=[x for x in rows if x[2]==x[1]]
print('faults',len(f),sorted(set(f)))
print('1<orig<plan holds for', sum(1 for o,p,l in f if 1<o<p),'/',len(f),'faults')
print('   ...and also for',   sum(1 for o,p,l in ok if 1<o<p),'/',len(ok),'NON-faults')"
```

The last line is the point: the condition is **necessary, not sufficient**.
Do not treat it as a rule.

## 8. Optional — reproduce the e3a variant

`apply_arm.sh e3a` is the same minus `patch2b_32209_slice.py`. It fails the
same way, which is how the slice hunk was ruled out.

## If it does not reproduce

Check, in this order: `ARM ... OK` printed on **both** legs; anti-markers at 0;
all five instrumented modules compile; 503 latency > 10 s; `strings` (not bare
`grep`) used on the logs. See `notes.md` §6 for the traps hit during these
rounds.
