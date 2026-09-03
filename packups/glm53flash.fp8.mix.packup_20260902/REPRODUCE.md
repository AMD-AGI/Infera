# Reproduce, cold, from zero

Assumes: an 8×MI355X (gfx950) node, docker, the weights on shared storage, and
SSH access. No secrets beyond those (`environment.md` §secrets).

Total wall time ≈ **35 min** on a warm-base node: build ~8 min, weight load
~11 min (NFS-bound, 62 shards at ~11.3 s each), verification ~2 min, plus a
second round if you want graphs on.

Every script referenced is in `scripts/`. Read `scripts/README.md` first for
what each one is and which parts are node-specific.

---

## 0. Preconditions — check these, they each cost a cycle if wrong

```bash
# (a) the weights, and the symlink trap.
#     /apps/data/models is a symlink to /perf_apps/data/models, and those are
#     SEPARATE NFS mounts. Bind the realpath or the dir is EMPTY in-container.
readlink -f /apps/data/models/GLM-5.3-Flash        # -> /perf_apps/data/models/GLM-5.3-Flash
ls /perf_apps/data/models/GLM-5.3-Flash/*.safetensors | wc -l   # -> 62

# (b) confirm this is the FP8 checkpoint, NOT the MXFP4 one.
python3 -c 'import json;c=json.load(open("/apps/data/models/GLM-5.3-Flash/config.json"));
print(c["model_type"], c["architectures"], c["quantization_config"]["fmt"])'
# -> glm5_next ['Glm5NextForConditionalGeneration'] e4m3

# (c) does this checkpoint need --disable-shared-experts-fusion? Ask the index,
#     do not copy the flag from the MXFP4 recipe. 1:1 pairing => fusion is safe.
python3 - <<'EOF'
import json, collections
d = json.load(open("/apps/data/models/GLM-5.3-Flash/model.safetensors.index.json"))["weight_map"]
print(collections.Counter(k.rsplit(".",1)[-1] for k in d if "shared_experts" in k))
EOF
# -> Counter({'weight': 129, 'weight_scale_inv': 129})   => launch WITHOUT the flag

# (d) ports. Do NOT assume 2379/8100 are free; on n05-29 they were not.
for p in 31400 23795 23796 18105 15570 18801; do
  ss -lnt "sport = :$p" | grep -q LISTEN && echo "$p BUSY" || echo "$p free"; done

# (e) who else is on the box, and how much are they holding?
docker ps --format '{{.Names}}'
rocm-smi --showmeminfo vram | grep -E 'GPU\[[0-7]\].*Used'
```

## 1. Build the image

From the **repo root** (`deploy/docker/Dockerfile.sglang.glm53`, unmodified):

```bash
cd /home/yihou/dev/git.16-19/infera.glm53.series.integration
docker build -f deploy/docker/Dockerfile.sglang.glm53 \
             -t infera/engine-sglang:glm53-c821c425 .
```

Run it in the background (`setsid nohup ... &`) and log to a file — it pulls a
~65 GB base if not already local.

**Verify the overlay by running the image, not by trusting the build log:**

```bash
docker run --rm --entrypoint bash infera/engine-sglang:glm53-c821c425 -c \
  'cd /sgl-workspace/sglang && git rev-parse HEAD && wc -l \
     python/sglang/srt/models/glm5_next.py \
     python/sglang/srt/layers/quantization/quark/quark.py'
```

Expect **exactly**:

```
c821c425c31b0e6c8151324b60fbc2857c39eaef
  1942 python/sglang/srt/models/glm5_next.py
  1172 python/sglang/srt/layers/quantization/quark/quark.py
```

If you see **1834 / 1103** you have the `9e692c92` trap: no AITER mHC (4.3-5.4×
slower, silently) and it cannot load MXFP4. Rebuild.

If the libionic ABI gate hard-fails, rebuild with `--build-arg
LIBIONIC_REQUIRE_ABI=` (empty) and note it — it is cosmetic for single-node MIX.
On this run it passed unaided (`LIBIONIC_ABI ... = 4..4`).

## 2. Bring up MIX, round 0 (decode CUDA graphs OFF)

Graphs off first: fewer unknowns if something is wrong.

```bash
# Run from a /tmp COPY. Editing a script on NFS while bash is reading it
# corrupts the rest of the run -- see notes.md §3(b).
cp scripts/mix_up.sh /tmp/yihou_mix_up.sh
setsid nohup env CUDA_GRAPH=0 bash /tmp/yihou_mix_up.sh > mix_up_r0.log 2>&1 < /dev/null &
```

`scripts/mix_up.sh` will, in order: refuse to touch any container not named
`yihou_f8_*`; wait for **our own** VRAM to drain; **measure free VRAM and compute
`--mem-fraction-static` from it**; start the container binding the model
*realpath*; start etcd; launch the worker; wait on `/health`; start the router.

Expect `chosen --mem-fraction-static = 0.60` on an idle node, and
`worker serving after ~820s`.

**`--mem-fraction-static` is a fraction of TOTAL GPU memory, not free memory.**
The script computes `min(0.60, 0.85 - worst_used_fraction)` across cards 4-7 and
aborts below 0.45. The ceiling is 0.60 rather than the vendor's 0.80 on purpose:
TP4 weights are ~71 GiB/card, so 0.60 (~173 GiB) already leaves ~100 GiB of KV
while leaving 40 % of every card to whoever else is on the box. On a node you
own, raise it.

## 3. Verify — and note that `/health` proves almost nothing

```bash
cp scripts/verify.sh /tmp/yihou_verify.sh
bash /tmp/yihou_verify.sh          # writes verify_<HHMMSS>.txt
```

Nine blocks. The ones that matter, and what a pass looks like:

| block | pass condition |
|---|---|
| A | exactly one worker, `"disagg_mode": "mixed"` |
| B | `/v1/models` → `glm5.3-flash` |
| C | `content` == `'391'`, `reasoning_content` non-empty and separate |
| D | coherent answer, **no repetition loop** |
| **E** | **`4` and `4`** — the two AITER mHC lines, once per rank |
| F | decode-line count with both `full token usage` and `mamba usage` == total decode lines |
| G | `Shared experts fusion optimization enabled.` **present** (FP8 only — on MXFP4 it must be ABSENT) |
| H | fault scan `0`, and the `_dynamo`/`metrics_context` exclusion count stated |

**Block E is the real health check.** Without those 8 lines the server still
starts, still answers correctly, and is 4.3-5.4× slower with nothing in any log
saying so.

Also grep the resolved clamp — **not** `server_args`, which lies about it
(`notes.md` §1):

```bash
docker exec yihou_f8_mix grep -m1 'capped to' /tmp/glm53_f8_mix.log
# -> max_running_requests is capped to 200 by the mamba state cache ...
```

## 4. Round 1 — decode CUDA graphs ON, re-verify

```bash
cp scripts/mix_up.sh /tmp/yihou_mix_up.sh
setsid nohup env CUDA_GRAPH=1 bash /tmp/yihou_mix_up.sh > mix_up_r1.log 2>&1 < /dev/null &
# then re-run scripts/verify.sh
```

Expect `Capture target decode CUDA graph end. elapsed≈33 s, mem usage≈15.4-17.4 GB`
per rank at TP4 — **not** the ~1.4 GB a TP8 run reports. Decode lines should now
read `cuda graph: True`.

## 5. Optional — the HIP IPC probe (no model needed, ~30 s)

Answers whether `hipIpcOpenMemHandle` can import across disjoint
`HIP_VISIBLE_DEVICES`, which decides the single-node PD topology. Full script,
output and caveats: **`results/ipc_probe.md`**. It needs only the image and two
free GPUs, so it can be run standalone.

## 6. Tear down — your own containers only

```bash
docker rm -f yihou_f8_mix yihou_f8_mix_etcd
sleep 15
rocm-smi --showmeminfo vram | grep -E 'GPU\[[4-7]\].*Used'
```

Confirm the cards return to the **neighbour's baseline**, not necessarily to
zero. `docker rm -f` returns well before KFD frees the pages — 100 GiB was still
held ~10 s after teardown returned (`notes.md` §3(c)).

**Never run `reset_gpus.sh` on a shared node.** Never `kill`, `docker stop` or
`docker rm` a container you did not create.

---

## If it fails

| symptom | cause | fix |
|---|---|---|
| `ValueError: ... model type 'glm5_next' but Transformers does not recognize this architecture` | wrong image — the overlay is missing | step 1. Do **not** upgrade transformers; the missing component is sglang |
| model dir empty inside the container | bound the symlink, not the realpath | `readlink -f` at runtime — see step 0(a) |
| `_load_w2` / `_load_w13` "size of tensor a (N) must match tensor b (2N)" | shared-experts fusion on a mixed-precision checkpoint | `DISABLE_SEF=1 bash mix_up.sh`. Should **not** happen on FP8; if it does, re-check step 0(c) |
| etcd exits 1, `--initial-cluster has default=http://localhost:2380` | only one of the three peer flags was overridden | `notes.md` §3(a) |
| `error reading input file: Stale file handle` | the script was edited on NFS mid-run | run from a `/tmp` copy — `notes.md` §3(b) |
| launch sizes itself smaller than expected | measured its own not-yet-freed VRAM | `notes.md` §3(c); the shipped script already waits |
| 0 or 4 mHC lines instead of 8 | `SGLANG_USE_AITER=1` missing, or wrong sglang pin | check the env in `mix_worker.sh` and re-verify step 1 |
