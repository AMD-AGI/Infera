# Exact reproduction — GLM-5.2 DPA + MTP + PD on gfx950, 2026-07-29

Everything needed to rebuild the state that produced `RESULTS_20260729.md`.

---

## 1. Environment (captured from the live container, not from memory)

### Hardware

| | |
|---|---|
| GPU | **AMD Instinct MI355X** (gfx950), 8 per node |
| Nodes used | `crsuse2-m2m-244` (prefill), `crsuse2-m2m-029` (decode) |
| Control pair | `crsuse2-m2m-215` (prefill), `crsuse2-m2m-046` (decode) |
| Scheduler | **Spur** (not stock Slurm) |
| KV NIC | **mlx5_0**, fw `28.43.3608`, link_layer Ethernet (RoCEv2) |
| Also present | `ionic_0`, fw `1.117.1-a-63` — **not used**, lacks ODP |
| GPUDirect | **dma-buf** via mlx5 (spur has no peermem) |
| GID index | **3** |

### Software

| | |
|---|---|
| Image | `infera.yihou.sglang.1.0` |
| Image ID | `sha256:347bcd45da0dee1bc87f10c348e41f20ed56e11d23f9fead164cdef4e51dc970` |
| Image tar | `/home/yihou/infera.yihou.sglang.1.0.tar` (28 GB, on shared `/home`) |
| Base | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`, mooncake rebuilt `USE_HIP_DMABUF` |
| sglang version | `0.5.15.post1` |
| sglang commit | **`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`** (editable checkout at `/sgl-workspace/sglang`) |
| torch | `2.9.1+rocm7.2.0.git7e1940d4` |
| ROCm/HIP | `7.2.26015-fc0010cf6a` |
| `PYTORCH_ROCM_ARCH` | `gfx942;gfx950` |
| Model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` |

There is no separate site-packages copy of sglang — `python/sglang/...` in the
checkout is what actually runs, so patching files there takes effect on restart.

### Absolute paths this work depends on

| path | what | committed? |
|---|---|---|
| `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` | model weights | no — cluster NFS |
| `/home/yihou/infera.yihou.sglang.1.0.tar` | image tarball | no — 28 GB |
| `/home/yihou/glm52_fix/` | live scratch workspace | copied into `patches/scripts_20260729/` |
| `/sgl-workspace/sglang` | in-container editable checkout | in image |

`/home/yihou` and `/shared_nfs` are both bind-mounted into the container, which
is how the patch scripts are reachable from inside.

### Credentials

**None required** for the reproduction itself — no registry pull (the image is
loaded from a local tar), no HF token (weights are already on NFS), no API keys.
Cluster access is the user's existing spur/Slurm account.

---

## 2. Allocate and prepare nodes

```bash
# two 1-node jobs, one per PD leg
sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00 --wrap "sleep infinity"
sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00 --wrap "sleep infinity"
```

Expect `JobHoldMaxRequeue` / `NODE_FAIL` bounces; retry and `--exclude` bad nodes.

**Health-gate every fresh node** — spur has nodes that enumerate 8 GPUs but
report `torch.cuda.is_available() == False`:

```bash
export DOCKER_CONFIG=/tmp/dockercfg
spur exec <JOB> bash -c "docker run --rm --device /dev/kfd --device /dev/dri \
  infera.yihou.sglang.1.0 python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'"
```

Must print `True 8`. Otherwise abandon the node.

Load the image (several minutes):

```bash
spur exec <JOB> bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker load -i /home/yihou/infera.yihou.sglang.1.0.tar"
```

Start the container, **detached**, named `dbg2`:

```bash
spur exec <JOB> bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker run -d --name dbg2 \
  --network host --ipc host --privileged \
  --device /dev/kfd --device /dev/dri \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 64g \
  -v /shared_nfs:/shared_nfs -v /home/yihou:/home/yihou \
  infera.yihou.sglang.1.0 sleep infinity"
```

Get each node's **ens3** (mlx5) IP — not the hostname, not another NIC:

```bash
spur exec <JOB> bash -c "docker exec dbg2 ip -4 addr show ens3"
```

---

## 3. Apply the patches

All scripts live in `patches/scripts_20260729/`; copy that directory to
`/home/yihou/glm52_fix/` (it is bind-mounted, so the container sees it).

Apply **in this order**, on the container of **each** leg unless noted:

```bash
R() { spur exec "$1" bash -c "docker exec dbg2 bash -c 'python3 /home/yihou/glm52_fix/$2'"; }

# 1. Bug 1  — dsa_indexer HIP DP rows            (both legs)
R <JOB> apply_fix.py

# 2. Bug 2  — dsa_backend rank-divergent syncs   (both legs)
R <JOB> fix_a_a2.py

# 3. Bug 5  — page_table rows vs topk rows       (both legs)
R <JOB> fix_bug5_page_table_rows.py

# 4. Bug 6  — Bug-1 slice must fire at q_offset==0 (both legs)
R <JOB> fix_bug6_idle_qoffset.py

# 5. Variant B — draft graph forced eager        (DECODE leg only)
R <DECODE_JOB> variant_b_no_draft_graph.py
```

Plus the **nextn / MTP bf16** patch on both legs — without it the weight load
dies with a `3072 vs 6144` shape error:

```bash
spur exec <JOB> bash -c "docker exec dbg2 bash -c 'cd /sgl-workspace/sglang && \
  patch -p1 < /home/yihou/glm52_fix/deepseek_nextn_glm52_mtp.patch'"
```

It changes one line in `python/sglang/srt/models/deepseek_nextn.py`:

```python
- ckpt_prefix = f"model.layers.{config.num_hidden_layers}"
+ ckpt_prefix = f"model.layers.{config.num_hidden_layers}.eh_proj"
```

### Verify before booting

```bash
spur exec <JOB> bash -c "docker exec dbg2 bash -c '
  cd /sgl-workspace/sglang && git status --short python/sglang/srt/
  printf \"Bug6=%s Bug5=%s FixA=%s VariantB=%s nextn=%s\n\" \
    \$(grep -c GLM52_BUG6 python/sglang/srt/layers/attention/dsa/dsa_indexer.py) \
    \$(grep -c _glm52_match_page_table_rows python/sglang/srt/layers/attention/dsa_backend.py) \
    \$(grep -c \"req_to_token.shape\[1\]\" python/sglang/srt/layers/attention/dsa_backend.py) \
    \$(grep -c GLM52_VARIANT_B python/sglang/srt/speculative/eagle_worker_v2.py) \
    \$(grep -c eh_proj python/sglang/srt/models/deepseek_nextn.py)'"
```

Expected on the decode leg: `Bug6=1 Bug5=3 FixA=5 VariantB=1 nextn=7`.

**Do NOT apply** `fix_bug3_broadcast.py` (proven no-op) or
`fix_bug4_uniform_event.py` (**actively harmful** — causes its own deadlock).

`patches/ALL_FIXES_20260729.patch` is the combined unified diff of the four
modified source files, taken live from the running container; the per-file
splits are `patches/fix_*_20260729.patch`. Either the scripts or the diffs can
be used, but the scripts are preferred: they handle the stale-`.pyc` trap and
self-verify by import.

---

## 4. Launch

Script: `patches/scripts_20260729/pd_leg_spur.sh` → `/home/yihou/glm52_fix/`.
It honours `EXTRA_ARGS`, appended verbatim to the sglang command line.

```bash
P_IP=<prefill ens3 IP>
D_IP=<decode  ens3 IP>

# prefill leg (MTP off)
spur exec <PREFILL_JOB> bash -c "docker exec -d dbg2 bash -c \
  'ROLE=prefill MY_IP=$P_IP P_IP=$P_IP DPA=1 MTP=0 bash /home/yihou/glm52_fix/pd_leg_spur.sh'"

# decode leg (MTP on)
spur exec <DECODE_JOB> bash -c "docker exec -d dbg2 bash -c \
  'ROLE=decode MY_IP=$D_IP P_IP=$P_IP DPA=1 MTP=1 bash /home/yihou/glm52_fix/pd_leg_spur.sh'"
```

Logs: `/tmp/pd_prefill_30000.log`, `/tmp/pd_decode_30000.log`.

Effective server args (from the script): TP8, `--dp-size 8 --enable-dp-attention
--ep-size 8`, `--speculative-algorithm EAGLE --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4`,
`--disable-custom-all-reduce`, `--kv-cache-dtype fp8_e4m3`,
`--mem-fraction-static 0.85` (decode) / `0.88` (prefill),
`--context-length 32768`, `--chunked-prefill-size 65536`,
`--cuda-graph-max-bs 128`, `--nsa-{prefill,decode}-backend tilelang`,
`--disaggregation-transfer-backend mooncake --disaggregation-ib-device mlx5_0`.

Key env the script sets: `MC_GID_INDEX=3`, `MC_DISABLE_HIP_TRANSPORT=1`,
`MC_MS_FILTERS=mlx5_0`, `MOONCAKE_DISABLE_HIP_DMABUF=0`, `NCCL_IB_DISABLE=1`,
`SGLANG_USE_AITER=1`, `SGLANG_DP_USE_GATHERV=1`.

Boot takes **8–10 min**. 8 live `sglang::scheduler_DP*` processes means it is
still working, not hung. Wait for
`The server is fired up and ready to roll!` — use `strings`, the logs contain
binary bytes.

### Router (on the prefill node)

```bash
spur exec <PREFILL_JOB> bash -c "docker exec -d dbg2 bash -c \
  'python3 -m sglang_router.launch_router --pd-disaggregation \
     --prefill http://$P_IP:30000 8998 --decode http://$D_IP:30000 \
     --host 0.0.0.0 --port 8031 --prometheus-port 29131 > /tmp/router.log 2>&1'"
```

Use a **fresh** `--port` / `--prometheus-port` on every restart. If it returns
503 `No available decode workers (all circuits open or unhealthy)`, its circuit
breaker tripped while decode was down — kill it by explicit PID and start a new
one on new ports.

---

## 5. Run the tests

```bash
# functional: 4 sequential 24-token requests
for i in 1 2 3 4; do
  curl -s -m 120 -X POST http://$P_IP:8031/generate \
    -H "Content-Type: application/json" \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":24,"temperature":0}}' \
    -o /tmp/r$i.json -w "req$i http=%{http_code} t=%{time_total}\n"
done
```

Stress — 5 rounds of conc=128 × 512 tokens, with per-round quality and health:

```bash
bash /home/yihou/glm52_fix/stress_rounds.sh 5 128 512
```

(needs `qcheck2.py` alongside it; both are in `patches/scripts_20260729/`.)

Differential harness against the eager control pair:

```bash
bash /home/yihou/glm52_fix/xcheck.sh health
bash /home/yihou/glm52_fix/xcheck.sh probe 4
bash /home/yihou/glm52_fix/xcheck.sh stacks graph   # py-spy every DP rank
```

Edit the job IDs and IPs at the top of `xcheck.sh` to match your allocation.

### Expected

* `128×200` per round, 0 exceptions, 0 KVTransferError, 8 schedulers alive
* `mean spec_accept_length` ≈ 2.7, `dp_rank` spread across 0–7
* **~2 % degenerate outputs** (`1.1.1.1...`) — known open issue, see
  `RESULTS_20260729.md` §5

### Optional instrumentation

`probe_stage_count.py` (per-rank LM-head all-gather counter + MTP stage tags) and
`probe_graph_path.py` (logs the graph/eager decision and its inputs). Both write
`/tmp/stage_probe_dp{0..7}.log`, diffable rank-against-rank. Apply
`probe_stage_count.py` first — `probe_graph_path.py` depends on its `_sp_log`.

Remember: **a replayed CUDA graph executes no Python**, so these probes are blind
inside a graph.

---

## 6. Teardown / restart

To restart the server without losing in-container patches — **never recreate the
container**, that silently drops them:

```bash
spur exec <JOB> bash -c "docker exec dbg2 bash -c '
  pkill -9 -f sglang.launch_server; sleep 5;
  for p in \$(ps aux | grep \"[s]glang::scheduler\" | awk \"{print \\\$2}\"); do kill -9 \$p; done'"
# verify VRAM released -- should print 8
spur exec <JOB> bash -c "rocm-smi --showmemuse | grep -c 'VRAM%): 0'"
```

Killing the launcher plus the 8 scheduler PIDs releases **all** VRAM (86 % → 0 %
on every GPU), so a recreate is never needed.

Kill by explicit PID. A broad `pkill -f <pattern>` can match your own shell.
