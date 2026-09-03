# Reproduce — GLM-5.3-MXFP4 single-node 1P1D

One 8×MI355X node. ~10 min bring-up per arm, ~2–7 min per sweep.

## 0. Prerequisites

- 8 GPUs free on one node; image `infera/engine-sglang:v0518-glm53`.
- Weights at the **resolved** path `/perf_apps/data/models/GLM-5.3-MXFP4`
  (`/apps/data/models` is a symlink onto a different NFS mount — bind the
  realpath or you get an empty dir that fails later as
  `Unrecognized processing class`).
- Repo `AMD-AGI/Infera` at branch `yihou.dev.glm53.expr`, **with the five patches
  in `patches/` applied** — without them the shape does not run. They are
  commits `4493e33`, `f6ee2da`, `1b5ea46`, `667b02f`, `b2b1a08`.
- **Secrets:** SSH to the node; docker registry login only if pulling the image.

## 1. Preflight — do not skip, and do not inherit another node's answers

```bash
# node actually free (NOT --showmemuse's VRAM%, which reads 76% on empty cards)
rocm-smi --showmeminfo vram | grep -E 'GPU\[[0-7]\].*Used Memory'
docker ps
ss -lnt | grep -E ':(2379|2380|8100|8998|12379|30000|30001|5557|5558|8801|8802) '

# this node's own RDMA registration mode + GID index
cd <repo>/examples/sglang_1p1d_glm5.2
IMAGE=infera/engine-sglang:v0518-glm53 bash preflight_rdma.sh mode
```

**Expect mode A** — `peer-mem module: PRESENT`, nothing pinned, KV pool full,
`MC_GID_INDEX=1` on a routable `192.168.x.x` GID (never the link-local `fe80::`).
**Mode B or exit 2 is a stop-and-report**, not a default.

Note preflight's single-node rule, which differs from the two-node one:
*"Single-node loopback: pin ONE device on both legs instead."* Hence
`RDMA_IB_DEVICES=ionic_0`, not all eight rails.

## 2. Bring up

```bash
cd <repo>/examples/sglang_1p1d_glm5.3
MY_IP=<fenic-ip>                              # e.g. 10.235.192.130
CTR=glm53_big_pd \
INFERA_IMAGE=infera/engine-sglang:v0518-glm53 \
IMAGE=infera/engine-sglang:v0518-glm53 \
PREFILL_IP=$MY_IP \
MODEL=/perf_apps/data/models/GLM-5.3-MXFP4 \
MODEL_MOUNT=/perf_apps/data/models \
RDMA_IB_DEVICES=ionic_0 MC_GID_INDEX=1 \
PREFILL_MTP=0 DECODE_MTP=0 \
  bash cluster.singlenode.sh up
```

Per-arm overrides used in this packup:

| arm | extra env |
|---|---|
| PD run 1 / hip-on reference | *(none — decode DPA defaults to 1)* |
| PD decode-DPA-off | `DECODE_DPA=0` |
| hip-off A/B | `MC_DISABLE_HIP=1` |

**Both `INFERA_IMAGE` and `IMAGE` are needed.** `preflight_rdma.sh` reads
`IMAGE`; `engine/up.sh` and `common.sh` require `INFERA_IMAGE`.

## 3. Verify — three checks, and two obvious ones that do NOT work

**GPU split — this is the check.** Both halves must show load:

```bash
rocm-smi --showmeminfo vram | grep -E 'GPU\[[0-7]\].*Used Memory'
# want ~208 GB on 0-3 (prefill, GMU 0.70) and ~249 GB on 4-7 (decode, GMU 0.85)
```

> **`base_gpu_id` does NOT discriminate.** It is an index into each leg's
> *visible* set, so `HIP_VISIBLE_DEVICES=4,5,6,7` renumbers the decode leg to 0-3
> and it reads `base_gpu_id=0` on both legs **whether the split is broken or
> correct**. It read 0/0 when both legs were stacked on GPUs 0-3, and it reads
> 0/0 now that they are not.

**Resolved args per leg — read them off the engine, never off the wrapper:**

```bash
for f in prefill decode; do
  docker exec $CTR bash -c "grep -aoE 'dp_size=[0-9]+|enable_dp_attention=[A-Za-z]+|speculative_algorithm=[A-Za-z]+' /tmp/glm52_\$f.log | sort -u"
done
```

**Liveness — a real generation, never a 200:**

```bash
bash cluster.singlenode.sh smoke      # block 2 must print an actual completion
```

The router answers `/health` and `/v1/models` from its own registry with a dead
engine behind it. `up.sh`'s own wait loop polls `/health` and **will lie to you**.

**For a hip A/B, the discriminator is NOT a log line:**

```bash
docker exec $CTR bash -c 'for p in $(pgrep -f infera.engine.sglang|head -2); do
  tr "\0" "\n" < /proc/$p/environ | grep ^MC_DISABLE_HIP=; done'
```

`HIP transport installed for intra-node GPU P2P` reads **4 per leg in both
states** — it is an install-time log and `MC_DISABLE_HIP` gates *selection*
(`multi_transport.cpp:489`). Verification is env-present + the source read; the
throughput differential is the measurement. See `notes.md` §2.

## 4. Sweep

```bash
CTR=glm53_big_pd HOST=$MY_IP PORT=8100 SERVED=glm-5.3-mxfp4 \
MODEL=/perf_apps/data/models/GLM-5.3-MXFP4 \
ARMS=p50 CONCS="24 16 8 1" \
OUTDIR=<out> CSV=<out>/fixlen.csv bash scripts/big_fixlen.sh
```

**Run concurrencies in decreasing order of what is being argued about**, not by
convention. If the run is cut short — and it was, twice — a truncated sweep then
degrades to the useful subset instead of the useless one.

Repetition arm: `ARMS=p90 CONCS="1"`, then score with `scripts/loopcheck.py`
(vendored here) — **after converting the format**, see `notes.md` §4:

```bash
python3 - <<'PY'
import json
d = json.load(open('<out>/jsonl/big_p90_isl15500_osl3300_c1.jsonl'))
with open('<out>/loopcheck_input.jsonl', 'w') as f:
    for i, (t, o) in enumerate(zip(d['generated_texts'], d['output_lens'])):
        f.write(json.dumps({"request_id": i, "text": t, "completion_tokens": o}) + "\n")
PY
python3 scripts/loopcheck.py <out>/loopcheck_input.jsonl
```

## 5. Archive, then tear down

```bash
docker cp $CTR:/tmp/fixlen/. <dest>/jsonl/
for f in prefill decode; do docker cp $CTR:/tmp/glm52_$f.log <dest>/; done
docker cp $CTR:/tmp/router.log <dest>/

docker inspect $CTR --format '{{.Created}}'   # confirm it is YOURS before removing
docker rm -f $CTR glm52-etcd
rocm-smi --showmeminfo vram | grep -E 'GPU\[[0-7]\].*Used Memory'
```

Archive **first**. Logs have been lost twice on this project by tearing down
before copying. Remove containers **by exact name** — this is a shared node.

## Ports used

prefill 30000, decode 30001, bootstrap 8998, router 8100, etcd 12379,
KV events 5557/8801 (prefill) and 5558/8802 (decode). **Re-check with `ss -lnt`
at the moment of launch** — a foreign etcd held 2379/2380 on one node and not on
another, hours apart.
