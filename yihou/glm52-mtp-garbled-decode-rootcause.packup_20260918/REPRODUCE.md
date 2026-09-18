# Reproduce

Four rounds. R05 is the one that proves the fix; R03 is the failing control you
need in order to believe it. Budget ~15 min per round (engine bring-up ~8 min,
probe ~3 min), so ~1 h for all four.

Everything below runs from the repo's `bench/glm5p2_pd/` directory on a machine
with ssh access to both nodes. The engines run in Docker on the nodes; nothing is
installed on the hosts.

## 0. Preconditions

```bash
cd <repo>/bench/glm5p2_pd
git rev-parse HEAD     # expect 5a342acffe10e09729662ff40e81b22a4367fd75
```

- ssh to `crsuse2-m2m-135` and `crsuse2-m2m-138` works without a password prompt.
- Model weights readable at the same absolute path on both nodes:
  `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.
- Base image `infera-sglang:v0519-yihou-0917` present on **both** nodes. If not,
  build it with the repo's `build_image.sh` from base
  `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917`
  (`sha256:21c1cc9ab...ca32`) — ~40-60 min, and it applies the DSA patch set.
- GPUs **2,3,4,5** idle on both nodes. `check_nodes.sh` will ERROR on 135 GPU[1]
  — that is a root-owned Kubernetes vLLM pod serving Qwen3-32B. Expected. **Do
  not try to clear it, and never relax `GPU_IDLE_VRAM_PCT` to hide it.** See
  `spec/preflight-nodes/README.md` for what a clean preflight looks like.
- No secrets are needed: no registry login (images are local), no API keys, no
  tokens. Access is ssh + local Docker only.

## 1. Apply the `engine.sh` escape hatch

Required by R05 and R06, which pass a one-off engine flag.

```bash
git apply <packup>/patches/0001-engine-sh-extra-args-env.patch
```

Adds role-scoped `{PREFILL,DECODE}_EXTRA_ARGS` / `_EXTRA_ENV`. Empty by default,
so it is a no-op for R03 and R04. See `patches/README.md`.

## 2. Stage the configs

```bash
W=results/yihou-mtp-fusion-fix
mkdir -p $W
cp <packup>/scripts/config.yihou.*.sh $W/
cp <packup>/scripts/topology.yihou.tsv $W/
cp <packup>/scripts/probe.yihou.sh $W/ && chmod +x $W/probe.yihou.sh
```

**The directory must sit exactly two levels below `bench/glm5p2_pd/`** — the path
above satisfies that. `config.yihou.base.sh` reaches the repo baseline with
`$(dirname "${BASH_SOURCE[0]}")/../..`, so placing the configs anywhere else
makes that resolve to the wrong directory and the source fails. The other four
configs resolve each other relative to their own location and are insensitive to
depth; only `base` is not.

`config.yihou.base.sh` is the shared baseline the other four source; it in turn
sources the repo's `config.sh`. **Do not edit `config.sh` or `topology.tsv` —
override with `CONFIG=` / `TOPOLOGY=` / `OUT_DIR=`.**

Confirm the chain resolves and that simulation really is off before launching —
`config.sh` uses `${DECODE_SIMULATE_ACC_LEN-3.61}`, a single dash, so an
empty-but-set value survives:

```bash
bash -c 'source '$W'/config.yihou.nocustomar.sh; \
  echo "IMAGE=$IMAGE SIM=[$DECODE_SIMULATE_ACC_LEN] MTP=$DECODE_MTP \
        EXTRA=[$DECODE_EXTRA_ARGS] GPUS=$DECODE_GPU_DEVICES"'
# expect: IMAGE=...-nextnfix SIM=[] MTP=1 EXTRA=[--disable-custom-all-reduce] GPUS=2,3,4,5
```

## 3. Build the NextN-fusion image (needed for R03, R04, R05)

Locally on **each** node — do not `docker save`/`load`, the base is 66 GB and is
already there.

```bash
mkdir -p /tmp/yihou-nextnfix-build
cp <packup>/scripts/Dockerfile.yihou.nextnfix \
   <packup>/scripts/apply_nextn_fusion_fix.sh /tmp/yihou-nextnfix-build/
cd /tmp/yihou-nextnfix-build
docker build -f Dockerfile.yihou.nextnfix \
  -t infera-sglang:v0519-yihou-0917-nextnfix .
```

Then verify — **with a GPU attached, by importing the class, not by grepping the
source.** Python caches bytecode keyed on mtime, so a source-only edit can be
masked; and the import needs a GPU, which is why it cannot live in the build (see
`notes.md` §2).

```bash
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --entrypoint python3 \
  infera-sglang:v0519-yihou-0917-nextnfix \
  -c "from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN as N, \
      GlmMoeDsaForCausalLM as T; \
      print(N.fused_shared_experts_architecture, T.fused_shared_experts_architecture)"
# expect: GlmMoeDsaForCausalLMNextN GlmMoeDsaForCausalLM
```

The two nodes produce **different image IDs** (independent builds). That is
expected; equivalence is established by this import check on each, not by ID.

## 4. Run a round

Same shape for each; only `CONFIG` and `OUT_DIR` change.

```bash
./launch.sh \
  CONFIG=$W/<config>.sh \
  TOPOLOGY=$W/topology.yihou.tsv \
  OUT_DIR=$W/<round>/launch
```

Bring-up took ~8 min per round here; `READY_TIMEOUT` is 3600 s. Watch
`$W/<round>/launch/server-logs/decode-0.log` rather than idling — CUDA graph
capture is the slow part and can take up to ~30 min on a cold AITER JIT cache.

Then probe:

```bash
DECODE_LOG=$PWD/$W/<round>/launch/server-logs/decode-0.log \
  ./$W/probe.yihou.sh $W/<round>/probe
```

`probe.yihou.sh` defaults to **this experiment's addresses** — router
`http://10.245.148.209:28000` and decode `10.245.157.237:29002`. On other nodes,
override them: `ROUTER=http://<control-ip>:28000 DECODE_HOST=<decode-ip>
DECODE_PORT_BASE=29002 MODEL_NAME=glm5.2-mxfp4 ./probe.yihou.sh <out>`.

And the 16-request rank test that exposes the failure *mode*:

```bash
mkdir -p $W/<round>/rank-test
for i in $(seq 1 16); do
  curl -sS --max-time 120 http://10.245.148.209:28000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"glm5.2-mxfp4","messages":[{"role":"user",
         "content":"What is 2+2? Answer with a single digit."}],
         "temperature":0,"max_tokens":12}' > $W/<round>/rank-test/r$i.json
done
```

Tear down before the next round — each round must start from idle GPUs, and
`launch.sh` refuses a pre-existing `OUT_DIR`:

```bash
./stop.sh CONFIG=$W/<config>.sh TOPOLOGY=$W/topology.yihou.tsv
```

### The four rounds

| round | `<config>` | `<round>` | what it changes vs the previous |
|---|---|---|---|
| R03 | `config.yihou.mtpfix.sh` | `r03-fusion-fix` | fusion fix in, MTP on, real acceptance — **the failing control** |
| R04 | `config.yihou.mtp0.sh` | `r04-no-mtp` | `DECODE_MTP=0` only |
| R05 | `config.yihou.nocustomar.sh` | `r05-nocustomar` | `DECODE_EXTRA_ARGS=--disable-custom-all-reduce` only |
| R06 | `config.yihou.nocustomar.nofix.sh` | `r06-nofix-nocustomar` | `IMAGE` back to the un-patched build only |
| R07 | `config.yihou.noaiterfusion.sh` | `r07-noaiterfusion` | `AITER_ALLREDUCE_FUSION=0` only, against R03 — custom all-reduce back on |

R04 uses the 16-request test only: with MTP off the spec gauges are meaningless,
so `probe.yihou.sh`'s acceptance read has nothing to say.

## 5. What you should see

| round | 16/8 requests | `spec_accept_length` |
|---|---|---|
| R03 | **all garbled**, two modes alternating with period 4 | 1.00 / 1.00 / 2.81 / 1.00 |
| R04 | all coherent | n/a |
| R05 | all coherent, nonce exact | 2.86-3.05, all four ranks |
| R06 | all coherent, nonce exact | 2.55-2.81, all four ranks |
| R07 | **all garbled**, indistinguishable from R03 | 2.51 / 3.65 / 1.00 / 1.00 |

Reference copies of every reply are in `results/<round>/rank-test-replies.txt` —
48 short strings, checkable by eye in a minute.

### Two traps in reading acceptance

1. **The idle-gauge trap.** A DP rank that has served no decode tokens reports
   `spec_accept_length 0.0`, which is indistinguishable from "accepts nothing".
   `probe.yihou.sh` snapshots the gauge before load on purpose, to make the trap
   visible, then generates real traffic and reads again. Only trust ranks whose
   counters moved.
2. **`accept_length` and `accept_rate` are different metrics.** The bar here is
   on **length** (`>= 2.0`). The failing baseline read length 1.25 / rate 0.05;
   quoting one for the other will mislead you by more than an order of magnitude.

## 6. Acceptance

| criterion | R05 | R06 |
|---|---|---|
| no degeneration on the arithmetic probe | PASS | PASS |
| nonce reproduced exactly | PASS | PASS |
| `spec_accept_length >= 2.0` on every active rank | PASS (min 2.8625) | PASS (min 2.5500) |
| 0 router affinity 503s | PASS | PASS |
| 0 Mooncake transfer failures | PASS | PASS |

> **Before copying `--disable-custom-all-reduce` into a performance run:** its
> throughput cost is **unmeasured**. It disables an optimisation. Nothing in the
> sibling packup `glm52-1p1d-samerail-c32-c40.packup_20260918` describes this
> configuration — every number there was taken with custom all-reduce ON and
> simulated acceptance ON. See `notes.md` §7.
