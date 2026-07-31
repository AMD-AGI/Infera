# Reproduction — GLM-5.2 PD multi-chunk KV corruption

Est. ~20 min bring-up per arm (cold start 6-8 min per PD leg), ~5 min per probe.

## 0. Prerequisites

- `infera/engine-sglang:pd-unified` on **both** nodes. Not on a public registry — stream it:
  `ssh <src> 'docker save infera/engine-sglang:pd-unified' | ssh <dst> docker load` (~10 min for 78 GB).
  `rocm/infera:sglang-v0.1.0-rc6` (public) is only needed for the single-node arm A/B.
- Model `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (shared VAST mount).
- 8 ionic NICs `PORT_ACTIVE` per node; host `libionic` injected into the container (the `up_*`
  script does this).
- Nodes free of other GPU jobs.

Scripts assume a jump host: `ssh root@149.28.124.225` then `ssh <node>`. Adjust `J()` in
`up_dpa_longctx.sh` / `run_router.sh` for your access path.

## 1. Stage scripts to shared VAST

    KIT=/mnt/vast/c_huggingface/glm52_longctx_pd     # visible to both nodes' containers
    for f in scripts/*.py scripts/pd_leg_dpa_longctx.sh; do
      cat "$f" | ssh <jump> "ssh <node> 'cat > $KIT/$(basename $f)'"
    done

## 2. Reproduce the BUG (default chunk)

    # prefill=chi2867, decode=chi2879; DPA=1 -> per-rank chunk 2048
    DPA=1 TAG=warm2 bash scripts/up_dpa_longctx.sh
    # wait for BOTH legs: grep -ac "ready to roll" $KIT/pd_{prefill_30000,decode_30001}_warm2.log -> 1
    bash scripts/run_router.sh                       # router on prefill node :8002

Sanity first (must be 4/4 — short prompts are always fine, they are single-chunk):

    docker exec pd_uni python3 /tmp/probe.py http://10.2.122.44:8002 glm5.2-mxfp4

Confirm RDMA, not TCP:

    grep -ac "MC_FORCE_TCP" $KIT/pd_prefill_30000_warm2.log        # 0
    grep -ac "dmabuf disabled" $KIT/pd_prefill_30000_warm2.log     # 8 (one per NIC)

### 2a. The canary — the clean demonstration

    docker cp $KIT/degrade_test.py pd_uni:/tmp/dg.py
    docker exec pd_uni python3 /tmp/dg.py http://10.2.122.44:8002 glm5.2-mxfp4 10 30303 0 /tmp/canary10.json

Sends the **same 30410-token prompt** 10×. Expect a mix like
`OK, GIB, OK, OK, MISS, OK, GIB, OK, GIB, OK`, with a hard bimodal latency split:
**~2.2 s ⇒ always OK, ~6.4 s ⇒ always corrupt.**

Confirm the two populations in the prefill log — this is the money shot:

    grep -a "Prefill batch" $KIT/pd_prefill_30000_warm2.log | tail -12 \
      | grep -aoE "#new-token: [0-9]+, #cached-token: [0-9]+"
    # fast/correct : #new-token: 64,   #cached-token: 30400   <- radix hit, ONE chunk
    # slow/corrupt : #new-token: 2048, #cached-token: 0  (×15) <- full multi-chunk prefill

### 2b. The boundary tracks the chunk size

    docker cp $KIT/prefix_test.py pd_uni:/tmp/pt.py
    docker exec pd_uni python3 /tmp/pt.py http://10.2.122.44:8002 glm5.2-mxfp4 unique \
      1500,1900,2000,2100,2500,4000 /tmp/subchunk.json
    # pt<=1983 (1 chunk) coherent; pt>=2108 (>=2 chunks) degraded

`unique` gives every request a distinct prefix, so prefix-cache reuse cannot mask the failure.
Use `shared` to see the masking effect.

## 3. Reproduce the MITIGATION (big chunk)

    # 262144 / dp_size 8 = 32768 per rank -> a 30K prompt is ONE chunk
    DPA=1 TAG=bigchunk CHUNK=262144 bash scripts/up_dpa_longctx.sh
    bash scripts/run_router.sh
    grep -ao "chunked_prefill_size=[0-9]*" $KIT/pd_prefill_30000_bigchunk.log | tail -1   # 32768

    docker exec pd_uni python3 /tmp/dg.py http://10.2.122.44:8002 glm5.2-mxfp4 10 30303 0 /tmp/bigchunk_canary.json
    # -> 0 GIBBERISH (was 4/10)

    docker exec pd_uni python3 /tmp/pt.py http://10.2.122.44:8002 glm5.2-mxfp4 unique \
      29292,30606,31515,33636,37373,41414,55555,66666 /tmp/bigchunk_boundary.json
    # <=32599 clean; >=34885 corrupt -> boundary moved to the new chunk size

## 4. Control arms (prove it is PD-specific)

    # same pd-unified image, single node, no disaggregation
    DPA=0 NAME=glm52-single-uni bash scripts/single_unified_coldtest.sh
    DPA=1 NAME=glm52-single-uni-dpa1 bash scripts/single_unified_coldtest.sh
    # then hit novel lengths never used before in that server's lifetime:
    docker exec <name> python3 /tmp/pt.py http://127.0.0.1:30000 glm5.2-mxfp4 unique \
      33131,36767,42323,48989,55151,62626,68181,73737,77373,81818,86464,91919 /tmp/single_hard.json
    # -> 0 corrupt on both arms (18 novel cold shapes total)

Single-node long-context + stress (arms A/B, rc6 image):

    bash scripts/launch_dpa_longctx.sh                                   # DPA=1, ctx 131072
    docker exec <ctr> python3 /tmp/lp.py http://127.0.0.1:30000 glm5.2-mxfp4 65536 /tmp/r.json
    CONC=32 ISL=32768 OSL=256 bash scripts/bench_longctx.sh

## Script reference

| script | purpose |
|---|---|
| `up_dpa_longctx.sh` | bring up both PD legs. Env: `DPA=0\|1`, `CHUNK=`, `TAG=` (per-arm log names) |
| `run_router.sh` | sglang_router mini-LB in the prefill container, port 8002 |
| `pd_leg_dpa_longctx.sh` | the leg launcher itself (from packup exp07, unchanged) |
| `single_unified_coldtest.sh` | control arm: same image, single node, no PD |
| `launch_dpa_longctx.sh` | single-node rc6 long-ctx server (arms A/B) |
| `degrade_test.py` | **canary** — same prompt N rounds, interleaved novel load. The key probe |
| `prefix_test.py` | length sweep with `shared` vs `unique` prefixes |
| `len_sweep.py` | length sweep + gibberish detector (finds the boundary) |
| `repeat_cold.py` | same prompt N× with optional per-request `/flush_cache` |
| `longctx_probe.py` | needle-in-a-haystack, self-calibrates to a target token count |
| `bench_longctx.sh` | `bench_serving` stress (needs `--tokenizer <real model dir>`) |
| `shape_granularity.py` | anchor + neighbour deltas (from the disproved first-touch theory) |
| `prefill_logprob_test.py` | **DO NOT RUN** — kept as a record of the leg-killing trap |

## Pitfalls

- **Never send a request directly to `:30000`/`:30001`** — `req.bootstrap_room should not be None`
  SIGQUITs the whole leg. Router only.
- **Give the probe enough `max_tokens`.** GLM-5.2 thinks before answering; at `max_tokens=96` a
  correct run is often truncated mid-reasoning and scores `MISS`. `MISS` ≠ corruption — only
  `GIBBERISH` (token salad) is the bug. Use ≥512 for a real answer.
- **Warm repeats hide the bug.** A repeated prompt hits the radix cache, becomes single-chunk, and
  always passes. Always probe with `unique` prefixes / fresh lengths.
- **`bench_serving` needs `--tokenizer /mnt/vast/xiaobo/models/GLM-5.2-MXFP4`** — the served model
  name is not a local dir and it will try to reach HuggingFace.
- Cold start is 6-8 min per leg (DPA) and JIT-compiles for several minutes after `ready to roll`;
  slow first requests are normal.
