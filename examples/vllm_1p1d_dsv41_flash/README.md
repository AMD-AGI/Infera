# DeepSeek-V4.1-Flash (MXFP4) — vLLM 1P1D on MI355X, served the infera way

One **prefill** and one **decode** vLLM engine, each DP2 × TP1, KV moved between
them by **Mooncake over RDMA**, fronted by the **infera router**, with
**DSpark** speculative decoding and a **UMBP** host-DRAM KV tier on the prefill.
The engine recipe is InferenceX's top MI355X vLLM point for DeepSeek-V4.1-Flash
(`dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml`, `override_dp2_umbp_c384`:
427.5k tok/s/GPU at concurrency 384, one DP2 engine on two GPUs) split into the
two legs. Both legs can share one node; KV still crosses the NICs.

```bash
bash build_image.sh                               # engine image from this checkout
IMAGE=<tag> bash preflight_rdma.sh                # which wrapper, which NICs
bash cluster/cluster.gem.sh up                    # (or cluster.peermem.sh)
bash cluster/cluster.gem.sh smoke                 # every feature, with evidence
bash cluster/cluster.gem.sh accuracy              # correctness gate
bash cluster/cluster.gem.sh golden                # decode -> InferenceX scoring mode
bash cluster/cluster.gem.sh agentx 384 768        # formal AgentX points
bash cluster/cluster.gem.sh down
```

## Contents

| path | what |
|---|---|
| [`cluster/`](cluster/README.md) | **the only files you edit**: one wrapper per RDMA registration mode |
| `build_image.sh` | builds `deploy/docker/Dockerfile.vllm.dsv41` (vLLM ROCm nightly + patches + GEM allocator + infera + Rust router) |
| `preflight_rdma.sh` | per node: amdgpu version, peer memory, active NICs and GIDs, RDMA visible in the image → wrapper to use |
| `common.sh` | container, etcd, UMBP server, router, health polling, leg teardown |
| `engine/leg.sh` | one leg; **the recipe lives here**, no site values |
| `engine/up.sh` / `down.sh` | bring-up (containers → etcd → UMBP → both legs → router) / teardown |
| `engine/smoke.sh` | service check plus a signal per feature that goes red when the feature is silently absent |
| `engine/accuracy.sh`, `engine/probes/` | GSM8K (200) and needle retrieval at 16k / 64k / 128k, with a verdict |
| `engine/golden.sh` | restart decode with synthetic acceptance (golden 3.51) or real verification |
| `engine/agentx.sh` | one InferenceX AgentX point through InferenceX's own client |
| [`PATCHES.md`](PATCHES.md) | every patch the kit needs: symptom, root cause, fix, when to drop it |

## Topology

```
node (or two nodes)
├─ etcd + infera router :8100      kv-aware, prefill session affinity, serial Mooncake dispatch
├─ UMBP standalone server          449 GB host-DRAM KV store, shared by both prefill ranks
├─ prefill  DP2 x TP1  GPUs 0,1    :30000  ══ Mooncake over RDMA ══▶  decode  DP2 x TP1  GPUs 4,5  :30001
│   prefix cache + UMBP                                                no prefix cache, DSpark
```

Legs on one node take GPUs on different sockets: each leg's Engram tables live
in host memory, and each GPU uses the NIC on its own PCIe switch.

## Prerequisites

- 8× MI355X (gfx950) per node; RoCE NICs (validated on AUS: 8× ionic 400G).
- DeepSeek-V4.1-Flash weights on every node (~476 GB).
- Docker with GPU and RDMA access, the user able to run `docker` on the nodes.
- For AgentX: a SemiAnalysisAI/InferenceX checkout and a Hugging Face cache that
  holds (or can download) `semianalysisai/cc-traces-weka-062126`.

## Image

`bash build_image.sh` builds from the vLLM ROCm nightly pinned in
`deploy/docker/Dockerfile.vllm.dsv41` (`nightly-9c1d45c2`, vLLM 0.34.0rc1) and
keeps its aiter / Mooncake / MORI (the DeepSeek-V4.1 kernels and UMBP only exist
there). It adds the vLLM patches (a moved anchor fails the build), the GEM KV
allocator, infera from this tree and the Rust router (toolchain copied from the
`rust` image, never left in the result), and runs a self-check — again with GPUs
after the build. Copy the image to other nodes with `docker save | docker load`.

## Recipe

Shared by both legs, verbatim from the c384 point: DP2 × TP1 with the MoE
tensor-parallel across the ranks (no EP), `--kv-cache-dtype fp8_e4m3`,
`--block-size 128`, DSpark K=5 (probabilistic drafting), the MXFP4
sparse-logits indexer, `--moe-backend aiter`, Engram in host memory,
`--max-num-batched-tokens 16384`, CUDA graphs up to 1024, no `--monokernel`,
`--gpu-memory-utilization 0.9`, `--max-model-len 1048576`, and the c384 AITER /
breakable-graph / Rust-frontend environment.

What P/D changes, and why:

| setting | c384 (aggregated) | prefill | decode | why |
|---|---|---|---|---|
| KV transfer | — | Mooncake producer | Mooncake consumer | RDMA, `MC_DISABLE_HIP=1` even on one node |
| `canonical_kv` | — | on | on | keeps the AITER sparse MLA kernel `fp8_e4m3` needs (PATCHES #8) |
| UMBP | 449 GB, standalone | 449 GB, standalone | — | prefix reuse lives on the prefill |
| prefix cache | on | on | **off** | every turn's KV arrives from prefill |
| `prefill-schedule-interval` | 11 | unset | 11 | on a prefill-only engine it idles N−1 of N steps |
| DSpark verification | golden 3.51 | real (block) | block for correctness, golden 3.51 for scoring | InferenceX's AgentX contract |
| routing | vllm-router `consistent_hash` | kv-aware + session affinity, per DP rank | vLLM's DP balancer (`INFERA_VLLM_DP_ENGINE_LB=1`) | |
| dispatch | — | | | serial: decode leg posted after the prefill reply (PATCHES #11) |

On amdgpu ≥ 6.19 (or without a peer-memory module) both legs allocate the KV
cache from GEM BOs (`cluster.gem.sh`, PATCHES #5); the UMBP server maps the
prefill's through the GEM allocator's HIP IPC (PATCHES #10).

## Verify

`smoke` reads, from the processes and logs themselves: one prefill + one decode
in the router; a correct non-streaming and streaming answer; RDMA bytes on the
NICs during a 16k-token request; no decode pull timing out; each leg running DP2
with `fp8_e4m3`; the KV cache in GEM BOs (or not) as configured; UMBP wired;
DSpark acceptance; the router's tokenizer and KV-event feed.

`accuracy` gates every bring-up before an AgentX run (a single failed root
request aborts a formal run): GSM8K ≥ 0.95 over 200, needles 32/32 at 16k, 64k
and 128k tokens, no errors or garbled output, no decode pull timeouts. It
refuses to run while decode is on golden acceptance, which commits draft tokens
unverified and would score near zero.

## Results (AUS MI355X, amdgpu 6.19, single runs)

| point | GPUs | total tok/s/GPU | interactivity p50 | TTFT p50 / p90 | notes |
|---|---|---|---|---|---|
| InferenceX published, aggregated DP2 + UMBP, c384 | 2 | 427,794 | 20.3 | 1.59 / 3.69 s | 2026-11-21 curve, same nightly-9c1d45c2 |
| aggregated DP2 + UMBP, c384 (same nightly, InferenceX recipe verbatim) | 2 | **418,000** | 19.6 | 1.64 / 3.84 s | |
| 1P1D, c384 | 4 | **374,108** | 53.3 | 1.56 / 4.60 s | prefill KFD memory at 0.82 (pre-#10), concurrent dispatch (pre-#11), kv-aware blind (pre-#12); prefix hit 98.2% |
| 1P1D, c640 | 4 | **471,698** | 34.4 | 5.20 / 15.88 s | same configuration as c768; prefix hit 98.0%; 54,356 measured requests, 0 errors |
| 1P1D, c768 | 4 | **459,183** | 32.8 | 16.96 / 31.75 s | everything in PATCHES #1–#13; prefix hit 97.9%; 53,690 measured requests, 0 errors |
| 1P1D, c892 | 4 | 400,567 | 36.6 | 42.89 / 73.69 s | same configuration; prefill GPU prefix hit falls to 22% (59% at c768) |

Correctness on every bring-up: GSM8K 0.965–0.980, needles 32/32 at 16k / 64k /
128k, zero errors.

At c384 each 1P1D GPU carries 96 sessions against 192 for the aggregated
engine; c768 is the equal-load comparison: +7–10% throughput and 1.6× the
per-user decode speed, at ~10× the median TTFT. That TTFT is the single DP2
prefill queue: each prefill rank runs ~4 requests with ~116 waiting, and 83% of
the mean TTFT (15.4 of 18.5 s) is prefill queue time; decode KV sits at 81%
(p90 99.7%). Mooncake transfer is not a factor (p90 40 ms).

Across c640 / c768 / c892 the prefill completes the same ~54k requests an hour
at c640 and c768 -- both sit on its ceiling, c768 only queues longer -- and
fewer (47k) at c892, where the session working set outgrows the prefill's GPU
prefix cache and 73% of prompt tokens come back from UMBP. On this 1P1D the
throughput plateau is c640–c768 and is prefill-bound.

## Notes & gotchas

1. **A green `/health` proves little.** Each of PATCHES #6, #8, #9 and #11
   presented as a deployment that serves: hung requests, a refusal at start-up,
   500s on every pull, an aborted benchmark. Run `smoke` and `accuracy` on every
   bring-up.
2. **The AgentX warmup is a prefill flood.** Every replay lane opens with a cold
   ~128k-token prompt; at c768 one DP2 prefill needs ~25 min to clear them, and
   prefill GPU prefix hits read ~0% until it does (UMBP serves the hits). This
   is capacity, not a broken cache: measurement-phase hits are 50–90%.
3. **Serial dispatch is required beyond the decode's KV capacity.** Concurrently
   dispatched decode legs reserve full-prompt KV and start their pull timeout
   while their prefill is still queued (PATCHES #11).
4. **kv-aware must render exactly what the engine renders.** One token off at
   the start and no block ever hits, with every health signal green. Check
   `cache_hits` on the router's prefill picks for a second turn sent under a new
   session id (PATCHES #12); the router's `infera_router_render_parity` probe
   only asks sglang workers.
5. **Do not edit a script a running bring-up is executing.** Bash reads scripts
   as it goes; an edit mid-run corrupts the rest of it.
6. **Prefill queueing.** For concurrency beyond ~400 on this recipe, consider a
   second prefill engine (2P1D) before tuning anything else.

## Validation status

| what | status |
|---|---|
| image build + self-check (incl. GPU) | validated |
| bring-up, smoke, accuracy gate (GEM legs, standalone UMBP over GEM IPC, serial dispatch) | validated, three bring-ups |
| UMBP data path on a GEM prefill (store, reset GPU cache, load) | validated |
| UMBP embedded mode | smoke + accuracy validated; performance not measured |
| AgentX formal c384 | scored (pre-#10 / pre-#11 / pre-#12 configuration) |
| AgentX formal c768 | scored (PATCHES #1–#13) |
| router DSv4.1 encoder (PATCHES #12) | validated: unit tests, two-turn probes 8/8, near-full hits on AgentX traffic |
| hyper 1.12.0 (PATCHES #13) | full c768 run with no stalled reply |
| `cluster.peermem.sh` (amdgpu 6.16 + peer memory) | not validated with this kit |
| two-node deployment | not validated |
