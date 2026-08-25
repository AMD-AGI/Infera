# GLM-5.2-FP8 on 2× gfx942 — Docker + shell deployment

Prefill/decode disaggregation (1P1D) for GLM-5.2-FP8 on two gfx942 nodes
(MI300X / MI325X): SGLang TP8 with DP-attention, MTP speculative decoding, fp8 KV
cache, Mooncake RDMA for the KV hand-off, and Infera kv-aware routing. Bring-up is
plain shell scripts against a long-lived container on each node.

For the same deployment under Kubernetes, see [`kubernetes.md`](kubernetes.md).
The two produce the same engine — verified flag by flag and on the same workload
— so anything you tune on one carries to the other.

All commands run from `examples/glm5.2_gfx942/` in the repository.

---

## 0. What to expect

Measured on 2× MI325X with a multi-turn agentic workload (60 conversations, 448
requests, median input 68k tokens):

| | Batch / high concurrency | Interactive / low concurrency |
|---|---|---|
| Configuration | default (`DP=8`, DP-attention on, `CHUNK=8192`) | `DP=1 CHUNK=2048` |
| Aggregate throughput at concurrency 16 | **182.7 tok/s** | 139.4 tok/s |
| Per-user speed at concurrency 1, median | 62.5 tok/s/user | **133–159 tok/s/user** |
| Mean TTFT at concurrency 16 | 12.4 s | 22.2 s |
| Cache efficiency | 100.00% | 100.00% |

Same weights, same image, one flag apart. **DP-attention is a concurrency
trade-off, not an optimisation that is always on** — §8 has the switch. Pick the
column before you tune anything else; nothing else measured here moved the numbers
nearly as much.

Cold start is 15–25 minutes per node. The log goes quiet during weight loading.
Do not kill a slow load.

---

## 1. Prerequisites

**Hardware.** Two nodes, 8× gfx942 each, ~272 GiB host RAM on the prefill node
(which also carries etcd and the router). The two nodes need a RoCE fabric between
them: the KV hand-off is RDMA, and there is no TCP fallback worth having.

**Read the host driver version first — on both nodes.** It decides which base
image you build on, and getting it wrong does not refuse to start:

```bash
dpkg -l | grep -E 'amdgpu-dkms|rocm-core'
```

A container brings its own ROCm userspace but talks to the *host's* amdgpu through
`/dev/kfd`, and AMD supports that pairing only within a window:

| Host driver | ROCm userspace | Base image |
|---|---|---|
| 6.4.x | up to 7.2.x | `rocm720` — the Dockerfile default |
| 6.3.x | up to 7.0.x | `rocm700` — you **must** override |

A mismatched pair initialises, loads weights, captures graphs, and then faults
with `Memory access fault by GPU node-N` somewhere under load, in a different
place each time. Two `dpkg -l` lines are cheaper than a rebuild.

**Model weights.** A local directory on both nodes, at the same path — a shared
filesystem or an independent download each. A HuggingFace snapshot path or a
symlink into one is fine; `host_container.sh` detects the symlink and mounts the
repository root so the snapshot's links into `blobs/` still resolve.

```bash
hf download zai-org/GLM-5.2-FP8
```

---

## 2. Build the image

**Do not use a vendor-preinstalled GLM-5.2 image.** GLM-5.2 on the gfx942 SGLang
base needs a rebuilt Mooncake and four SGLang source patches, and a preinstalled
image predating any of them fails in ways that look like something else. On both
nodes (or build once and push to a registry both can pull):

```bash
bash build_image.sh                                        # 6.4.x host driver
SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.16-rocm700-mi30x \
  bash build_image.sh                                      # 6.3.x host driver
```

The image is ~107 GB and takes about 40 minutes. Three of the four patches leave
markers, which is the cheap way to tell a correctly built image from one where a
patch silently no-op'd against a moved anchor:

```bash
docker run --rm --entrypoint python3 \
  -v "$PWD/check_image.py:/check.py:ro" infera:sglang-gfx942-glm52 /check.py
```

Three `YES` is the pass:

```text
sglang_rocm/host_alloc       srt/mem_cache/pool_host/common.py            YES
sglang_rocm/staged_wb        srt/mem_cache/pool_host/mla.py               YES
sglang_disagg/early_send     srt/disaggregation/mooncake/conn.py          YES
```

This is also how to check an image someone handed you. The GLM-5.2 image
preinstalled on the validated nodes reports `no` on the first line, which is why
these guides tell you to build your own.

Tag with the source commit if you build more than once — it is what lets a running
container be traced back to the tree it came from.

---

## 3. Configure

Everything cluster-specific lives in one file, on **both** nodes:

```bash
cp cluster.env.example cluster.env    # then edit
```

`env.sh` reads it before its own defaults and every script sources `env.sh`, so
there is nothing else to edit and nothing to remember at the call site. The repo
is bind-mounted at the same path inside the container, so one copy serves the host
and the container.

| Variable | What it is |
|---|---|
| `PREFILL_NODE` / `DECODE_NODE` | Hostnames. The prefill node also runs etcd and the router. |
| `PREFILL_IP` / `DECODE_IP` | Their addresses **on the data network**, not the management NIC. |
| `ETCD_ENDPOINT` | Where PD's own etcd listens. `2379` unless something already holds it. |
| `IB_DEVICE` | The RoCE rail Mooncake moves KV over — `mlx5_0`, `rdma0`, … |
| `MC_GID_INDEX` | The RoCE v2 GID index on that rail (§4). |
| `MODEL` | Local weights directory. |
| `IMAGE` | What you built in §2. |

`cluster.env.example` gives, for each value, the command that finds it. Three of
them fail *silently* when wrong and are worth the two minutes each:

- **`MC_GID_INDEX`** — §4. Wrong means slow, not broken.
- **The advertised IPs** — a management address does not fail at startup. Both
  legs register happily and the hole opens at the first KV hand-off, which is
  after the weights load. Discovering it costs a full cold start.
- **`ETCD_ENDPOINT`** — if a Kubernetes control plane on the same node already
  holds 2379/2380, move PD's etcd. `launch_etcd.sh` derives the peer port as
  client port + 1.

---

## 4. Check the fabric

Cross-node PD ships the KV cache on every request, so this is worth doing
properly. Run on each node's host shell:

```bash
bash preflight_rdma.sh
```

The count of active RDMA ports visible inside the container must equal the node's
own — **not zero**. Zero means `ibv_get_device_list()` found nothing, Mooncake
falls back to TCP, and the deployment still comes up and answers.

**Then find `MC_GID_INDEX` rather than copying one.** A wrong GID index does not
error. It just runs slower, and the range is wide:

| Configuration | Single-rail bandwidth | Relative |
|---|---|---|
| Correct RoCE v2 IPv4 index | 8.8 GB/s | baseline |
| Left for Mooncake to choose | 0.5 – 2.4 GB/s | **4–18× slower** |
| Forced to TCP | 0.08 GB/s | **110× slower** |

All three answer correctly with nothing in any log. The index is also **not
portable between clusters** — the same number that is the RoCE v2 entry on one
cluster is the management NIC on another. Read it off the rail you picked:

```bash
for i in $(seq 0 7); do
  echo "$i $(cat /sys/class/infiniband/$IB_DEVICE/ports/1/gid_attrs/types/$i 2>/dev/null)" \
       "$(cat /sys/class/infiniband/$IB_DEVICE/ports/1/gids/$i 2>/dev/null)"
done
```

Take the index whose type is `RoCE v2` and whose GID is the IPv4-mapped form of
this node's data IP (`::ffff:10.0.0.1`).

**Confirm the rail is ACTIVE on both nodes.** Both legs pin to one rail, and a
rail that is down on one side does not report a transport error — the decode leg
loads weights, allocates KV, starts serving, and then dies in PD warmup with
`Memory access fault by GPU node-N`, thousands of log lines after the three
info-level lines that explain it.

```bash
for d in /sys/class/infiniband/*; do echo "$(basename "$d") $(cat "$d/ports/1/state")"; done
ib_write_bw -d "$IB_DEVICE" -x "$MC_GID_INDEX"                  # decode node
ib_write_bw -d "$IB_DEVICE" -x "$MC_GID_INDEX" "$DECODE_IP"     # prefill node
```

---

## 5. Bring up

```text
prefill host:       host_container.sh  →  launch/launch_etcd.sh
decode host:        host_container.sh
prefill container:  launch/launch_prefill.sh  →  launch/launch_router.sh
decode container:   launch/launch_decode.sh
prefill container:  verify.sh
```

The two legs find each other through etcd, so decode can start in parallel with
prefill; the router only has to see both registered before it takes traffic.

On both hosts:

```bash
bash host_container.sh
```

This creates the long-lived engine container and checks, before you spend 20
minutes on a load, that the image is present and that `config.json` is readable
through the mounts as the engine will read it.

On the prefill host only:

```bash
bash launch/launch_etcd.sh
```

Then inside the container (`docker exec -it infera-glm52-gfx942 bash`) —
on the prefill node:

```bash
bash launch/launch_prefill.sh
bash launch/launch_router.sh
```

and on the decode node:

```bash
bash launch/launch_decode.sh
```

**Cold start is 15–25 minutes** — weight loading, then memory pools, JIT and CUDA
graph capture, which print almost nothing. Follow it with `tail -f logs/prefill.log`.
`Ctrl-C` on that tail does not stop the engine; the launch scripts use `nohup`, so
use `stop.sh`.

Both legs run the same flags apart from the port and `--disaggregation-mode`:

```text
--tp-size 8 --dp-size 8 --enable-dp-attention
--kv-cache-dtype fp8_e4m3
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang
--mem-fraction-static 0.85 --max-running-requests 128 --chunked-prefill-size 8192
--speculative-algorithm EAGLE --speculative-num-steps 5
--speculative-eagle-topk 1 --speculative-num-draft-tokens 6
--disaggregation-transfer-backend mooncake
```

---

## 6. Verify

```bash
bash verify.sh
```

Six checks, each aimed at a failure this stack produces **without returning an
error**. It exits non-zero if any assertion fails.

1. **Both legs registered** in etcd.
2. **Correctness** — a padded prompt with a known answer. A broken KV hand-off
   returns HTTP 200: the decode leg reads a corrupt prefix and writes fluent text
   that has nothing to do with the question. Only checking the answer catches it.
3. **kv-aware routing is live** — the router logged a prefill pick with
   `request_blocks > 0`. With no block hashes it silently routes on load alone,
   and looks perfectly healthy doing it.
4. **MTP is accepting** — `sglang:spec_accept_length` on the decode leg's
   `/metrics`. Speculative decoding is dropped silently when the two legs disagree
   on its shape.
5. **kvd** — skipped at the default `KVD=0`.
6. **RDMA hand-off** — Mooncake transport lines in the decode log.

---

## 7. Benchmark

Two workloads answering different questions.

### 7.1 Sizing, on random prompts

```bash
bash bench.sh                        # ISL=4096 OSL=1024 CONC=16
ISL=8192 OSL=512 CONC=32 bash bench.sh
bash run_sweep.sh                    # concurrency 1..128
```

The cache-hit line will read ~0 and **that is correct** — random prompts share no
prefix, so a kv-aware router has nothing to reuse. This measures raw serving
throughput. Use `run_sweep.sh` rather than a bare loop over `CONC`: at a fixed
seed each point's prompt set is a superset of the one below it and the cache still
holds the smaller one, which reads as ~50% hits at every point from 16 up.

### 7.2 The agentic workload, which is what the recipe was tuned on

Real multi-turn agent traffic with long contexts and heavy prefix sharing — the
shape PD disaggregation and kv-aware routing exist for. Build the dataset once, in
the container on the prefill node:

```bash
hf download semianalysisai/cc-traces-weka-062126-256k --repo-type dataset
SRC=$(ls "$HF_HOME"/hub/datasets--semianalysisai--cc-traces-weka-062126-256k/snapshots/*/traces.jsonl)

python3 weka_to_agentic_trace.py "$SRC" -o "$TRACE" \
  --output-len "$OUTPUT_LEN" --min-turns 4 --max-context 100000 \
  --verify 20 --tokenizer "$MODEL"
```

The corpus carries per-turn token counts and KV block ids but no text, so the
converter synthesises filler while preserving exactly what matters: per-turn
lengths and block-level prefix reuse. Then:

```bash
NUM_PROMPTS=60 CONC=16 bash run_agentic_trace.sh docker
```

`NUM_PROMPTS` counts **conversations**, not requests — 60 conversations averaging
~7.5 turns is 448 requests.

### 7.3 Read the result

`sglang.benchmark.serving` mis-reports the input side in multi-turn mode: it keeps
the conversation-level `prompt_len` for every turn, so its own summary can print
`Total input tokens: 0` next to a cache hit rate above 100%. The per-request
`cached_tokens` come from the server and are correct, so `score_agentic_trace.py`
recomputes against the dataset's verified per-turn lengths and prints the number
worth comparing:

```text
  actual hit rate              84.61 %
  ideal  hit rate              84.61 %
  efficiency (a/i)            100.00 %
  tokens lost to evict              0 (0.00% of ideal)
```

**Efficiency** is actual hits over what a cache that evicted nothing could have
returned. 100% with no eviction means the run is below the pressure point and
kv-aware routing has nothing to distinguish itself on — raise `CONC` until
eviction appears. Efficiency **above** 100% means the flush did not take:
`flush_cache` is a no-op while requests are in flight and still returns success.

A run containing any failed request is refused rather than scored — a failure is
recorded with `cached_tokens=0`, so scoring it would make a dead worker read as a
cache problem.

---

## 8. The two operating points

The default recipe was tuned at concurrency 16, where DP-attention wins: pure TP8
prefill measured 25.9% slower. At concurrency 1 the picture inverts, and it is not
a small effect.

Under DP-attention every rank carries the **whole, unsharded** attention weights.
With one request in flight only one rank has real work, but all eight run the full
78-layer forward to take part in the MoE all-reduce — on a matrix shape nobody
tuned for. Profiling the default at concurrency 1 shows the GPU 79–86% busy, over
half of it in a single output-projection GEMM that occupies 48 of 304 compute
units. Attention itself is 0.7% and collectives are 6–13%: neither the fabric nor
the KV bandwidth is the constraint.

Turning it off:

```bash
DP=1 CHUNK=2048 bash launch/launch_prefill.sh     # DP=1 disables dp-attention
DP=1 CHUNK=2048 bash launch/launch_decode.sh
```

Measured over three runs at concurrency 1, against the default recipe:

| | Default (`DP=8`) | `DP=1 CHUNK=2048` |
|---|---|---|
| Mean TPOT | 18.63 ms | 7.30 – 8.02 ms |
| Median tok/s/user | 62.5 | **133 – 159** |
| Cache efficiency | 100.00% | 100.00% |
| MTP accept length | 4.29 | 4.32 |

Efficiency and acceptance are unchanged, so this is operator efficiency and not
correctness traded away.

**`CHUNK` must move with `DP`.** `CHUNK` is a global budget that SGLang divides by
`dp_size` **only while DP-attention is on**. At the default, `8192` is 1,024 per
rank; at `DP=1` the same `8192` is 8,192 per rank — eight times the activation
memory. At concurrency 1 it survives; a long prefill at concurrency 16 goes
straight to `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. `2048` is the largest value
measured not to OOM here, and it beats `1024` on every metric. Read back what
actually took effect — the log line `chunked prefill size adjusted from … to …`
means it was *divided*, not rejected:

```bash
curl -s "$PREFILL_URL/get_server_info" | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print(d["chunked_prefill_size"], "per rank x", d["dp_size"], "ranks")'
```

**What it costs at high concurrency**, same workload at concurrency 16:

| | DP-attention on | off |
|---|---|---|
| Output throughput | **182.7 tok/s** | 139.4 tok/s (−23.7%) |
| Mean TTFT | **12.4 s** | 22.2 s (1.79×) |
| Mean TPOT | 21.50 ms | **9.05 ms** |

Part of that TTFT is `CHUNK` dropping from 8192 to 2048, not DP-attention alone.

**Two things to know before you tune further.** Single-point measurements at
concurrency 1 carry about ±5% run-to-run jitter on this workload, so any
conclusion smaller than 10% needs a repeat. And per-user speed rises with context
length here — the 50–80k bucket medians 153–163 tok/s/user while short opening
turns sit near 99 — because long agent conversations repeat tool-call JSON that
the draft model predicts well. Report the bucket your traffic actually lives in.

Three other tuning directions were tried and ruled out on this hardware: custom
all-reduce **deadlocks** during EAGLE verify on gfx942/gfx950, so
`--disable-custom-all-reduce` has to stay; tree-shaped drafting
(`--speculative-eagle-topk > 1`) is rejected by the DSA attention backend at
`page_size > 1`; and a deeper draft chain is a net loss — acceptance rises 31% but
per-step cost rises 33%. The shipped `5/1/6` is already at the optimum.

---

## 9. Stop

Inside the container on both nodes:

```bash
bash stop.sh
```

Then on the hosts:

```bash
bash host_container.sh --rm
docker rm -f infera-glm52-etcd     # prefill host only
```

**Always clear the engine processes before relaunching**, or the next start OOMs
against VRAM the previous one still holds.

---

## 10. Troubleshooting

**Nothing answers, or answers slowly, with no error anywhere.** Work down §4:
active port count, GID index, rail state on both nodes. Every RDMA failure in this
stack is silent.

**A fluent answer that ignores the question.** The KV hand-off is broken; the
decode leg is reading a corrupt prefix. `verify.sh` check 2 exists for this.

**`Memory access fault by GPU node-N` after a clean-looking startup.** Two causes
seen: the pinned rail is down on one node (look for `topology.cpp … is not
active`, `has no active ports, skipping` thousands of lines earlier), or the base
image does not match the host driver (§1).

**`chunked prefill size adjusted from … to …`.** Not a rejection — it was divided
by `dp_size`. See §8.

**Cache efficiency above 100%.** The flush did not take, because something was
still in flight. `flush_cache` is a no-op with requests running and returns
success anyway.

**Killing a benchmark leaves it running.** `pkill -f run_agentic_trace.sh` only
kills the wrapper. Find the container and remove it:

```bash
docker ps --no-trunc --format '{{.Names}}|{{.Command}}'
```

**Zombie python processes after a failed launch.** `pkill -9` does not reap them.
Restart the container.

**Both legs must agree on the MTP shape.** SGLang rejects a mismatched
disaggregated pair — and it does so *after* the weights load, so a one-sided
change costs a full cold start to discover. Change `MTP_STEPS` / `MTP_DRAFT_TOKENS`
in `cluster.env`, which both legs read.

---

## Where these numbers come from

Every figure here was measured on a pair of MI325X nodes; the full record —
raw benchmark output, profiler traces, the flag-by-flag comparison against the
Kubernetes deployment, and the tuning attempts that failed — is in
[`mi325x-handoff/`](../../mi325x-handoff/README.md).
