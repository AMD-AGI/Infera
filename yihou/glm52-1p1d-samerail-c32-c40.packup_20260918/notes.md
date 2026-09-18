# Notes — gotchas, wrong turns, error analysis

Ordered by how much time each costs if you don't know it.

---

## 1. Mooncake per-GPU VRAM transfer fails unless you pin the NIC yourself

**What.** With the stock shared device list (`RDMA_DEVICE=ionic_0,…,ionic_7`),
`preflight`'s per-GPU VRAM transfers between 135 and 138 failed on **GPUs 4-7 in
both directions**, while GPUs 0-3 passed at 45.1-45.9 GB/s, byte-verified. All
16 ionic ports were `ACTIVE / LinkUp / 400 Gb/s` on both nodes, and the CPU-DRAM
RDMA path passed on the same NICs.

**Why.** Mooncake's auto-discovery assigns each GPU an HCA from its NUMA-local
pool, and the two ends choose **independently**. These rails are physically
isolated — `ionic_i` can only reach `ionic_i`; every off-diagonal pairing dies
with `IBV_WC_RETRY_EXC_ERR` — so when the ends disagree the transfer is
*unreachable*, not slow. The mechanism was already hypothesised in the
2026-09-17 fleet survey (`infera.glm52.view/rdma.survey.8node.packup_20260917`
§5); this run confirmed it first-hand.

`apply_mooncake_topology_default()` in `infera/engine/rocm_rdma_env.py` exists to
pin exactly this, but it (a) is only called from the **vllm** path, not sglang,
and (b) resolves NICs through IPv4-mapped GIDs while these rails are IPv6-only,
so it no-ops here regardless.

**How.** `sglang`'s `parse_ib_device_config()` accepts three forms for
`--disaggregation-ib-device`: a shared list, a **per-GPU JSON map**, or a path to
a JSON file. Use the map — no repo change needed:

```bash
RDMA_DEVICE='{"0":"ionic_2","1":"ionic_3","2":"ionic_4","3":"ionic_5"}'
```

Keys are the **local (visible)** device index, so under
`HIP_VISIBLE_DEVICES=2,3,4,5` key `0` is physical GPU 2. Confirmed empirically,
not assumed: key `"2"` lands on `ionic_4`, which a physical-index reading could
not produce.

**Context.** Two traps around this:

- `MC_TE_FILTERS` defaults to `$RDMA_DEVICE` in `config.sh`, which would hand the
  transport engine a JSON blob where it expects a device list. Set it explicitly
  to `ionic_2,ionic_3,ionic_4,ionic_5`.
- A bare `}` inside `${VAR:=…}` terminates the parameter expansion early and
  **silently truncates the JSON**. Assign with an explicit `if [[ -z … ]]` test.
  This cost a debugging cycle; the symptom is a config value missing its last
  character.

**The lesson worth keeping.** A symmetric, cleanly-grouped failure looks exactly
like dead hardware, and was reported as such right up until the cheap
contradicting evidence was checked: *the CPU-DRAM RDMA path passed over the very
same NICs.* If the rails were dead, that path would have failed too. Look for
the cheap contradiction before naming hardware.

## 2. The DSA patch set can fail silently in exactly one place

**What.** `deploy/docker/scripts/apply_sglang_dsa_patches.sh` was cut against the
**20260916** nightly; this run used **20260917**.

**Why it is mostly safe.** It applies with `--fuzz=0` and verifies seven
identifiers reached the compiled bytecode, so a drifted base **hard-fails** —
which is the good outcome. (Note: `Hunk #N succeeded at X (offset Y lines)` in
the build log is a *line shift*, not fuzz. Offsets are fine; rejects are not.)

**The exception.** `draft_cuda_graph_dp_vote.diff` rides **DP-sync slot 8**
(upstream took slot 7 for `prefill_cuda_graph_max_prefix_len`). If upstream
shuffles the slots again, a mechanical re-anchor compiles **and passes the
bytecode markers** while reading a prefix length as a boolean. The markers
cannot catch this.

**How to check** — read the columns by hand, both sides, in the built image:

```bash
docker run --rm --entrypoint /bin/bash <image> -lc \
  'sed -n "108,146p;218,228p" \
   /sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/dp_attn.py'
```

`_get_local_tensor` must pack 9 elements with index 7 =
`prefill_cuda_graph_max_prefix_len` and index 8 = `int(can_run_draft_cuda_graph)`;
the unpack must read `[:, 7].max()` and `[:, 8].min()` respectively. Verified
correct on 20260917.

## 3. RESOLVED — garbled decode output on P4D4

> **Closed on 2026-09-18, after this packup was written.** Root cause: **custom
> all-reduce on the decode leg corrupts the speculative path.** Passing
> `--disable-custom-all-reduce` to the decode engine restores correct text and
> healthy MTP acceptance (`spec_accept_length` 2.55-3.05 on all four DP ranks,
> against the 1.25 recorded below). Full five-round A/B, evidence and
> reproduction kit:
> `yihou/glm52-mtp-garbled-decode-rootcause.packup_20260918/`.
>
> **What that means for the numbers in this packup.** Every measurement here was
> taken with custom all-reduce **on** — i.e. in the configuration now known to
> decode incorrectly — and with `DECODE_SIMULATE_ACC_LEN=3.61` forcing the
> acceptance length, which masked it. The timings remain valid *as timings
> against other runs using the same simulation*, which is what §1 of the README
> already says. They are **not** measurements of the fixed configuration, and the
> throughput cost of the fix is unmeasured. Do not carry these numbers across.
>
> The section below is left exactly as written. Its evidence was all correct and
> it is the trail that led to the answer; only its final "open" verdict is
> superseded.

**What.** With `DECODE_SIMULATE_ACC_LEN=` (simulation off), a trivial prompt at
`temperature=0` returns real tokens in nonsense order. Probing by `max_tokens`:

| max_tokens | reasoning_content |
|---|---|
| 1 | `1` |
| 3 | `1!!` |
| 8 | `1!!!!!!!` |

**What is established.**

- The **first token is correct** and every later one degenerates. In PD the
  first token comes from prefill, so the fault is on the **decode** side.
- Output is valid tokens in nonsense order → the fault is **numerical**, not
  tokenizer or parser.
- Mooncake failure count **0** → the transport layer is clean and is *ruled out*,
  not merely unsuspected.
- Measured MTP accept rate **0.05** (`accept len: 1.25`) against the 3.61 the
  bench simulates.
- `index_share_for_mtp_iteration=false` (the `issue.md` §2.4 workaround) **was**
  applied, so §2.4's trigger is absent.

**What is NOT established.** A low accept rate alone does not explain wrong
output — rejected drafts should fall back to the target model's token, giving
correct-but-slow results. Getting both a 5% accept rate *and* wrong text means
either the target model is also wrong on the decode side, or the verify/fallback
path is not doing its job. **That question is open.**

**Context.** The obvious next experiment is `DECODE_MTP=0` — a single variable
that splits "MTP path" from "plain decode path". It was launched and then stopped
when the user waived correctness for this delivery. P4D4 is a new shape; the
bench's validated baseline is TP8/DP8, which cannot be run on these two nodes
because 135's GPU[1] is occupied. Raw evidence: `spec/debug_working_process.md`.

**How it actually went** (added 2026-09-18, after the fact). That `DECODE_MTP=0`
experiment was the right call and did decide the question — 16/16 coherent with
MTP off against 0/16 with it on, which exonerated the target model at TP4/DP4,
the PD KV handoff including this packup's same-rail work, and the plain decode
path. The paragraph above marked "NOT established" was also right to refuse the
inference: the low accept rate genuinely was a *symptom* rather than the cause,
and the cause turned out to sit in the all-reduce path, which nothing here had
suspected. Two further rounds then showed the NextN shared-experts-fusion defect
— the lead that looked most promising — is real but **not** the cause, and one
more showed the AITER fusion path is not involved either.

## 4. crsuse2-m2m-135 has two permanent constraints

**GPU[1] is not yours.** A root-owned Kubernetes pod (`pod5d84e491`,
`vllm serve Qwen3-32B`, 22 days uptime) holds 96.94 GB of 309.22 GB. There is no
sudo and no kubectl here, and killing the bare PID would only make kubelet
restart it. **Do not attempt to clear it.** `check_nodes.sh` will report it busy
and exit non-zero; that is correct behaviour — do not raise
`GPU_IDLE_VRAM_PCT` to silence it.

**`ionic_7` is defective.** No netdev (`enP3p0s12` absent) and GID index 1 is
all-zero, so rail `0200` is missing on that node. Independently recorded in the
2026-09-17 fleet survey. Avoid GPU 7 on this node.

Devices **2,3,4,5** dodge both while keeping `mem_fraction_static` at the
validated 0.85 and leaving four healthy dedicated rails.

**crsuse2-m2m-139 / -140** look attractive (fully idle) but `docker.service` is
**masked** on both — `/var/run/docker.sock` absent, unmasking needs root. Not an
option.

## 5. Things that look broken and are not

- **`returned=0/66` during AgentX warmup.** Not a hang. AgentX agentic
  trajectories have very long contexts and the prefix cache starts cold; prefill
  saturates (8192-token chunks, ~1.5 M pending tokens per rank) for several
  minutes before the first handoff completes. It drained to
  `59/66, errors=0` at 570 s.
- **12 `error` matches in a worker log at startup.** All
  `Ignore import error when loading sglang.srt.models.inkling / mamba` — sglang
  probing optional deps for unrelated models.
- **24 `error` matches in the AgentX console.** 23 are the literal field
  `errors=0`; the 24th is `errored=0 … parents_failed_due_to_child_error=0`.
- **`cleanup completed with errors` from `stop.sh`.** It also tries prefixes for
  preflight and client containers that don't exist. Check the actual container
  and GPU state rather than the exit line.
- **`rdma-default` failing in preflight** with *"auto-selected GID is link-local,
  not routable"*. Expected: `MC_GID_INDEX=1` is mandatory on this fabric, not
  tuning. Index 0 is `fe80::` link-local and cannot route.

**A method error of my own, recorded so it isn't repeated:** decode liveness was
checked with `docker logs --tail 300 | grep -c "Decode batch"` and read 0. The
zero came from the tail window — those 300 lines were entirely `/metrics`
polling — not from decode being idle. Don't infer liveness from a truncated log.

## 6. Prefill HBM sits at 90-91% — watched, not a problem here

`issue.md` §3.4 documents prefill `HSA_STATUS_ERROR_OUT_OF_RESOURCES` at 99%,
with the allocator-GC treatment bringing the peak to 93%. Both runs sampled
prefill GPUs 2-5 repeatedly during profiling: **90/91/91/90 every time** for
C32, and 86% idle rising to 90-91% under load for C40 — a steady-state working
point rather than a climb. No OOR occurred in either, including across C40's
3600 s window at BS 128 with HiCache on. Worth sampling again at higher
concurrency — §3.4 is explicit that the failure accumulates gradually and a
pre-launch snapshot cannot see it.

## 7. `config.full.sh` does not run as written on this base

**What.** Three of its values are rejected by the pinned nightly's own argparse:

```bash
DSA_PREFILL_BACKEND=flydsl   DSA_DECODE_BACKEND=flydsl   DSA_TOPK_BACKEND=aiter
```

**Why.** Verified first-hand inside `infera-sglang:v0519-yihou-0917`, in
`sglang/srt/arg_groups/fields/exec_.py` and `spec.py`:

| field | accepted choices |
|---|---|
| `dsa_prefill_backend` | `flashmla_sparse`, `flashmla_sparse_q8`, `flashmla_kv`, `flashmla_auto`, `flashinfer_sparse_mla`, `fa3`, `tilelang`, `triton`, `aiter`, `trtllm` |
| `dsa_topk_backend` | `sgl-kernel`, `torch`, `flashinfer` |

Neither list contains `flydsl`; the top-k list does not contain `aiter`. And
`engine.sh` **does** forward both (lines 148-149 and 165-166), so the engine
dies at startup rather than ignoring them. This matches `issue.md` §2.1/§2.2,
which list both as unported TODOs needing AITER *and* SGLang kernel plus
integration work — not a flag flip.

**How.** `scripts/config.yihou.full.sh` substitutes `tilelang` (engine.sh's own
default) for both DSA backends and leaves `DSA_TOPK_BACKEND` empty so
`engine.sh` omits the flag and sglang uses `sgl-kernel`. **Everything else is
`config.full.sh` as written.** The C40 result is therefore "full minus two
items" and must be labelled that way.

**Context — a bash trap that cost a cycle.** Clearing `DSA_TOPK_BACKEND=""`
*before* sourcing does nothing: `config.full.sh` assigns it with
`${DSA_TOPK_BACKEND:-aiter}`, and `:-` treats an empty string as unset, so
`aiter` comes back. It must be cleared **after** the source. The same trap does
not apply to the two backends, which the adapted config sets with `:=` before
sourcing. Both facts are commented in the config file itself.

## 8. OPEN — C40 dropped 3 real inference errors

**What.** C40's `request_accounting` reports
`records_error_dropped: 3`, `error_categories: {"InvalidInferenceResultError": 3}`
out of 4572 total. AgentX's headline `0/4128` counts *completed* requests, so
these do not appear there. C32 had **zero** such errors.

**Why it is worth a second look.** C40 is also the run that **removed** the
`index_share_for_mtp_iteration=false` workaround (`config.full.sh` empties
`JSON_MODEL_OVERRIDE_ARGS`), and that workaround exists precisely to suppress
garbled output — `issue.md` §2.4. Three invalid inference results appearing in
exactly the run that dropped the anti-garbling workaround is suggestive.

**What is NOT established.** Three errors in 4572 could equally be transport
hiccups, trajectory-replay edge cases, or truncation. No per-request evidence
was captured. **This is recorded, not concluded.** To close it: re-run C40 with
`JSON_MODEL_OVERRIDE_ARGS='{"index_share_for_mtp_iteration":false}'` restored as
the single changed variable and see whether the count goes to zero.

**A second candidate, added 2026-09-18 after §3 was root-caused.** Both C32 and
C40 ran with custom all-reduce **on**, now known to corrupt the speculative path
on this stack. That does not obviously explain why C40 erred and C32 did not —
the flag was identical in both — but it does undercut the IndexShare framing as
the *only* live hypothesis, and it raises a specific question worth checking
before re-running anything: `DECODE_SIMULATE_ACC_LEN=3.61` forces the acceptance
length, so draft tokens are taken without the verification that would normally
reject corrupted ones. If that is what the flag does here, then **both** runs may
have been emitting degraded text throughout and the three errors are the tip of
it rather than an anomaly. Neither the flag's exact semantics nor the output
quality of those runs was examined. Unverified, and it would change how §1's
"valid as timings" caveat should be read — check it before treating either run's
output as sane. The C40 re-run proposed above should carry
`--disable-custom-all-reduce` as well, or it reproduces the same unknown.

**Context.** Also seen in C40 and left unexplained: 15 `OSL mismatch` warnings
(delivered output length below requested, e.g. −29.5%). Frequency noted; effect
on the throughput figures not analysed.

## 9. HiCache behaviour observed in C40

**What.** With Prefill HiCache on (ratio 1.5, `write_through`), the host-side KV
pool reached **`cpu_kv_usage=100.0%`** and stayed there, while HBM-side
`kv_usage` oscillated between 44% and 59%.

**Why it matters.** Prefill HBM at *idle* was **86%** with HiCache on, versus
90-91% for the C32 run without it — i.e. offloading to host memory measurably
relieved HBM, which is the point. Under load it rose back to 90-91%, still well
short of the 99% at which `issue.md` §3.4 records OOR failures.

**Not concluded.** Whether a saturated host pool starts thrashing (evicting and
re-fetching, hurting TTFT) was **not** determined. C40's median TTFT is 29%
*better* than C32's, but six variables differ between the runs, so that
improvement cannot be attributed to HiCache — or to anything else — from these
two points alone.
