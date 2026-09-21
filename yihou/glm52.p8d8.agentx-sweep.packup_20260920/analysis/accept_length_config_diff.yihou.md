# MTP accept-length gap — the configuration axis

**Question.** GLM-5.2 MXFP4, EAGLE 5 steps / 6 draft / topk 1, real acceptance.
Three `sglang:spec_accept_length` measurements disagree:

| id | run | shape | accept length |
|---|---|---|---|
| **A** | **today, this phase, 2026-09-18 ~13:10**, P8D8 probe | TP8/DP8 | **4.31–4.88, mean ~4.69** |
| **B** | `agentx.c72.v0519.packup_20260918/`, AgentX C72 | TP8/DP8 | **2.05–3.26, mean 2.57** |
| **C** | `glm52-agentx-c40-realacc.packup_20260918/`, AgentX C40 | TP4/DP4 | **2.15–3.08, mean ~2.57** |

Is the A↔{B,C} gap explained by **configuration**, or is configuration essentially
the same (pointing at the workload/input instead)? My axis is configuration only.

Evidence tags: **[1h]** = read first-hand from a log / argv / config in the tree;
**[2h]** = a claim in someone's README or the task brief.

Sources, all first-hand argv where possible:
- **A** argv: `ssh crsuse2-m2m-137 cat /mnt/m2m_nobackup/yihou_p8p4/logs/launch-probe-p8d8.log` **[1h]**; config `scripts/config.yihou.p8d8.sh` **[1h]**; probe `scripts/probe_mtp.yihou.sh` **[1h]**; accept 4.69 from the task brief **[2h]**.
- **B** decode `server_args` dump: `agentx.c72.v0519.packup_20260918/logs/decode.excerpt.log.gz` **[1h]**; `README.md` **[1h]**.
- **C** argv: `.../glm52-agentx-c40-realacc.packup_20260918/results/launch-command-lines.txt` **[1h]**; `README.md` **[1h]**.

---

## Three-column comparison

Rows that can move MTP acceptance are grouped first. `custom AR` = the custom
all-reduce kernel; `--disable-custom-all-reduce` **present** means custom AR **OFF**.

| # | setting | A — today P8D8 (4.69) | B — c72.v0519 P8D8 (2.57) | C — c40 P4D4 (2.57) | verdict |
|---|---|---|---|---|---|
| 1 | `--speculative-algorithm` | EAGLE [1h] | EAGLE [1h] | EAGLE [1h] | SAME |
| 2 | `--speculative-num-steps` | 5 [1h] | 5 [1h] | 5 [1h] | SAME |
| 3 | `--speculative-eagle-topk` | 1 [1h] | 1 [1h] | 1 [1h] | SAME |
| 4 | `--speculative-num-draft-tokens` | 6 [1h] | 6 [1h] | 6 [1h] | SAME |
| 5 | `num_reserved_decode_tokens` | default 256 (not in argv) [1h] | 256 [1h] | default 256 (not in argv) [1h] | SAME |
| 6 | `speculative_use_rejection_sampling` | True (ROCm auto) [2h] | True [1h] | True (ROCm auto) [2h] | SAME |
| 7 | NextN shared-experts fusion fix | present [1h, aiter 257/9 in log] | present [1h, patch 0003 + decode log] | present [1h, decode log 2×] | SAME |
| **8** | **`index_share_for_mtp_iteration`** | **ABSENT — not overridden** [1h] | **`false`** [1h] | **`false`** [1h] | **DIFFERENT (A vs B,C)** |
| **9** | **custom all-reduce** | **ON** (`--disable-custom-all-reduce` absent) [1h] | **OFF** (`disable_custom_all_reduce=True`) [1h] | **OFF** (`--disable-custom-all-reduce` present) [1h] | **DIFFERENT (A vs B,C)** |
| 10 | `enable_aiter_allreduce_fusion` | True [1h] | True [1h] | True [1h] | SAME |
| 11 | `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK` | 1 [1h] | UNKNOWN (not in env.sh, image default) | 0 [1h] | DIFFERENT (A=1 vs C=0); B UNKNOWN |
| 12 | `SGLANG_OPT_USE_TOPK_V2` | false [1h] | UNKNOWN (not in env.sh) | false [1h] | A=C SAME; B UNKNOWN |
| 13 | `attention_backend` / dsa prefill/decode | dsa / tilelang / tilelang [1h] | dsa / tilelang / tilelang [1h] | dsa / tilelang / tilelang [1h] | SAME |
| 14 | `dsa_topk_backend` | sgl-kernel (cleared → default) [1h] | sgl-kernel (image default) [1h] | sgl-kernel (cleared → default) [1h] | SAME |
| 15 | `kv_cache_dtype` | fp8_e4m3 [1h] | fp8_e4m3 [1h] | fp8_e4m3 [1h] | SAME |
| 16 | `page_size` | 64 (default) [2h] | 64 [1h] | 64 (default) [2h] | SAME |
| 17 | sglang commit | `a9fb1c3238` [2h] | `a9fb1c3238` [1h] | `a9fb1c3238` [1h] | SAME |
| 18 | `enable_dp_attention` | True [1h] | True [1h] | True [1h] | SAME |
| 19 | `mem_fraction_static` (decode) | 0.85 [1h] | 0.85 [1h] | 0.85 [1h] | SAME |
| 20 | TP / DP / EP | 8 / 8 / 1 [1h] | 8 / 8 / 1 [1h] | **4 / 4 / 1** [1h] | DIFFERENT (C is P4D4) |
| 21 | `max_running_requests` / cuda-graph max bs | 128 [1h] | 72 [1h] | 64 [1h] | DIFFERENT (all three) |
| 22 | prefill HiCache (#37152) | **ON** [1h] | OFF [1h] | OFF [1h] | DIFFERENT (A vs B,C) |
| 23 | decode HiCache | OFF [1h] | OFF [1h] | OFF [1h] | SAME (engine rejects decode HiCache+MTP) |
| 24 | base image / ROCm | v0519-yihou-0917, rocm720 base [2h] | v0519-yihou, **rocm10** base [1h] | v0519-yihou-0917, rocm720 base [1h] | DIFFERENT (B rocm10) |
| 25 | simulated acceptance | OFF [1h] | OFF [1h] | OFF [1h] | SAME (all real) |
| **W** | **workload that produced the number** | **synthetic: 1 fixed short prompt (first-10-primes list), 16x, temperature 0, MAXTOK 2048 (the 256-token default reads ~4.05; 4.69 requires 2048 -- corrected 14:20 UTC)** [1h] | **real AgentX C72 agentic replay; ISL to 830 k, OSL mean 526** [1h] | **real AgentX C40 agentic replay; 1151 req, diverse tool-calling/reasoning** [1h] | **DIFFERENT (A synthetic vs B,C real)** |

---

## Assessment of every DIFFERENT row

**#8 `index_share_for_mtp_iteration` — A absent vs B,C `false`. GENUINE candidate.**
This override sits directly in the DSA-indexer path used by the MTP draft. B and C
force it `false`; A leaves it unset so the model-config default (expected `true` for
GLM DSA NextN) applies. If the default aligns the draft's index selection better
across the 5 draft iterations, the draft predicts more accurately → higher
acceptance. Mechanism is plausible and on the speculative path. **Cannot be
excluded.** But it is confounded — see below.

**#9 custom all-reduce — A ON vs B,C OFF. Candidate, but ambiguous in direction.**
The garbled-decode root-cause pack-up established, on **TP4 only**, that custom AR
**ON** causes GLM-5.2 MTP garbling *and* a **low** accept (1.25). A here has custom
AR **ON** on **TP8** and reads a **high** 4.69 — the opposite sign. Two readings, not
separable from this evidence: (a) on TP8 custom AR ON does not garble and genuinely
helps; (b) A's high number is a garbled-but-high artefact — with rejection sampling,
a draft and target that degenerate into the *same* repeated tail still count as
"accepted", inflating the gauge. **A high accept length with custom AR ON is not
proof of coherence.** The probe's own coherence verdict (its PRIMARY check is
"contains first-10-primes in order") is the arbiter, and I do not have that verdict
in hand. Flag: confirm A's coherence before trusting 4.69.

**#11 `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK` — A=1 vs C=0, B unknown. Weak.**
JIT vs non-JIT is a kernel-implementation choice for grouped MoE top-k; the top-k
selection itself is deterministic given weights, so the two paths should return
identical experts. No plausible first-order effect on acceptance. Undetermined vs B.

**#20 TP/DP (P8D8 vs P4D4) — EXCLUDED as the driver.** B (P8D8) and C (P4D4) sit in
the *same* config arm (#8 false, #9 OFF) on the *same* real workload and both read
**2.57**. Shape does not move acceptance here.

**#21 `max_running` / graph bs (128/72/64) — Weak.** A scheduler concurrency cap.
Batch composition under DP-attention can reorder reductions, a second-order effect
at most; not a first-order driver of per-sequence accept length.

**#22 prefill HiCache (#37152) — NOT a candidate.** Acceptance is a decode-leg
property measured on the decode engine. Prefill HiCache changes prefix-cache hits on
the prefill side; the KV content handed to decode is identical. No mechanism to
change decode MTP accept.

**#24 base image / ROCm (B rocm10 vs A,C rocm720) — EXCLUDED as the driver.** The
sglang **commit is identical** across all three (`a9fb1c3238`, #17), so the
speculative logic is byte-identical. Runtime numerics could differ infinitesimally,
not 2.57→4.69. Decisively: **A and C share the same base and image** yet read 4.69 vs
2.57 — base/image cannot explain the A↔C gap.

**#W workload — the largest DIFFERENT row (someone else's axis; recorded here as
required).** MTP acceptance is fundamentally *how predictable the output sequence
is*. A's output is "2, 3, 5, 7, 11, 13, …" plus one boilerplate sentence — trivially
predictable for an EAGLE draft, so a very high accept length is the *expected* result
regardless of engine config. B and C replay real agentic content (code, tool calls,
reasoning), far less predictable. This difference alone is mechanistically sufficient
to produce 4.69 vs 2.57.

---

## Verdict on the configuration axis — suspended, deliberately

Configuration is **not** essentially the same: rows #8 (`index_share_for_mtp_iteration`)
and #9 (custom all-reduce) are real, MTP-relevant differences between today's probe
(A) and **both** low runs (B, C), with #11 differing vs C as well.

But this evidence **cannot attribute the gap to configuration**, because every one of
those config differences is **fully confounded with the workload difference (#W)**: A
differs from B and C on config *and* on workload at the same time. There is **no run
in these three that isolates #8 or #9 at a fixed workload.** What the three *can*
establish:

1. **Shape is excluded** (#20): B=C=2.57 across P8D8 vs P4D4.
2. **Base image / ROCm is excluded** (#24): A and C share them, differ in result.
3. The surviving config candidates are **#8 `index_share_for_mtp_iteration`** and
   **#9 custom all-reduce** — both **untested at fixed workload**.
4. The workload difference (#W) is large and, on its own, a sufficient explanation.

So: I do **not** conclude configuration is the cause, and I do **not** conclude it is
not. The candidates and the confound are named above.

**The experiment that would separate them** (one axis fixed): either run today's
synthetic 16×-primes probe against a deployment in the B/C arm
(`index_share_for_mtp_iteration:false` **+** `--disable-custom-all-reduce`), or run
the real AgentX workload against today's exact P8D8 arm. The parallel live experiment
is taking the workload axis; a config-side A/B on #8 and #9 at fixed (synthetic)
workload would close the config axis.

## What is UNKNOWN (not inferred)

- B's `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK` (#11) and `SGLANG_OPT_USE_TOPK_V2`
  (#12): not in `env.sh`; the C72.v0519 stack took image defaults and captured no
  container-env dump, so these are UNKNOWN, not assumed equal to A or C.
- A's per-rank coherence verdict from the probe: not in hand; needed to decide
  whether 4.69 is coherent or a garbled-but-high artefact (bears on #9).

---

## Leader addendum 2026-09-18 13:55 UTC — the coherence UNKNOWN is now closed

This report listed "A's probe coherence verdict" as UNKNOWN, and correctly refused to
trust 4.69 without it, because of a real alternative explanation: **rejection sampling
counts a draft and target that degenerate to the same garbage tail as "accepted"**, so
a garbled run can read HIGH acceptance. That would have inverted the sign of the
custom-all-reduce row.

**It is ruled out by first-hand evidence the leader holds.** Probe run 2 on arm A
(`/mnt/m2m_nobackup/yihou_p8p4/probe/p8d8-run2` on `crsuse2-m2m-137`, simulation OFF,
`MAXTOK=2048`):

- **VERDICT: COHERENT.** 16/16 completions contain the first ten primes **in order**;
  0 empty; 0 degenerate tails.
- Verbatim from `req-16.json`: `2, 3, 5, 7, 11, 13, 17, 19, 23, 29` followed by a
  correct one-sentence explanation of why 1 is not prime.
- Per-rank acceptance at that moment: 4.31–4.88.

So arm A is **coherent AND high**, not garbled-and-high. Consequences for this report:

1. **Row #9 (custom all-reduce) loses its ambiguity in sign.** On TP8 with custom
   all-reduce ON the output is correct, so the TP4 garbled-decode failure does not
   reproduce on this shape. It remains a config delta versus B and C; it is no longer
   a candidate explanation for a *spurious* 4.69.
2. The workload explanation is strengthened rather than weakened: a coherent run on a
   memorised sequence is exactly where a one-layer EAGLE draft should score highest.
3. Row #8 (`index_share_for_mtp_iteration`) is **untouched by this addendum** and
   remains a genuine, unexcluded candidate, still confounded with workload.

The suspended verdict stands otherwise: configuration is not identical, no run
isolates #8 or #9 at fixed workload, and the workload difference alone is
mechanistically sufficient. See `accept_length_is_input_driven.yihou.md`, where the
same deployment spans 2.51–4.46 on prompt class alone.
