# Originating task

Posed by the user in-session on 2026-07-30 (there was no pre-existing spec file).
Reproduced verbatim, in order, so the deliverables can be checked against what
was actually asked.

## 1. The opening request

> 回答一下infera对于sglang engine的kvaware和kvd的的支持情况。实验一下简单的pd + dpa分离打开kvaware和kvd的正确性

Two parts: (a) a support survey of kvaware and kvd for the **sglang** engine, and
(b) an experiment — a simple **PD + DP-attention** deployment with kvaware and
kvd switched on, checked for **correctness**.

## 2. Scope discipline (given when the port bug surfaced)

Asked whether the `free_tcp_port_block` bug was in scope, the user answered:

> 你说的这个是kvaware和kvd的问题么？是的话按1来，强调一下专注问题，不要做额外的事，不要发散

It is in scope — `free_tcp_port_block` is only reached when `enable_kv_events` is
on. Fix approach 1 (fix the root cause in `net.py`). The standing instruction —
stay focused, no extra work, no wandering — was written into the user-level
`CLAUDE.md` at the user's request.

## 3. Nodes for the real GLM-5.2 run

> glm5.2真实实验用chi2879和chi2867

## 4. Escalation to the real model

After the single-node Qwen3-1.7B MVP produced garbled output on both the
feature-on and baseline runs, the user asked for the real thing:

> 要，并且要做正确性测试

i.e. run it on GLM-5.2 across chi2879/chi2867, and include a correctness test.

## 5. Follow-up questions answered along the way

> 关于kvaware + kvd infera都做了什么？具体回答：哪些是sglang原有的，哪些是infera自身的，哪些是infera为了适配sglang做的。

→ the three-layer split in `results/support_matrix.md`.

> 所以我们现在的kvaware + kvd都用了你上面说的哪些？

→ `results/kvaware_kvd_activation_evidence.txt` (what actually lit up) and the
"What is NOT proven" section of `notes.md`.

> 你开了pd和dpa么？

→ yes, symmetric on both legs; `results/pd_dpa_flags_verified.txt`.

## 6. Scope of the follow-on steps

After step 1 passed 4/4 but kvd showed `gets=0`, the user asked to keep going.
Proposed a two-phase plan (basic pass first, then instrumented workload) rather
than flipping every switch at once, so a failure would stay attributable:

> 两步

Then, after step 2:

> 依次进行

— i.e. work the remaining three items in order: cross-restart reuse, ≥2 workers
per role, real NVMe L3. GPU-direct was explicitly dropped from scope:

> gpu direct暂时不用验证了

(It was already on the exclusion list — ionic dmabuf has a known 2× shadow issue
and depends on kernel P2PDMA, which would muddy attribution.)

Before rebuilding the containers for step 5 — a destructive step that would
discard the live kvd store — the user required the progress be packed up first:

> 1. 另外重建前packup一下确保我们能完整复现进展。包括patch等

## Success criteria used

Correctness is judged by `scripts/probe.py` — four temp=0 factual prompts
(paris / beijing / 4 / jupiter) through the router, **≥3/4 required** (the script
exits non-zero below that). This is the same probe the verified
`glm5.2.mxfp4.packup_20260727/07_pd_mooncake_dpa_sweep` kit used, so results are
directly comparable to that known-good 4/4.
