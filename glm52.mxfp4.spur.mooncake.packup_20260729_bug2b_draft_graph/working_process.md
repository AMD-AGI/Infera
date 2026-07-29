# Bug 2b — draft CUDA graph deadlock: debug loop index

**Goal (binary, measurable):** PD decode leg with DPA8 + MTP(steps=3,topk=1) and the
**draft CUDA graph ENABLED** passes warmup + 4×24-token + 1×512-token + conc=128×512
with 0 hangs, matching the Variant-B (draft graph forced eager) result. Any fix must be
localized by measurement, not by "it stopped hanging once".

**Control (对拍):** Variant B — `can_cuda_graph = False` forced in `draft()` — is the
known-good arm. Every round runs BOTH arms unless the round explicitly says otherwise.

**Standing rules:** no deletion of prior workspaces; every round gets its own dir with
CMD / HYPOTHESIS / LOG / RESULT; `.pyc` purge + marker verification on every patch.

---

## Round index

| # | dir | hypothesis / purpose | result |
|---|---|---|---|
| 00 | `r00_setup` | Release stale servers, plan, hold both nodes | done — 11428/11429 idle, VRAM 0% |
| 01 | `r01_instrument` | Log all 4 guard terms per rank/iter; decide H1/H2/H3 | **hang reproduced + cause measured.** `final` diverges on exactly the frozen iteration (it=9): dp2 busy→eager, 7 idle→graph; dp2 is the rank py-spy caught in eager `init_forward_metadata`. H2 refuted (`can_run_graph` uniform). H1 confirmed: t4 flips, t2 makes it asymmetric. Explains mix/PD asymmetry. |
| 02 | `r02_fix` | Vote the eager-need over the full TP group (gloo, 1 elem) | applied; first 4/4 pass with draft graph ON |
| 03 | `r03_verify` | All 5 exit criteria + uniformity/graph-usage measurement | **PASS.** warmup ✓, 4×24 ✓, 1×512 ✓, conc 1/2/4/8/16/128/256 → **927/927**, 0 failures. Over 2992 iterations: LOCAL diverges 38×, **VOTED diverges 0×**, draft graph used **98.4%**, vote flipped a rank 190× (= 190 averted deadlocks). |
| 04 | `r04_control` | 对拍: revert the fix, same node/config/traffic | **hang returns immediately** — 0/4, first request times out at 120 s. py-spy frozen; `final` diverges on exactly it=8 with **dp3** alone eager, and dp3 is the rank stuck in eager `init_forward_metadata`. Victim rank differs from R1 (dp2), confirming a race. |
| 05 | `r05_clean` | Fix only, no instrumentation (the shippable build) | **PASS** — 4/4 + conc 1/8/64/128/256 → **457/457**, 0 failures |
| 06 | `r06_mix` | Regression: does the added collective break single-node mix? | **PASS** — 132/132 (4/4 + conc128×512). First attempt died on the nextn `eh_proj` trap (bad idempotency grep), not the fix. |

(rows appended as rounds complete)

---

## Standing facts carried in (already measured, do not re-derive)

* The hang needs the **draft** CUDA graph specifically. Target-decode and draft-extend
  graphs can stay on (`RESULT_variant_B_draft_graph.md`).
* `--disable-cuda-graph` also fixes it, but is blunt — it disables all three.
* Removing only the `not is_idle()` term from the guard **does not** fix it and moves the
  failure earlier (into warmup). Measured graph/eager split with that fix applied:
  dp0 16/3 vs dp7 14/4 — still rank-divergent.
* "0 all-gathers on the graph path" was a **probe blind spot** (replayed graphs run no
  Python), not a measurement. Retracted.
* Mix (single-node) with DPA+MTP + graphs **passes** 256/256. Only PD hangs. That
  asymmetry is unexplained and is the sharpest lead.
