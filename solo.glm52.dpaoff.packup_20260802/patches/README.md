# Patches

Four, at three layers. Two are inherited from the DPA-on baseline kit; **two are
new to this experiment and exist to make the comparison valid.**

| | target | applied where | without it |
|---|---|---|---|
| **0004** `GLM52_P1V3` | engine, decode container | runtime, per boot | decode leg **crashes** within minutes |
| **0005** `SOLO_M1` | driver, jump host | once, to the staged repo | **no E2E / TPOT ladders** |
| **0006** `EP_DECOUPLE` | `glm52_leg.sh` | before launching prefill | run becomes **two-variable**, result unattributable |
| **0007** `DPA_PASSTHROUGH` | `start_leg.sh` | before launching prefill | `DPA=0` **silently ignored**, you re-measure the baseline |

The two new ones share a theme worth internalising: **neither failure is
visible at runtime.** Both produce a leg that boots, serves, and yields a clean
105-sample run — just of the wrong deployment. They are caught only by checking
resolved arguments and the live process command line *before* spending the
window.

---

## 0004 — `dsa_indexer.py`: handle the REVERSED padding case (`GLM52_P1V3`)

**Inherited, unchanged, and still live on the decode leg.** This experiment did
**not** restart decode (PID 2420132, up since 2026-08-01 13:29:28), so the patch
carried over from the baseline run. Full write-up:
`../notes/notes.dsa.mtp.crash.md`.

**Applied by** `../scripts/apply_p1v3.py`, inside the **decode** container.
Verified `P1V3: 3` in the loaded module before this window opened.

**One-line why.** The image's own `GLM52_P1V2` trim guards only
`real < padded`. On a DP-attention **IDLE** rank under MTP draft-extend the
inequality inverts, no trim runs, and `fast_topk_v2` raises
`Expected lengths.size(0) == B`.

> Note for anyone reproducing DPA-**off** on the *decode* leg: this bug needs an
> idle DP rank, which pure TP does not have. That is a hypothesis, not a
> measurement — decode ran with DPA **on** throughout this experiment and the
> patch stayed loaded. Do not drop it on the strength of that reasoning.

## 0005 — driver persists per-request E2E + TPOT (`SOLO_M1`)

**Inherited, unchanged.** Full write-up: `0005-driver-persist-e2e-tpot.py.txt`.

Applied by `../scripts/apply_solo_metrics.py` on the jump host. Adds
`new_e2es` / `new_tpots` to `metrics.jsonl`.

**The trap it encodes:** `actual_tpots` is *filtered* (appended only when
`gen_len > 1 and gen_time >= 50 ms`), so it is **not index-aligned** with
`actual_ttfts` and must never be sliced with the same cursor. The patch adds a
separate always-appended `actual_tpots_aligned` where filtered entries are
`0.0`. This run: **0 filtered**, alignment verified `ttft=e2e=tpot=105`.

## 0006 — `EP_DECOUPLE` → `0006-ep-decouple-from-dpa.md`

Hoists `--ep-size` out of the DPA branch. Without it `DPA=0` also collapses
`ep_size` 8 → 1, changing MoE expert-parallelism simultaneously with attention.

## 0007 — `DPA_PASSTHROUGH` → `0007-dpa-passthrough.md`

`start_leg.sh` hardcoded `DPA=1` into the container env block. Without it the
prefill leg comes up with `--enable-dp-attention` no matter what you pass.
