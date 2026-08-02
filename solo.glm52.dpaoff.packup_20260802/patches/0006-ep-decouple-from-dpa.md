# Patch 0006 — `EP_DECOUPLE`: make `--ep-size` independent of DP-attention

**Target:** `scripts/glm52_leg.sh` (runs inside the engine container)
**Applied by:** `scripts/patch_leg_epsize.py` (idempotent, anchors on exact text)
**Status:** REQUIRED for this experiment. Without it the run is invalid.

## What

Hoist `--ep-size "$TP"` out of the `if [ "$DPA" = "1" ]` branch so it is passed
unconditionally.

```diff
 DP_ARGS=()
+DP_ARGS+=(--ep-size "$TP")
 if [ "$DPA" = "1" ]; then
-  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention --ep-size "$TP")
+  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention)
```

## Why

This experiment must vary **exactly one** thing: how attention is sharded
(DP8 → pure TP8). As shipped, `--ep-size` lived inside the DPA branch, so
`DPA=0` silently dropped the flag and sglang resolved **`ep_size` 8 → 1**.

That would have collapsed MoE expert-parallelism at the same moment as
attention DP — a **two-variable change**, and the measured 2× TTFT improvement
could not have been attributed to DPA at all.

Verified before launching, not after:

```
$ prepare_server_args(<base args, no --ep-size>)
NO --ep-size flag  -> ep_size = 1  dp= 1  dpa= False
```

## How it was applied

```bash
cd $W/scripts
cp glm52_leg.sh glm52_leg.sh.bak_dpaon_$(date -u +%Y%m%d-%H%M)
python3 patch_leg_epsize.py glm52_leg.sh
# expect: patched OK - EP_DECOUPLE occurrences: 1
```

## Context — proving the DPA-on path is unchanged

The edit touches a line that the DPA-on configuration also executes, so "this
only affects DPA=0" is a claim that had to be tested. Both versions were
dry-run with `DPA=1` and their emitted argument lists compared:

```
OLD : --dp-size 8 --enable-dp-attention --ep-size 8 --enable-prefill-delayer ...
NEW : --ep-size 8 --dp-size 8 --enable-dp-attention --enable-prefill-delayer ...
```

Same flags, same values; order differs, which argparse does not care about.
The DPA-on baseline this experiment compares against therefore remains a valid
control.

## Symptom it cures

None visible at runtime — that is precisely the danger. With `ep_size=1` the
leg still boots and still serves. It would simply have been a different
deployment, and the headline number would have been silently wrong.
