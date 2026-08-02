# Patch 0007 — `DPA_PASSTHROUGH`: stop hardcoding `DPA=1` in the launcher

**Target:** `scripts/start_leg.sh` (runs on the node, outside the container)
**Applied by:** inline edit (see below); default behaviour unchanged
**Status:** REQUIRED for this experiment. Without it `DPA=0` does nothing.

## What

`start_leg.sh` built the `docker exec ... env` block with a **literal** `DPA=1`,
so the value never came from the caller's environment.

```diff
+# DPA_PASSTHROUGH: DP-attention was hardcoded to 1 in the docker-exec env
+# block, so an outer `DPA=0` was silently ignored and the leg came up with
+# --enable-dp-attention anyway. Default stays 1 (unchanged behaviour); the
+# DPA-off latency experiment sets DPA=0.
+DPA="${DPA:-1}"
 DSA_ROWS="${DSA_ROWS:-0}"
 ...
-  CTX=262144 ISL=8192 TP=8 DPA=1 CUDA_GRAPH_BS=128 MAX_RUNNING=2048 \
+  CTX=262144 ISL=8192 TP=8 DPA="$DPA" CUDA_GRAPH_BS=128 MAX_RUNNING=2048 \
```

Default remains `1`, so every prior run's behaviour is bit-for-bit unchanged.

## Why

The first launch of this experiment used `DPA=0 bash scripts/start_leg.sh`. The
launcher printed its usual success line:

```
[p7] prefill launched on chi2879 mtp=0 ctx=262144 gmu=0.80 -> .../p7_prefill.log
```

and the leg came up **with DP-attention still enabled**. The outer variable was
shadowed by the hardcoded one and discarded without warning.

## How it was caught — the generalisable lesson

By reading back the **live process command line** after launch instead of
trusting the launcher's echo:

```bash
ssh <node> 'ps -eo pid,lstart,cmd | grep "[s]glang.launch_server" | head -1'
```

The output still contained `--dp-size 8 --enable-dp-attention`. Had this not
been checked, the run would have produced 105 clean, plausible, **completely
meaningless** samples — a second copy of the DPA-on baseline, reported as the
DPA-off result.

**A launcher's success message is not evidence that the launcher did what you
asked.** Verify the process, not the wrapper. This is the same class of error as
"verify the loaded module, not the file on disk" (`__pycache__` staleness) that
has invalidated an experiment in this tree before.

## Symptom it cures

`DPA=0` being silently ignored. Post-fix the command line reads:

```
... --hicache-size 16 --ep-size 8 --disaggregation-mode prefill ...
```

with **no** `--dp-size` and **no** `--enable-dp-attention`, and the engine logs
`DSA with TP mode is active, dp_size=1, tp_size=8` plus per-rank lines prefixed
`TP0..TP7` rather than `DP0 TP0`.
