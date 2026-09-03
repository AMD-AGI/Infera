# scripts/ — verbatim, with what to trust and what to change

All three ran exactly as shipped here. They were adapted from
`../glm53flash.mix.packup_20260830/scripts/` (the TP8 FP8 bring-up); those
originals were **not** modified.

| script | what it does | trust |
|---|---|---|
| `mix_up.sh` | teardown → VRAM measurement → container → etcd → worker → router | **ran as-is.** Node-specific values at the top |
| `mix_worker.sh` | the engine launcher, runs **inside** the container | **ran as-is.** Carries the FP8 recipe |
| `verify.sh` | the nine-block correctness + health battery | **ran as-is.** Runs on the host, talks in via `docker exec` |

## What is node-specific and must be changed

| variable | value used | why it is not portable |
|---|---|---|
| `GPUS` / `TP` | `4,5,6,7` / `4` | half the node belonged to a colleague |
| ports | 31400 / 23795 / 18105 / 15570 / 18801 | 2379 and 18100 were **busy** on this host |
| `SELF` | hardcoded to the workspace dir | deliberately **not** `BASH_SOURCE` — see below |
| `MODEL_HOST` | `/apps/data/models/GLM-5.3-Flash` | resolved with `readlink -f` at runtime; the symlink crosses NFS mounts |
| `CEIL` in the VRAM block | **0.60** | vendor says 0.80. Lowered because we were guests. On a node you own, raise it |

## Three deliberate design choices, each bought with a failure

1. **Teardown hard-refuses foreign names.** The loop `case "$name" in yihou_f8_*)
   ... ;; *) echo REFUSING; exit 1 ;; esac` is not decoration — this ran on a box
   with four of a colleague's containers live. `reset_gpus.sh` is **deliberately
   not called** and is not shipped here.

2. **`--mem-fraction-static` is computed, never hardcoded.** It is a fraction of
   **TOTAL** GPU memory, not free memory, so a hardcoded 0.80 on a node where the
   neighbour holds 18 % will OOM *their* job. The script reads
   `rocm-smi --showmeminfo vram --json`, takes the worst used fraction across the
   target cards, and uses `min(CEIL, 0.85 - worst)`, aborting below 0.45. **It
   fired for real**: a foreign container took 161 GiB on two cards and the script
   computed 0.29 and refused to launch rather than competing.

3. **`SELF` is a hardcoded path, not `$(dirname "${BASH_SOURCE[0]}")`.** Because
   the script must be run from a `/tmp` copy — editing it on NFS while bash is
   still reading it corrupts the rest of the run (`../notes.md` §3b) — and the
   `BASH_SOURCE` form would then look for `mix_worker.sh` in `/tmp` and fail with
   `lstat /tmp/mix_worker.sh: no such file or directory`. That exact failure cost
   one launch.

## The recipe, and how it differs from the MXFP4 sibling

`mix_worker.sh` passes, for **FP8**:

```
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang
--kv-cache-dtype bfloat16  --moe-runner-backend triton
--reasoning-parser glm45   --tool-call-parser glm47
env SGLANG_USE_AITER=1
```

and deliberately does **not** pass:

| flag | why not |
|---|---|
| `--quantization quark` | `config.json` already declares e4m3; the engine resolves `Fp8MoEMethod` on its own. That flag belongs to the MXFP4 lane |
| `--disable-shared-experts-fusion` | **the point of this run.** 129/129 index pairing ⇒ uniform block-FP8 ⇒ fusion is legitimate. `DISABLE_SEF=1` is wired as a one-variable fallback and was **not needed** |
| any `--speculative-*` | MTP is unvalidated for this model on ROCm |
| `--nsa-*` | that is GLM-5.2's spelling; this model uses `--dsa-*` |

## Not shipped

- `collect_env.sh` — **was not run**; `../environment.md` records that gap.
- `bench.sh` / any sweep script — **this operator did not benchmark**, by
  decision (`../notes.md` §5). The sweep in `../results/` was run by the **team
  lead** with their own harness invocation; it is recorded there verbatim from
  the run's own `benchmark_args` rather than reconstructed as a script we did
  not write.
- `reset_gpus.sh` — excluded on purpose; see above.
- The IPC probe is inline in `../results/ipc_probe.md` rather than a script here,
  because it is two heredocs and needs no model.
