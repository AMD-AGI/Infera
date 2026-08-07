# Scripts

Every file here is a **verbatim copy** of what ran on the cluster at
`/mnt/vast/c_huggingface/glm52_mix_20260806/scripts/`, md5-verified (table in
`../environment.md`). Byte-level flags survive.

## Used by THIS phase

| script | role |
|---|---|
| `run_agentic.sh` | **the entry point.** `WORKLOAD=<yaml> TAG=<name> bash run_agentic.sh` — launches one agentic run against the router. Passes **no load knobs**: the YAML is the single source of truth. `--dashboard-mode` is mandatory or nothing is persisted. |
| `analyze_solo.py` | **the analysis.** `python3 analyze_solo.py <result dir> [label]` — sustain-phase-only per-request TTFT / E2E / TPOT, dropping SOLO_M1's `0.0` filtered markers. Runs offline on the shipped results; needs no cluster. |
| `envsnap.sh` | on-node environment snapshot. Must be run **while the deployment is live** — the resolved engine cmdline and kvd counters cannot be recovered later. Output: `../env/env_chi2835.txt`. |
| `accept_len.sh` | MTP acceptance distribution off an engine log. **Unscoped** — it reads the whole file. These logs are appended across all three phases, so scope it by time window yourself (`../REPRODUCE.md` §8) unless you know the log belongs to one run. |

## Deployment scripts — shared with Phase 1

The deployment was brought up **once** and all three phases ran on it. These are
the scripts that stood it up; they are carried so this packup can reproduce from
bare metal without reaching into the Phase-1 packup.

| script | role |
|---|---|
| `mix_site.sh` | **the only file carrying site values** (`MY_IP`, `MODEL`). Verbs: `up`, `smoke`, `down`. Edit this, not the others. |
| `mix_up.sh` | the 5-stage bring-up: container → etcd → kvd → mix worker → router. |
| `mix_engine.sh` | the mix worker launch — `python3 -m infera.engine.sglang`, TP8, the DSA-on-ROCm env block, all engine flags. |
| `mix_common.sh` | shared helpers: `reap` (waits for VRAM to drain, does not just kill), `wait_health`, `start_router`. |
| `mix_smoke.sh` | the feature gate — the seven checks in `../REPRODUCE.md` §3. Read the blocks, not the exit code. |

## Not carried

`mix_bench_fixlen.sh` and `summarize_fixlen.py` belong to **Phase 1** and are not
in this packup. They are in `../fixlen.glm52.mix.packup_20260806/scripts/`.

`specs/mix_load.yaml` belongs to **Phase 3** and is likewise not carried here.

## The driver itself is NOT in this packup

`run_agentic.sh` invokes a staged copy of `Optimus-AgenticBench` on the shared
mount, at commit `1cf01cb` **plus the SOLO_M1 patch**. See `../environment.md`
§ "The benchmark driver" and `../patches/README.md`. The patch is load-bearing:
without it `analyze_solo.py` has no `new_e2es` to read.
