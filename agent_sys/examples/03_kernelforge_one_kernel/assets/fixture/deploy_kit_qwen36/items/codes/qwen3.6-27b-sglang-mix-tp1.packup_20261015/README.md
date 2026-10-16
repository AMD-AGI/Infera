# Qwen3.6-27B · SGLang through infera · MIX · TP1 · one MI300X

## What was asked

The task was to bring up `Qwen3.6-27B` in `mix` mode on `smc300x-ccs-aus-a16-19`
at TP1 and context 32768, on card 6 as named by the operator, with
`--reasoning-parser qwen3` and no DSA flags. It had to be proven to answer
through the router and handed back as a kit somebody else can run. The image was
`infera:sglang-gfx942-glm52` and the weights were already at
`/data/yihou_hf_cache/models/Qwen3.6-27B` (52G, 15 safetensors shards).

## What was run

All times are UTC on 2026-10-15. Every bring-up below went through this kit's
own `scripts/deploy.sh`, so the scripts are byte for byte what ran.

1. **Preflight** at 06:50:12Z. The image was present as a `sha256:` digest, so
   nothing was built. `rocm-smi --json` listed 8 cards, each with 284 MiB used.
   The ports in band 18101–18106 were free (`results/preflight.json`).
2. **Deployment `r16aea3`**, band 18101–18106, card 6. It started at 06:52:53Z
   and was READY through the router at 06:57:29Z, **276 s** on the first
   attempt (`logs/deploy.log`).
3. **Read-back and proof** (`scripts/verify.sh`) at 06:57:50Z. Then the
   validator's 11 diagnostic probes, then a 180 s load at concurrency 16 with
   1k in and 1k out through the router.
4. **Teardown** of `r16aea3` at 07:02:22Z (`results/teardown_first_deployment.json`).
5. **Callability self-test**, tag `selftest-r16aea3`, band 18121–18126, its own
   work root, the same card 6, with all three seams set. It was READY in 276 s
   at 07:13:18Z. Collision and isolation checks ran against it, then it was
   torn down at 07:14:13Z (`results/selftest/`).
6. **Final teardown proof** under the default tag (`results/teardown.json`).

## Result

**It works.** The checks below are from the running system, each with its file.

| criterion | reading | evidence |
|---|---|---|
| router `/health` | HTTP 200, `{"status":"ok","active_workers":1}` | `results/verification.json` |
| worker registered, mode (router's reading) | 1 worker, `"status": "active"`, `"disagg_mode": "mixed"` | `results/router_workers.json` |
| mode (worker's own reading) | `worker ready: … disagg=DisaggMode.MIXED` | `results/worker_mode_line.txt` |
| served name | `/v1/models` = `["Qwen3.6-27B"]`; no id starts with `/` | `results/router_models.json` |
| completion through the router | `finish_reason: "stop"`, `content: "\n\nParis"`, 157 reasoning tokens kept in `reasoning_content`, `model: "Qwen3.6-27B"` | `results/chat_completion.json` |
| validator probe set (`probes.yaml`) | 11 of 11 pass, including streaming, `/health_generate` and context-limit refusal | `results/probe_results.json` |
| load, 1k/1k, conc 16, 180 s | 112 requests, 0 errors, mean OSL 1024.04. ITL 23.92 ms avg (p99 25.4). TTFT 2347 ms avg (p99 4501). 610.7 output tok/s | `results/load_conc16_summary.json` |
| CUDA graph ceiling covers the load | ceiling 32 ≥ effective decode concurrency 14.593 (max 16); all 193 logged decode batches `cuda graph: True` | `results/graph_ceiling_and_load_judgement.json` |
| kit callable by a program | second deployment with a different tag, port band and work root came up, wrote its handshake, passed `verify.sh`, and was removed by `teardown.sh` | `results/selftest/` |
| seams | `ENGINE_EXTRA_ENV` reached the worker and its sglang child, not the router or the container. `ENGINE_EXTRA_ARGS` and `ROUTER_EXTRA_ARGS` are the last argv tokens | `results/selftest/seams_and_mounts.json` |
| measurement mounts | a file written inside the container appeared on the host for the work root, the run root and the scratch root | `results/selftest/seams_and_mounts.json` |
| torn down | 0 containers with either run label, none of the 12 band ports listening, card 6 back to 285 MiB | `results/teardown.json`, `results/teardown_selftest.json` |

The deployment came up on the first attempt both times. The self-test was
refused once by `deploy.sh` itself before it started anything: a co-tenant had
taken all eight cards. It succeeded once card 6 was free again (`notes.md`).

## Where things are

- `scripts/`: `deploy.sh`, `wait_ready.sh` and `teardown.sh` (the contract),
  `env.sh` (every parameter), `worker.sh` and `router.sh` (run inside the
  container), `verify.sh` (the read-back and completion proof),
  `lib_mounts.sh` and `gpu_used.sh`.
- `results/`: the JSON evidence named above.
- `logs/`: worker, router and etcd logs, the engine and router argv as read
  from `/proc`, the deploy transcripts, and the refused first self-test attempt.
- `REPRODUCE.md`: commands and expected output. `notes.md`: decisions, the
  numbers behind them, and traps. `environment.md`: the record, rendered.
