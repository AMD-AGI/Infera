# Track-F ladder — results

| rung | image | source | launcher | model dir | result |
|---|---|---|---|---|---|
| 0 | vendor rocm724 | bind-mount c821c425 | bare launch_server | original | **PASS** (with --disable-shared-experts-fusion) |
| 1 | ours rocm720 | bind-mount c821c425 | bare launch_server | original | **PASS**, serving after 220s |

Rung 1 evidence: 17*23 -> '391', reasoning_content separated (73 chars),
8 AITER mHC lines (2 per rank x 4), 'Shared experts fusion optimization
enabled' absent (fusion off, as intended).

So the base image is NOT a variable: rocm720 and rocm724 both serve.
The only delta from the vendor's published recipe is the one flag.

| 2 | ours `glm53-c821c425` | baked overlay | bare launch_server | original | **PASS**, 220s |
| 3+4 | ours `glm53-c821c425` | baked overlay | **infera** + etcd + kv-aware router | original | **PASS**, worker 210s, router healthy |

## Rung 3+4 — full infera MIX stack, evidence

Endpoint `http://10.235.192.139:8100` (router). Model `glm5.3-flash-mxfp4`, TP4, GPUs 0-3.

| check | reading |
|---|---|
| router `/v1/workers` | **1** worker, `disagg_mode: mixed` — aggregated, not PD |
| `/v1/models` | `glm5.3-flash-mxfp4` |
| coherent answer via router | yes; 2111 chars of `reasoning_content` separated from `content` |
| `17 * 23` | `391`, obeying "reply with only the number" |
| AITER mHC lines | **8** = 2 per rank x 4 — the fast path is live |
| `Shared experts fusion optimization enabled` | **0** — fusion off, as required |
| both memory pools | 18 decode lines carrying `full token usage` AND `mamba usage` |
| `memory access fault` / `HIP error` | 0 / 0 |
| Traceback (excluding torch._dynamo telemetry) | 0 |
| router policy | `kv-aware` |

`max_total_num_tokens=7650368`, Mamba cache `ssm_state 78.76GB`, `available_gpu_mem=55.47GB`.

## What the ladder established

Every rung passed once `--disable-shared-experts-fusion` was added, so NONE of
these is a variable: the base image (rocm720 vs the vendor's rocm724), the
overlay delivery (bind-mount vs baked), the infera wrapper, etcd, or the
kv-aware router. The single delta from the vendor's published recipe is that one
flag, and the reason is a genuine engine bug — see `results/root_cause.md`.
