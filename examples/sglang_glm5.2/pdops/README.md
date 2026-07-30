# pdops — PD bring-up tooling, kv-aware diagnostics, and the runs behind the reports

Everything here was written or captured while bringing up GLM-5.2-FP8 1P1D on MI325X
and getting kv-aware routing to work on PD + DPA + MTP. It is checked in for two
reasons: the diagnostics are the fastest way to re-localise the same class of failure,
and the result files are the evidence behind the numbers in
[`KV_AWARE.zh.md`](../KV_AWARE.zh.md) and [`REPORT.zh.md`](../REPORT.zh.md) — the cluster
nodes are ephemeral, so anything only living in `/tmp` on a node is one reclaim away
from being gone.

These are debugging tools, not part of the product. They hardcode this cluster's IPs
and model path as *defaults* only; every one takes them as arguments or env vars.

**No keys are in git.** `start_agent.sh` needs an ssh keypair you generate yourself
(see its header); the private keys used during the session were deliberately left out.

## Cluster ops

| File | What it does | When you need it |
| --- | --- | --- |
| `podenv.sh` | Exports the container's real environment by reading `/proc/1/environ`. | **Source this first in every ssh session into a pod.** A non-interactive ssh shell does not inherit the image's ~15 `SGLANG_*` / `AITER_*` vars, and dropping any of them silently changes engine behaviour — e.g. `SGLANG_USE_AITER` decides the DSA page size, so losing it makes the two PD legs disagree on the KV layout. |
| `start_agent.sh` | Starts an sshd (default port 2223) inside a worker pod so the master pod can drive it. Run once per pod. | Multi-node work where you only get a terminal into one pod. `KEYDIR=` to point at the keypair, `PORT=` to move the port. |

## kv-aware diagnostics

The two bugs behind "router cache view stays empty" are an event **transport** problem
and an event **decode** problem, and they look identical from the outside. These scripts
split the chain apart. Full write-up with symptoms and root causes:
[`KV_AWARE.zh.md`](../KV_AWARE.zh.md) §1.

| File | Question it answers | Usage |
| --- | --- | --- |
| `zmq_kv_probe.py` | Is the engine publishing at all? An independent subscriber on every DP rank's port, listening while a request is in flight — if it sees frames the fault is downstream in the router, if not it is in the engine. | `python3 zmq_kv_probe.py [host] [base_port] [ranks]` |
| `zmq_dual_probe.py` | Is it address-dependent? Subscribes over loopback **and** the node IP in one process during one request. Two sequential single-address runs cannot tell "wrong address" from "missed the publish window", because frames only appear while a request is in flight — this removes that ambiguity. **This is the script that found bug 1.** | `python3 zmq_dual_probe.py 127.0.0.1,10.32.17.210 29992 8` |
| `zmq_host_matrix.py` | Same address question with no engine involved: plain PUB→SUB over the (address, payload size) matrix. | `python3 zmq_host_matrix.py [port] [hosts]` |
| `zmq_pub_isolated.py` | Is SGLang's `ZmqEventPublisher` class itself broken? Drives it directly. | `python3 zmq_pub_isolated.py [port]` |
| `zmq_pub_ordering.py` | Does the *engine's* ordering matter — publisher and its thread idling for minutes before any subscriber connects, which is the opposite of the isolated test? **This is what falsified the ZMQ thread-safety theory**: frames arrive fine. | `python3 zmq_pub_ordering.py [port]` |

For the routine "is kv-aware working right now" check, use
[`../verify_kv_aware.py`](../verify_kv_aware.py) instead — one command, asserts the whole
contract (miss, view fills, repeat is a full prefix hit).

## `glm52_suite/` — verbatim copies from a non-versioned sibling directory

Copied from `/wekafs/llying/code/inference_glm5p2_sglang`, which is **not a git repo** and
would be lost with the node. `README.md` and `REPORT.zh.md` both reference these as the
validated aggregated (single-node) recipe that PD is compared against, so they need to
survive. Not modified here; if you change them, change them upstream.

| File | What it is |
| --- | --- |
| `verify_correctness.py` | The correctness suite used throughout: weights, basic, determinism, idle, needle-in-haystack, HumanEval (short and long context), code-retrieval, deep-api. Point it at the router to cover the full kv-aware path: `python3 verify_correctness.py --base-url http://127.0.0.1:8000 --model /wekafs/models/GLM-5.2-FP8 --json-out out.json` |
| `run_sglang_mtp.sh` | Aggregated single-node launch with MTP — the recipe PD's engine flags were derived from. |
| `run_sglang.sh` | Same, without MTP. |
| `KNOWN_ISSUES.md` | Engine-level issues found on this stack before PD work started. |
| `repro_token_corruption.py` | Minimal reproducer for the token corruption issue described in `KNOWN_ISSUES.md`. |

## `results/` — the runs the reports cite

| Path | What it is |
| --- | --- |
| `verify_correctness_kvaware_pd_mtp.json` | Correctness suite on PD + DPA + MTP + kv-aware, all 9 sections passing (`humaneval-long` 19/20, `determinism` 1/1). Cited in `KV_AWARE.zh.md` §2.4. |
| `agentic_1_baseline_broken_view/` | Agentic bench baseline: broken router view, cache **not** flushed. 47.2% hit, 49.9% efficiency, TTFT p50 16.9s. |
| `agentic_2_control_blind_view/` | Control group: broken view, cache flushed. 54.3% hit, 57.3% efficiency, TTFT p50 14.6s. Isolates "the fix helped" from "the empty cache helped". |
| `agentic_3_kvaware_fixed/` | Both bugs fixed, cache flushed. 83.6% hit, 88.2% efficiency, TTFT p50 2.5s. |

All three are 60 requests of `code_agent_128k.yaml` at identical seed and token totals
(`input_tokens=3,169,196`, `prefix_tokens=3,002,633`), so they are directly comparable —
see `KV_AWARE.zh.md` §2.5 for the exact command and §3 for the attribution.

Each run directory holds `summary.json` (aggregate metrics), `metadata.json` (the
effective workload parameters) and `metrics.jsonl.gz` (per-window time series: session
counts, per-request prompt lengths, TTFTs). The time series is gzipped — 660 KB each raw,
30 KB compressed. Read it with:

```bash
zcat results/agentic_3_kvaware_fixed/metrics.jsonl.gz | python3 -c "
import json, sys
rows = [json.loads(l) for l in sys.stdin if l.strip()]
print('sessions:', rows[-1]['num_sessions_total'])
pl = [p for r in rows for p in r.get('new_prompt_lengths', [])]
print('requests:', len(pl), 'mean prompt:', sum(pl) // len(pl))
"
```

Reproduce a run with [Optimus-AgenticBench](https://github.com/AMD-AGI/Optimus-AgenticBench);
the exact invocation is in `KV_AWARE.zh.md` §2.5.
