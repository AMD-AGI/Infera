# Workloads and traces

```{admonition} One-pager
:class: tip
**What:** the three ways to tell the simulator what traffic to run.
**Why:** a synthetic Poisson stream and your production trace produce very
different cache behaviour, and the difference is usually the answer.
**Cost:** none — pick whichever source you have.
```

Every source funnels into the same thing: a list of requests, each with an
arrival time, a prompt length, an output length and — where the source supplies
it — an ordered sequence of block-hash ids. The scheduler and cost kernel do not
know which source produced them.

| Source | Flag | Arrivals come from | Prefix reuse comes from |
|---|---|---|---|
| **Synthetic arrivals** | `--arrival-model` + `--request-rate` | a generated process (Poisson, deterministic, gamma-bursty) | a synthetic prefix pool you configure |
| **Workload file** | `--des-workload-file` | the file's `arrival` column | nothing — no hash ids, so no content-addressed reuse |
| **Mooncake trace** | `--des-mooncake-trace` | the trace's own `timestamp` | the trace's real `hash_ids` |

## Synthetic arrivals

The default. You state the offered rate and the shape of the arrival process,
and lengths are sampled around the configured input/output lengths.

```bash
inferasim inference ... \
  --request-rate 1.5 --arrival-model poisson --des-num-requests 120 \
  --des-burstiness 0.5 --des-range-ratio 0.3
```

`--des-burstiness` makes arrivals gamma-distributed (`1.0` is Poisson, lower is
burstier); `--des-range-ratio` spreads per-request lengths rather than making
every request identical. Both matter for tails: a homogeneous workload arriving
on a smooth process understates p99 because nothing ever collides.

### Synthetic prefix pools

To get reuse without a trace, declare a pool of shared prefixes and let requests
draw from it:

```bash
  --des-num-prefixes 8 --des-prefix-len 2048 --des-block-size 512 \
  --des-prefix-zipf 1.1
```

Each request is assigned a prefix id, and its leading blocks are that prefix's
blocks — so reuse is *realised* through the block cache rather than asserted as
a rate. `--des-prefix-zipf` skews popularity the way a few hot system prompts
dominate real traffic; omit it (or pass `0`) for uniform popularity.

This is the synthetic counterpart to `--prefix-cache-hit-rate` on the
[analytical path](projection_runs.md#declare-prefix-reuse). The analytical flag
*states* a hit rate; the pool *produces* one, and the produced number depends on
capacity and routing. They are different knobs and do not substitute for each
other.

## Workload file

For replaying a recorded length-and-arrival sequence that has no cache
information:

```bash
inferasim inference ... --des-workload-file workload.csv
```

JSON (a list of objects) or CSV (a header row). Keys and columns are
case-insensitive:

| Field | Aliases | Meaning | Missing |
|---|---|---|---|
| `arrival` | `arrival_ms`, `time`, `step` | arrival time in milliseconds | `0` |
| `isl` | `input_len`, `prompt_len` | prompt length in tokens | `1` |
| `osl` | `output_len` | output length in tokens | `1` |

```csv
arrival,isl,osl
0,3412,512
118,2980,640
250,7711,256
```

Because there are no hash ids, the block cache has nothing to match on — expect
a 0% hit rate. Use this source when the question is about queueing and length
heterogeneity, and a trace when it is about cache reuse.

## Mooncake trace

The high-fidelity source. A Mooncake trace is JSON-lines (one object per line)
or a JSON array, where each record carries `timestamp` in milliseconds,
`input_length`, `output_length` and `hash_ids` — the ordered list of block-hash
ids for the prompt.

```bash
inferasim inference ... \
  --des-mooncake-trace trace.jsonl \
  --des-instances 4 --des-routing kv --des-block-size 512
```

```json
{"timestamp": 0,   "input_length": 3412, "output_length": 512, "hash_ids": [11, 12, 13, 41]}
{"timestamp": 118, "input_length": 2980, "output_length": 640, "hash_ids": [11, 12, 13, 77]}
```

Requests sharing a system prompt share leading `hash_ids`, which drives genuine
content-addressed reuse instead of an assumed rate. The two records above share
three leading blocks, so the second one's prefill skips them if they are still
resident on the replica it lands on.

The trace supplies its own arrivals, so `--request-rate` is not needed. Field
names are matched case-insensitively and common aliases are accepted
(`arrival`/`time` for the timestamp, `block_hashes`/`blocks` for the hash ids).

## Which source to use

- **You are sizing for a latency SLO under a stated load.** Synthetic arrivals,
  with `--des-burstiness` set to something pessimistic.
- **You are choosing a routing policy or a cache capacity.** A Mooncake trace.
  Routing decisions are decisions about content locality, and a synthetic pool
  can only tell you what you already assumed.
- **You have production lengths but no hashes.** A workload file, and treat the
  cache numbers as absent rather than as zero.

## Next steps

- Route the traffic across several replicas: [Fleet and routing](fleet_and_routing.md).
- Read the resulting percentiles: [Simulation runs](simulation_runs.md#reading-the-report).
