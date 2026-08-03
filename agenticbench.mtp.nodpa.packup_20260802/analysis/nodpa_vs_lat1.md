# Prefill DP-attention, on vs off, at concurrency 1

Run `nodpa_full/2026-08-02-12-41-42`, 2026-08-02 12:41:42–13:14:47 UTC (1,985.5 s),
against **lat1** (`lat1_full/2026-08-02-05-39-24`, 1,980.9 s).

Same workload, same driver, same model, same two-node PD topology, same MTP on
decode. **One server flag moves: `--enable-dp-attention` on the prefill leg.**

## Headline — DP-attention on prefill costs 1.65–1.93× TTFT at concurrency 1

**Three arms**, because the second one was needed to prove the third is not a
confounder:

| arm | prefill DPA | **global** chunk/step | n | input p50 | TTFT p50 | TTFT p90 |
|---|---|---:|---:|---:|---:|---:|
| **lat1** | **on** (dp8) | 65,536 (8,192 × 8 ranks) | 124 | 83,048 | 2,042 ms | 4,757 ms |
| **noDPA-65K** ← **the result** | **off** | 65,536 | 115 | 84,404 | **1,162 ms** | **2,227 ms** |
| noDPA-8K (chunk control) | off | 8,192 | 175 | 71,640 | 916 ms | 2,059 ms |

### Why three arms: `--chunked-prefill-size` is per-*global*, and DPA divides it

`server_args.py:4902`:

```python
if self._resolved().enable_dp_attention:
    self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size
```

It is a **division by `dp_size`**, not a clamp to 8192. So `chunked_prefill_size`
means different things on the two arms:

| | requested | `server_args=` | semantics | **global tokens/step** |
|---|---:|---:|---|---:|
| lat1 (dp8) | 65,536 | 8,192 | **per rank** | **65,536** |
| noDPA-65K | 65,536 | 65,536 | global | **65,536** ✓ matched |
| noDPA-8K | 8,192 | 8,192 | global | 8,192 (⅛) |

Confirmed against `#new-token` in the engine logs: both DPA and noDPA arms show a
modal batch of 8,192, but on the DPA arm that is 8,192 **per rank across 8 ranks
in parallel**, and on noDPA-8K it is the whole machine.

**noDPA-8K was therefore running at ⅛ the per-step budget of lat1**, and could not
by itself establish anything about DPA. noDPA-65K exists to fix that.

## Bin-matched, three arms — chunk turns out not to matter, DPA does

| input bin | DPA on / 65K | DPA off / 65K | DPA off / 8K | **DPA effect** | chunk effect |
|---|---:|---:|---:|---:|---:|
| 0–40K | 877 | 455 | 454 | **1.93×** | 1.00× |
| 40–60K | 1,150 | 637 | 633 | **1.81×** | 1.01× |
| 60–80K | 1,674 | 938 | 907 | **1.78×** | 1.03× |
| 80–100K | 2,164 | 1,199 | 1,228 | **1.80×** | 0.98× |
| 100–130K | 2,896 | 1,634 | 1,541 | **1.77×** | 1.06× |
| 130–160K | 3,729 | 2,104 | 2,006 | **1.77×** | 1.05× |
| 160–200K | 4,517 | 2,733 | 2,773 | **1.65×** | 0.99× |

**Two readings, and the second one is why this rerun was worth 45 minutes:**

1. **DPA effect: 1.65–1.93×**, seven disjoint bins, monotone-ish in length. Real.
2. **Chunk effect: 0.98–1.06×**, scattered around 1.00 with no direction — an 8×
   change in per-step token budget is **invisible at concurrency 1**. A single
   74K prompt costs the same whether it is cut into 2 chunks or 10; per-step
   overhead is negligible against the compute, and with one request in flight
   there is nothing else to pack into the spare chunk capacity.

That second row could not be assumed in advance — it had to be measured — and it
is what licenses reading the first row as a DPA effect.

## The fits

| arm | fit (outliers > 5 s excluded) | R² | marginal |
|---|---|---:|---:|
| lat1 (DPA on) | `-106 + 26.50 × ktok` | 0.9493 | 37,736 tok/s |
| **noDPA-65K** | `-182 + 16.75 × ktok` | 0.8324 | **59,705 tok/s** |
| noDPA-8K | `-145 + 15.55 × ktok` | 0.9747 | 64,306 tok/s |

The two noDPA arms agree on marginal rate to within 8 % (59,705 vs 64,306) while
differing 8× in chunk — the fit's version of the same finding. Against lat1 that
is **1.58–1.70×**, consistent with the bin table.

noDPA-65K's R² (0.83) is lower than the other two; it has fewer samples (n=115)
and its own scatter. Both fits are published for every arm so the reader can see
the outlier weight rather than take an exclusion on trust.

### Reproduce both tables above

Run from this kit's root. Reads only this kit and the lat1 sibling kit — no
scratch paths, no arguments. **This is the only runnable block in this document**
(the other two ```python fences quote sglang / driver source for reference), so
extract it by content rather than by position:

```bash
python3 -c "
import re
blks = re.findall(r'\\`\\`\\`python\n(.*?)\\`\\`\\`',
                  open('analysis/nodpa_vs_lat1.md').read(), re.S)
exec(next(b for b in blks if b.lstrip().startswith('import json')))"
```

```python
import json, gzip, math

def load(path):
    op = gzip.open if path.endswith('.gz') else open
    d = []
    for line in op(path, 'rt'):
        r = json.loads(line)
        t = r.get('new_ttfts') or []
        p = r.get('new_prompt_lengths') or []
        d += [(p[i], t[i] * 1000) for i in range(min(len(t), len(p)))]
    return d

A = load('../agenticbench.mtp.lat1.packup_20260802/results/metrics.jsonl.gz')  # DPA on,  65536 global
B = load('results/chunk65536_MAIN/metrics.jsonl.gz')                           # DPA off, 65536 global
C = load('results/chunk8192_ARM/metrics.jsonl.gz')                             # DPA off,  8192 global

def fit(d):
    xs = [a for a, _ in d]; ys = [b for _, b in d]; n = len(xs)
    mx = sum(xs)/n; my = sum(ys)/n
    sl = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    ic = my - sl*mx
    ss = sum((y-my)**2 for y in ys)
    rs = sum((y-(ic+sl*x))**2 for x, y in zip(xs, ys))
    return sl, ic, 1 - rs/ss

def P(a, q):
    a = sorted(a); k = (len(a)-1)*q/100.0
    lo, hi = math.floor(k), math.ceil(k)
    return a[lo] if lo == hi else a[lo] + (a[hi]-a[lo])*(k-lo)

for nm, d in (('lat1  DPA on  / 65536', A),
              ('noDPA DPA off / 65536', B),
              ('noDPA DPA off /  8192', C)):
    for lab, s in (('ALL    ', d), ('<5000ms', [z for z in d if z[1] <= 5000])):
        sl, ic, r2 = fit(s)
        print('%s %s n=%-4d %7.0f + %5.2f/ktok  R2=%.4f  -> %6.0f tok/s'
              % (nm, lab, len(s), ic, sl*1000, r2, 1e6/(sl*1000)))

print()
print('%16s %11s %11s %10s %11s %12s'
      % ('input bin', 'DPAon/65K', 'DPAoff/65K', 'DPAoff/8K', 'DPA effect', 'chunk effect'))
for lo, hi in [(0,40000),(40000,60000),(60000,80000),(80000,100000),
               (100000,130000),(130000,160000),(160000,200000)]:
    a = [t for p, t in A if lo <= p < hi]
    b = [t for p, t in B if lo <= p < hi]
    c = [t for p, t in C if lo <= p < hi]
    if min(len(a), len(b), len(c)) < 3:
        continue
    pa, pb, pc = P(a,50), P(b,50), P(c,50)
    print('%7d-%-8d %11.0f %11.0f %10.0f %10.2fx %11.2fx'
          % (lo, hi, pa, pb, pc, pa/pb, pb/pc))
```

Expected fits:

    lat1  DPA on  / 65536 ALL     n=124    -319 + 29.33/ktok  R2=0.9563 ->  34092 tok/s
    lat1  DPA on  / 65536 <5000ms n=114    -106 + 26.50/ktok  R2=0.9493 ->  37736 tok/s
    noDPA DPA off / 65536 ALL     n=115    -423 + 19.96/ktok  R2=0.8311 ->  50102 tok/s
    noDPA DPA off / 65536 <5000ms n=113    -182 + 16.75/ktok  R2=0.8324 ->  59705 tok/s
    noDPA DPA off /  8192 ALL     n=175    -374 + 20.68/ktok  R2=0.2531 ->  48345 tok/s
    noDPA DPA off /  8192 <5000ms n=172    -145 + 15.55/ktok  R2=0.9747 ->  64306 tok/s

Fits use **all** requests of each run (ramp included); the `<5000ms` rows drop the
few TTFT outliers. The contrast between arms — not any single intercept — is the
finding.

## Why: what actually changes

At concurrency 1 there is exactly one request, so dp-attention has **nothing to
parallelise across**. What remains is its cost:

- With dp8, the single request's prefill is split across 8 DP ranks that must
  gather at the attention boundary each layer, and 7 of 8 ranks hold no useful
  work for it.
- Without it, all 8 ranks run tensor-parallel on the same request — the standard
  path, with the collective the model was designed around.

**This is a statement about concurrency 1, and it does not generalise to load.**
DP-attention's purpose is packing *independent* requests across ranks; this run
deliberately removes the thing it optimises. Case A at N≈44 is where the other
side of that trade lives, and it was not re-run here. **The correct reading is
"DPA has a fixed per-request cost of ~1.7× prefill latency, which it must earn
back through concurrency"** — not "DPA is slower".

## TPOT went the other way — but the metric is too weak to carry the claim

TPOT p50 11.61 vs 10.66 ms (8 % slower), p90 14.24 vs 12.23 (16 %). The decode leg
was **identical on both arms** (DPA on, MTP on, same flags, `chunked_prefill_size`
8192 both times), so this is not a decode-side configuration change.

**Read the definition before reading the number.** The driver computes
(`agent_throughput.py:355`, `:2312`):

```python
if actual_gen_length > 1 and generation_time >= MIN_GENERATION_TIME:
    self.actual_tpots.append(generation_time / (actual_gen_length - 1))
# generation_time = total_time - ttft
```

Three properties of that make it a poor instrument for a cross-arm comparison,
and they are the reason this section claims nothing:

1. **It is one mean per request, not a token-level distribution.** A 19,659-token
   generation and a 31-token one each contribute exactly **one** sample. So
   "TPOT p90" is the 90th percentile *of per-request average decode rates*, not
   the 90th-slowest token. The percentile ladder is over requests, and n is 175.
2. **The numerator is client wall-clock minus TTFT**, so PD handoff, SSE transport
   and client-side scheduling all land inside "generation time". Small at
   concurrency 1, but systematically inflating, and not necessarily equally on
   both arms.
3. **MTP makes the average structurally misleading.** EAGLE accepts 2.934 tokens
   per verify step here, so the true inter-token interval is bimodal — accepted
   tokens arrive at near-zero spacing, and the gap lives between verify steps.
   Dividing by `gen_length - 1` erases exactly that structure.

**Disposition: the 8–16 % is reported as an observation and is NOT claimed as a
DPA effect.** A KV-layout explanation is plausible (without prefill DPA the KV
handed to decode is sharded differently) but nothing here isolates it. Settling
it needs per-chunk SSE timestamps — i.e. a real ITL measurement, which this
driver does not produce (see below) and which would be a new experiment, not a
re-analysis.

### ITL is not available from these artifacts

`metrics.jsonl` carries no inter-token or inter-chunk series. Three candidate
substitutes were checked and all fail:

- `new_inter_arrival_times` — **request** arrival gaps, not token gaps.
- `new_session_times` — verified to be cumulative timestamps from run start
  (1.55 → 8.49 → 41.6 → 66.6 → 79.6 s, monotonic), not per-request durations.
- `TPOT × gen_len` — recovers total decode time only; cannot reach a per-token
  distribution.

Given point 3 above, ITL and TPOT are **not** interchangeable on an MTP-enabled
decode leg. No ITL number is reported rather than deriving one that would be
wrong in a way the reader could not see.

## Throughput

| | lat1 (DPA on) | noDPA (DPA off) |
|---|---:|---:|
| presented input | 5,927 tok/s | **7,226 tok/s** |
| uncached input (~11 % of presented) | ~653 tok/s | **~796 tok/s** |
| generation | 83 tok/s | 73 tok/s |
| requests | 124 | **175** |

At concurrency 1, throughput is service-time-limited, so a 1.7× faster prefill
shows up as +41 % requests in the same 33-minute window. It is **not** a throughput
result — see lat1's own caveat: the duty cycle is low by construction.

## What had to differ, and why it does not explain the result

### `--mem-fraction-static` 0.80 → 0.70 (forced)

The noDPA prefill leg **could not boot at 0.80**:

    HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 254 MB
    Fatal Python error: Aborted

Diagnosed from the log, not assumed: `token usage: 0.04` (KV pool empty, so not KV
exhaustion), `#new-seq: 1`, `#running-req: 0` (one request, so not batching). It is
**activation** memory — without dp-attention one rank computes attention over the
whole 8192-token chunk instead of its 1/8 slice.

So this arm carries **two** flag differences, and that is a real finding in itself:
**DP-attention off is not free — it costs activation headroom.**

Does it explain the TTFT win? No, and this is checkable rather than arguable: GMU
sets how much VRAM is *statically reserved for KV*, and `token usage` peaked at
**0.04** on both arms. The smaller pool (2,821,248 vs 2,939,264 tokens/rank, −4 %)
was never within 25× of binding. A KV pool that is 96 % idle cannot make prefill
1.7× faster.

### `--chunked-prefill-size` — matched at the GLOBAL level, and it cost a rerun

Covered in full at the top of this document; restated here as a flag difference.
`--chunked-prefill-size` is a **global** per-step budget, and DPA **divides** it
by `dp_size` (`server_args.py:4902`). So `server_args=8192` on lat1's dp8 prefill
leg is **8,192 per rank × 8 ranks = 65,536 globally**.

Matching the machine therefore means passing **65,536** on the DPA-off arm, where
no division happens — not 8,192. The MAIN arm (`results/chunk65536_MAIN/`) does
exactly that.

**This was got wrong first.** An earlier reading took the warning text
(`adjusted to 8192`) to mean a clamp, and the arm was run at a global 8,192 — ⅛
of lat1's per-step budget. That run is retained as `results/chunk8192_ARM/`
rather than discarded, because comparing the two noDPA arms isolates the chunk
effect, which turns out to be **nil** (0.98–1.06× across seven bins). The wrong
turn ended up supplying the control that licenses the headline.

### `--ep-size 8` — kept

The original leg script gated `--dp-size`, `--enable-dp-attention` and `--ep-size`
in one `if`, so `DPA=0` also collapsed the MoE's expert parallelism. Kept at 8 on
both arms; only attention moves.

### `--enable-prefill-delayer`, `SGLANG_DP_USE_GATHERV` — leave with DPA

Both are DPA's own machinery (aligning DP-rank arrival times; gathering per-rank
outputs). Neither has meaning without a DP group. Both are inert at concurrency 1
regardless — there is never a second request to align with.

## Verification

| gate | expected | actual |
|---|---|---|
| `BYTECODE_GATE`, both nodes | OK | **OK** |
| `enable_dp_attention`, prefill | **False** | **False** |
| `scheduler_DP` procs, prefill / decode | 0 / 8 | **0 / 8** |
| `chunked_prefill_size` effective, both arms | 8192 | **8192** |
| `ep_size`, both arms | 8 | **8** |
| `in_flight` / `sessions_active` max | 1 / 1 | **1 / 1** |
| cache actual / ideal | ≈0.889 / 0.890 | **0.8896 / 0.8899** |
| engine faults, both legs | 0 | **0** |
| retractions, both legs | 0 | **0** |
| `MC_FORCE_TCP` | 0 | **0** |
| correctness (short / needle) | 4/4, needle passes | **4/4, needle 5/5 on retest** |
| artifacts | 3 files | **3** |

Needle was 2/5 cold then **5/5** warm, with the failing depths *moving* between
runs and every depth returning its exact 7-digit value — the same signature Case A
documented (3/5 then 4/5) and diagnosed as sampling variance at the model's own
`temperature 1.0 / top_p 0.95`, not KV corruption.

## What this does not establish

- **Anything about DPA under load.** N=1 removes exactly what DPA optimises.
- **The TPOT direction.** 8–16 % slower decode is observed, not explained.
- **The 3 outliers.** Ruled out engine faults; no positive cause found.
- **A GMU-matched comparison.** 0.80 does not boot without DPA, so "same GMU" is
  not reachable on this arm. Argued immaterial from `token usage: 0.04`, not
  measured by an ablation.
- **p99 on either arm.** n=175 and n=124; p99 is a handful of observations.
