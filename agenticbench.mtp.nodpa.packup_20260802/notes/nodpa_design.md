# noDPA — design note

## The question

lat1 measured the concurrency-1 latency floor of the full-feature deployment,
with DP-attention 8/8 on both legs. This run asks: **what does prefill-side
DP-attention actually buy at concurrency 1?**

Concurrency 1 is the right place to ask. DP-attention's *throughput* argument is
about packing independent requests across ranks; with exactly one request in
flight there is nothing to pack, so what remains is the pure single-request
question — is a 74K-token prefill faster when attention is data-parallel across
8 ranks, or tensor-parallel across the same 8?

## The comparison, and what makes it valid

| | lat1 | this run |
|---|---|---|
| client workload | `lat1_full.yaml` | `nodpa_full.yaml` — **byte-identical except `random_seed`** |
| prefill DP-attention | **ON** (dp8) | **OFF** |
| decode DP-attention | ON (dp8) | **ON** — unchanged |
| everything else | — | unchanged |

lat1 held the server fixed and varied the client. This run does the exact
opposite. That is the only reason the two are comparable at all.

**Decode keeps DPA.** The task is about prefill. Moving both legs would answer a
different (and less attributable) question, and would also disturb MTP, which
lives on the decode leg.

## Three confounders the leg script would have introduced, and the disposition of each

`DPA=0` in the original `glm52_leg_spur_mtp.sh` did not disable one thing. It
disabled four. Only one of them is the variable under test.

### 1. `--chunked-prefill-size` — **REAL confounder, and the hardest one**

`--chunked-prefill-size` is a **global** per-step token budget. DPA **divides** it
by `dp_size` (`server_args.py:4902`), so the lat1/Case A value of 65,536 at dp8
becomes 8,192 *per rank* while remaining 65,536 machine-wide.

The stock `else` branch hardcoded 8192 when `DPA=0`. Since no division happens
there, that is 8,192 **globally** — ⅛ of what the DPA arm gets. Left alone,
flipping DPA off would also have cut the global chunk 8×, and any TTFT rise would
be attributed to "DPA off" when much of it could be the smaller chunk.

**Disposition: `CHUNK` is caller-supplied, and the MAIN arm passes 65,536** to
match the *global* budget. See "Resolved from source" below — this took two
attempts and a rerun, because the engine's warning text reads like a clamp.

If the noDPA leg then OOMs on activation memory, that is a *real finding* about
non-DPA prefill (one rank must now hold the whole chunk's activations rather than
its 1/8 slice) and gets reported as such — not papered over by shrinking the
chunk. It did OOM, on `mem-fraction-static` rather than chunk; see below.

### 2. `--ep-size 8` dropped — **REAL confounder, kept**

GLM-5.2 is a MoE. `--ep-size` selects **expert** parallelism; `--enable-dp-attention`
selects **attention** parallelism. Different axes. The original `if` gated both,
so `DPA=0` also collapsed the MoE from ep8 to the TP default — changing the
expert-dispatch collective at the same time as the attention layout. **`--ep-size 8`
is kept**, holding the MoE side fixed.

### 3. `--enable-prefill-delayer` dropped — **not a confounder, correctly leaves**

It delays a prefill batch so DP ranks with different arrival times can be batched
together. With no independent DP ranks there is nothing to align — it is DPA's own
machinery and leaves with it. At concurrency 1 it is inert on *both* arms anyway
(there is never a second request to align with), so this costs nothing here.
Stated rather than relied on, so the reasoning does not rest on "N=1 hides it".

### 4. `SGLANG_DP_USE_GATHERV` dropped — **not a confounder**

Selects the variable-length gather that collects per-rank DP-attention outputs.
No DP group → read by nothing. Correctly leaves with DPA.

**Net functional diff vs the lat1 leg script: 3 lines**, all inside the DPA
branch. Verified by diffing with comments stripped.

### Resolved from source, after a wrong first reading

The boot logs print:

    WARNING:sglang.srt.server_args:DP attention is enabled. The chunked prefill
    size is adjusted to 8192 to avoid MoE kernel issues.

**That text is easy to misread as a clamp. It is not.** `server_args.py:4902`:

```python
if self._resolved().enable_dp_attention:
    self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size
```

It is a **division by `dp_size`**. The 8192 is `65536 // 8` and it is the
**per-rank** budget; the global per-step budget is still 65,536.

Consequences, all read from `server_args=` in the boot logs (carried in `../env/`)
rather than inferred:

| leg | DPA | requested | `server_args=` | semantics | **global tokens/step** |
|---|---|---:|---:|---|---:|
| lat1 prefill | on (dp8) | 65,536 | 8,192 | **per rank** | **65,536** |
| **noDPA-65K prefill** | off | **65,536** | **65,536** | global | **65,536** ✓ |
| noDPA-8K prefill | off | 8,192 | 8,192 | global | 8,192 (⅛) |
| decode, both arms | on (dp8) | 65,536 | 8,192 | per rank | — (inert) |

**Decode is not part of this comparison at all.** A PD decode leg does not run
prefill, so `chunked_prefill_size` there is inert on both arms. It is still
printed in `server_args=`, which is what made an earlier version of this note
treat it as a matched quantity.

**The wrong turn, recorded because it shaped the kit.** The first reading took
the warning as a clamp and concluded lat1's *effective* prefill chunk was 8,192,
so the arm was booted at `CHUNK=8192` and a full 33-minute window was measured
there. That arm was running at ⅛ lat1's global per-step budget — the confounder
this design existed to remove, reintroduced in the opposite direction.

Rather than discard it, that run is retained as `../results/chunk8192_ARM/`. The
rerun at a matched global 65,536 is `../results/chunk65536_MAIN/`, and the pair
measures the chunk effect in isolation: **nil, 0.98–1.06× across seven bins**.
The mistake ended up supplying the control that licenses the headline.

**Cross-check against `#new-token` in the engine logs**, which is the tokens
actually processed per prefill step and does not depend on reading the flag
semantics correctly:

- noDPA-8K: modal batch 8,192 — the whole machine.
- noDPA-65K: **max 65,536** — impossible under an 8,192 budget.
- lat1 (dp8): modal batch 8,192 — but **per rank, on 8 ranks concurrently**.

## What moves mechanically when DPA goes off

Stated as expectations to check against, **not** as predictions to confirm:

- **KV pool per rank.** Under dp-attention each rank holds full KV for its own
  requests; without it KV is sharded by head across all 8. The reported
  tokens/rank figure should change, and it is recorded from the boot log rather
  than assumed.
- **Activation memory.** One rank now processes the whole 65536-token chunk's
  attention rather than its 8192-token slice. This is the OOM risk above.
- **Collectives.** Attention output needs an all-reduce across TP ranks each
  layer, where dp8 needed a gather only at the boundary.

Which of these dominates at 74K–235K is the measurement.

## Finding at bring-up: noDPA prefill OOMs at the DPA arm's `mem-fraction-static`

The first noDPA prefill leg died mid-probe:

    rocdevice.cpp:3582 ... HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 254 MB
    Fatal Python error: Aborted

Same signature Case A hit at GMU 0.88, and the same diagnosis applies — read off
the log rather than assumed:

- `token usage: 0.04` — the KV pool is **empty**. Not KV exhaustion.
- `#new-seq: 1`, `#new-token: 8192`, `#running-req: 0` — one request, concurrency 1.
  Nothing to do with batching.
- 254 MB free — **activation** memory.

This is a real property of the arm, not a misconfiguration: without dp-attention
one rank computes attention over the whole 8192-token chunk instead of its 1/8
slice, so the transient activation peak is larger at identical chunk size. The
DPA arm's `--mem-fraction-static 0.80` was tuned with that 8× split in place.

**Disposition: lower GMU to 0.70 on this arm, and report it as a result.** The
alternative — shrinking the chunk — would reintroduce exactly the confounder this
design spent effort removing. GMU changes how much VRAM is *statically reserved
for KV*, which at concurrency 1 with `token usage: 0.04` is provably not the
binding resource; chunk size changes the *unit of work*, which is.

So this arm differs from lat1 in two server flags, not one, and the honest
statement is: **DP-attention off is not free at equal chunk size — it costs
activation headroom, and 0.80 is not reachable.** The knock-on is a smaller KV
pool, which is recorded from the boot log and is immaterial at concurrency 1
(`token usage` peaked at 0.04, i.e. 4 % of the pool that already existed).

## Seeds

lat1 used `1337` (Case A, contaminated), `20260802` (full), `2026080299` (probe).
This run's arms use **`2026080211`** (MAIN full) and **`2026080212`** (MAIN probe);
the retained chunk-control arm used **`2026080201`** / **`2026080202`**. All four
are distinct from each other and from lat1's three.

Two earlier candidates, `20260802001` and `20260802002`, were **rejected by numpy**
— they exceed `2**32-1` (~2.03e10 against a 4.29e9 bound). Caught by the probe;
see `notes.nodpa.md` §4.

The engine here is a **freshly built image on two new nodes** (the 2026-08-01 pair
expired at 08:36 UTC and was released), so the radix tree is genuinely cold and
`20260802` would probably have been safe. "Probably" is not a basis for a 30-minute
window, and the contamination defect is silent when it happens. Both seeds are
under `2**32-1` (numpy rejects larger — lat1's defect 2).

## Success criteria

| gate | expected |
|---|---|
| `BYTECODE_GATE` both nodes | OK |
| leg gate table | every row; `dp_size=8` **absent** on prefill, present on decode |
| `chunked_prefill_size` in prefill log | **65536** — the pinned value, verified not defaulted |
| correctness | short 4/4 at temp 1.0 / top_p 0.95 |
| probe: `in_flight` / `sessions_active` max | 1 / 1 |
| probe: cache actual | ≈ 0.889, **not** ≈ 1.0 |
| full run | ~30 min sustain, 3 artifacts, 0 engine faults |
| input distribution p50/p90/p99 | within sampling noise of lat1's 83.0K / 171.3K / 233.5K |
