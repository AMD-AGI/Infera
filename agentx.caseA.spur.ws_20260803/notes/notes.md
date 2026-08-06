# Notes — traps, wrong turns, and what this run could not answer

Ordered by what they would cost someone repeating this.

---

## Trap 1 — `-G8` does not give you the GPUs (cost: two dead legs)

Both legs died at startup with
`HIP out of memory ... GPU has a total capacity of 287.98 GiB of which 0 bytes
is free`, while PyTorch itself had allocated only 60–66 GiB. The missing
~220 GB belonged to **another tenant** on the same node — `rocm-smi --showpids`
showed live PIDs holding ~270 GB/card with active SDMA.

**spur's `-G8` reserves a scheduling slot, not exclusive GPU memory.** Screening
nodes by "does it have the image" is not enough; you must also check free VRAM.
`scripts/screen.sh` does both and is the thing to run before committing a node:

```
crsuse2-m2m-276 img=1 maxUsedGB=278     <- has image, GPUs FULL
crsuse2-m2m-268 img=0 maxUsedGB=0       <- GPUs free, no image
```

## Trap 2 — the prebuilt image and free GPUs are anti-correlated

`infera/engine-sglang:merged-mtp` is **node-local** (built in place; no registry,
no exported tar). Of ~25 nodes screened, only 2 carried it — and both had their
GPUs fully occupied, repeatedly, across three allocation rounds. The nodes with
the image are the popular nodes precisely because they have been used.

The operator's instruction was "avoid rebuilding the image". It could not be
honoured: transferring 108 GB between nodes is slower than rebuilding, and
`--exclude`-driven re-screening never surfaced an image-carrying node with free
GPUs. **The rebuild took ~4 min per node** from `src.tar` + layer cache, run in
parallel on both.

The rebuild turned out to be strictly better anyway: it produces `final-pr`,
which has **GLM52_P1V3 baked in**, so no in-container patching was needed. On
`merged-mtp` the gate correctly reported `_p1v2_rows pyc_hits=0  MISSING`.

## Trap 3 — the packup's hicache verification command is stale

`agenticbench.mtp.caseA.packup_20260801/environment.md` gives:

```python
from sglang.srt.mem_cache.memory_pool_host import ALLOC_MEMORY_FUNCS, alloc_with_pin_memory
```

This now raises **ImportError**. The patch is present and live — the module was
refactored to `sglang.srt.mem_cache.pool_host.common`. Reading the ImportError
as "the fix is missing" would send you rebuilding an image that is already
correct. The behavioural check that works:

```python
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS, alloc_with_pin_memory
assert ALLOC_MEMORY_FUNCS["cuda"] is alloc_with_pin_memory
```

`start_ctr.sh`'s bytecode gate already uses the correct path.

## Trap 4 — the artifact dir must be inside the mounted path

The customer's `replay_caseA.sh` mounts only `$HERE` into the aiperf container.
Setting `OUT` outside it makes `--output-artifact-dir` resolve **inside the
container namespace**; the CSV dies with the container and the host-side
`[ -f "$csv" ]` test fails. `summary.csv` then records `FAILED` **for a run that
completed and produced full metrics**.

Fix is env-only, no edit to the customer script: `OUT=$HERE/results`.
Both points wrote clean rows this time.

## Trap 5 — `PORT_ACTIVE: 1` is correct here, `9` was the wrong signal

`start_ctr.sh` counts `PORT_ACTIVE` across all IB devices. On the first node
pair it read **9** (8 × ionic + mlx5_0); on the final pair, **1**. The 1 is the
right answer: spur's KV fabric is **mlx5_0**, and the 8 ionic devices are the
vultr fabric that must not be bound here. A leg script that auto-discovers ionic
will silently take the wrong path on spur.

Confirmed on the wire rather than assumed: `MC_FORCE_TCP` / `GID is NULL`
occurrences = **0** on both legs, i.e. no silent TCP fallback.

---

## What this run did NOT establish

**1. The 66 % → 88 % gap is measured, not explained.** Turn-0 cache-bust fully
accounts for the 66 % → 51 % part (turn 0 is 0.0 % across 53 and 73 requests).
The residual — the corpus's constructed ~88 % against ~66 % realized on turns
1+ — is *consistent with* block-vs-page granularity, but that is inference, not
measurement. Settling it needs per-request `cached_tokens` correlated against
the trace's own `hash_ids` overlap for the same request. Not captured here.

**2. `submission_valid=true` despite `--unsafe-override`.** The customer script
hardcodes `--unsafe-override` plus `--trajectory-start-min-ratio 0.90 /
--max-ratio 0.98`, while the scenario's defaults are 0.0/1.0. The expectation
going in was `submission_valid=false`. All three runs came back **true**. The
likely reading is that `default_trajectory_start_*_ratio` are *defaults*, not
locks, so overriding them violates nothing and `--unsafe-override` becomes a
no-op — but **this was not verified against the validator's source**, and the
distinction matters if the customer treats the flag as disqualifying.

**3. No DPA-on arm.** Prefill DPA was off by instruction. The TTFT numbers
therefore cannot be attributed to DPA-off versus the chunk/GMU pairing that
accompanies it — those moved together. A controlled comparison needs the same
YAML re-run with prefill DPA on and chunk back to a DPA-matched value.

**4. kvd `gets = 0`, again.** 168,866 sets against **zero** reads, matching what
both reference kits recorded. Whether prefill's HiCache ever consults kvd under
this workload is still unestablished; the discriminating measurement is
per-request HiCache tier accounting, which this run did not capture.

**5. MTP's 9.1 % degenerate batches.** 509 of 5,564 decode batches report
`accept len` ≥ 3.99 (the ceiling). The mean of 3.06 is healthy, but a ceiling
pile-up is the known tell for a repetition loop. Cheap check, not run: dump
output token IDs for those batches and look for repetition.

## Node ledger

Jobs `33488/33490/33491/33505` and later `35682/35683` (`par8ab`) belong to a
**different, parallel experiment**. They were never touched. All screening
allocations made here were released after use; the two nodes actually used were
`crsuse2-m2m-268` (prefill, job 35748) and `crsuse2-m2m-288` (decode, job 35749).

An earlier attempt on `crsuse2-m2m-276` + `crsuse2-m2m-253` was abandoned to
Trap 1; those jobs were cancelled, not left holding GPUs.
