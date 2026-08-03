# Patches

Four patches are load-bearing for this run. **One is new here
(CHUNK_PASSTHROUGH); three are inherited** and are reproduced from the kits
named below so this folder stays self-contained.

Three of the four share a failure mode worth stating once, because it is the
single most expensive class of bug on this stack:

> **They fail silently.** The leg boots, serves, and returns a clean run — of
> the wrong deployment. No error, no warning, plausible numbers.

| # | patch | file | new? | what it prevents |
|---|---|---|---|---|
| 1 | **CHUNK_PASSTHROUGH** | `../scripts/patch_leg_chunk.py` | **NEW** | outer `CHUNK=` silently dropped → leg runs 8192 while operator believes 16384 |
| 2 | DPA_PASSTHROUGH | in `../scripts/start_leg.sh` | inherited | outer `DPA=0` silently ignored → leg comes up **with** DP-attention |
| 3 | EP_DECOUPLE | in `../scripts/glm52_leg.sh` | inherited | `DPA=0` also collapses `ep_size` 8→1 → two variables move at once |
| 4 | GLM52_P1V3 | `../scripts/apply_p1v3.py` | inherited | decode leg **crashes** under MTP+DPA on an IDLE rank |

Patches 2–3 come from `../solo.glm52.dpaoff.packup_20260802/patches/`;
patch 4 from `../caseA.glm52.fullfeature.packup_20260801/patches/`.

---

## 1. CHUNK_PASSTHROUGH — NEW in this run

**What.** `start_leg.sh` hardcodes `ISL=8192 TP=8` in its `docker exec ... env`
block and never forwards `CHUNK`. `glm52_leg.sh:73` then derives it:

```bash
if [ "$DPA" = "1" ]; then CHUNK="${CHUNK:-$((ISL * TP))}"   # 65536
                     else CHUNK="${CHUNK:-8192}"; fi
```

**Why it matters.** sglang divides `chunked_prefill_size` by `dp_size` **only**
under DP-attention (`server_args.py:4902`). So the pre-patch behaviour is:

| | passed | engine divides? | per-forward |
|---|---|---|---|
| DPA=1 | 65536 | yes, ÷8 | 8192 |
| DPA=0 | 8192 | **no** | 8192 |

Both land on 8192 — which is *why the earlier DPA-off solo run was a fair
comparison*. But it also means **a DPA-off leg cannot be given a larger chunk
from outside**: the outer `CHUNK=` is dropped on the floor, and the leg boots at
8192 while the operator believes it is running 16384.

**How it was caught.** By reading `glm52_leg.sh:73` before launching, not after.
This one was caught *before* costing a window — unlike its two siblings.

**How applied.**

```bash
cp start_leg.sh start_leg.sh.bak_prechunk_$(date +%Y%m%d-%H%M)
python3 patch_leg_chunk.py start_leg.sh     # -> "patched OK ... occurrences: 1"
```

**Verified no-op when unset.** `${CHUNK:+CHUNK="$CHUNK"}` expands to `[]` on
both the backup and the patched file, so the **decode** leg (which passes no
CHUNK) is byte-identical to pre-patch. Checked by expanding both.

**Effect here.** Prefill launched with `CHUNK=16384`; the live process command
line reads `--chunked-prefill-size 16384`, and the resolved server arg is
`chunked_prefill_size=16384` (not divided, `dp_size=1`).

---

## 2. DPA_PASSTHROUGH (inherited)

**What.** `start_leg.sh` wrote a literal `DPA=1` into its docker-exec env block,
shadowing any outer `DPA=0`.

**Why it matters.** The launcher still printed its success line and the leg came
up **with DP-attention enabled**. A DPA-off experiment would have produced a
clean, plausible, meaningless second copy of the baseline.

**How it was caught.** By reading back the **live process command line** after
launch instead of trusting the launcher's echo — it still contained
`--dp-size 8 --enable-dp-attention`.

**The generalisable rule.** *A launcher's success message is not evidence that
the launcher did what you asked.* Verify the running process.

---

## 3. EP_DECOUPLE (inherited)

**What.** `--ep-size "$TP"` lived inside `if [ "$DPA" = "1" ]`. Running with
`DPA=0` dropped the flag and sglang resolved `ep_size` **8 → 1**.

**Why it matters.** MoE expert-parallelism would collapse at the same moment as
attention DP — two variables at once, and the measured delta unattributable.

**How it was caught.** By resolving the arguments offline before launching.

**Verified here.** `env/env_chi2835.txt` shows `--ep-size 8` on the live
DPA-off prefill command line.

---

## 4. GLM52_P1V3 (inherited) — and the trap that nearly voided this run

**What.** The image's own `GLM52_P1V2` trim guards only `real < padded`; on a
DP-attention **IDLE** rank under MTP draft-extend the inequality inverts, and
`fast_topk_v2` asserts:

```
RuntimeError: Expected lengths.size(0) == B to be true, but got false.
```

**Applied at runtime, inside the decode container, and LOST ON RESTART.**

```bash
docker cp apply_p1v3.py bench_run:/tmp/ && docker exec bench_run python3 /tmp/apply_p1v3.py
# -> "patched OK - GLM52_P1V3 occurrences: 3"
```

**⚠ The `.pyc` trap — hit live during this run's bring-up.** After patching the
*source* of an **already-running** engine, the loaded bytecode is still the old
one:

```
$ ls -la .../dsa/__pycache__/dsa_indexer.cpython-310.pyc
-rw-r--r-- 44050 Aug  1 05:59   <- image build time, NOT the patch time
```

The source said `GLM52_P1V3 occurrences: 3` while the engine was running
unpatched code. **Fix: delete the `.pyc`, relaunch the leg, then verify the
compiled bytecode** — not the source:

```bash
docker exec bench_run rm -f .../dsa/__pycache__/dsa_indexer*.pyc
# relaunch leg, then:
# (marshal.loads here reads a .pyc the container itself just compiled from our
#  own patched source -- it is first-party data, not an untrusted input. It is
#  the only way to inspect COMPILED bytecode, which is the whole point: the
#  source already claims to be patched. Do not point this at a foreign .pyc.)
docker exec bench_run python3 -c "
import marshal
c = marshal.loads(open('.../dsa_indexer.cpython-310.pyc','rb').read()[16:])
def walk(co):
    yield co
    for k in co.co_consts:
        if hasattr(k,'co_names'): yield from walk(k)
print(sum(1 for co in walk(c) if '_p1v2_rows' in (co.co_varnames+co.co_names)))
"
# -> 1   (and the .pyc mtime is now AFTER the patch)
```

This is CLAUDE.md principle 5 — *verify bytecode, not source* — and it has
invalidated a full experiment in this tree before.
