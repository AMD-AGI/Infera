#!/usr/bin/env python3
"""Did decode fit inside the captured CUDA graph, or fall back to eager?

**One implementation, two call sites** — m2's bench and m5's arms — on the
leader's ruling 2026-09-04. Not two implementations: `min_requests` went wrong
because the same bar was *written twice*, and the fix there was one name with
two overrides. This is the same predicate asked of **different engines**, which
is a different thing and needs one body.

## Why an absolute bar and not a comparison

m5 measured this standalone with constructed inputs. `stock_vs_m2_block`
compares m2's bench against m5's stock arm at 10 %, and it cannot see this
fault:

    case                          m2_bs  stock_bs   argv differ?  ceiling>=conc?
    1 agree   (healthy)             512       512   False         both True
    3 bad==bad (eager decode)         8         8   False         both False
    2 disagree (one side eager)     512         8   True          m2 T, stock F

**In case 3 the two sides agree — 8 == 8 — so a cross-comparison sees nothing.**
The block it produces is *structurally identical* to a healthy agreement: same
keys, same three `within_tolerance: True`, differing only in absolute values
that nothing bars. Measured, not argued.

The general form, and it is why this file exists rather than a tighter
tolerance: **a relative check cannot detect a fault both sides share**
(CONTRACT §4.6).

## What it reads, and both numbers are already in the handoffs

* **the ceiling** — `env/engine_argv.txt`, the engine's own `/proc/<pid>/cmdline`.
  m5's arms capture it at `serve/round.sh:174-175` and publish it at `:222`;
  m2's `merge.py` `LIFT` carries `env/` for every part, with a comment that
  already states the reason — *"the load configuration in it is what makes the
  two benches comparable, and is not recoverable from the numbers."* **The data
  has been in both handoffs all along, under the same filename, read by
  nothing.**
* **the concurrency** — `effective_decode_concurrency` out of
  `profile_export_aiperf.json`, which is **the same file `stock_vs_m2_block`
  already opens.**

`--cuda-graph-bs-decode` takes a **list** of batch sizes, so the ceiling is the
list's maximum, not a scalar.

## Validated on four real artefacts, both sides of the flag

m5 could only validate the healthy side — the sealed baseline. **m2 had failing
captures on disk and ran the other half**, which is the test this module could
not give itself:

    sealed aiperf_baseline   ok=True    ceiling 128 >= decode  6.93
    sealed aiperf_profiled   ok=None    no flag, correctly unchecked
    m2's m2itl-a1 (max 8)    ok=False   ceiling   8 <  decode 13.19   <- real refusal
    m2's m2itl-c1 (max 16)   ok=True    ceiling  16 >= decode 15.35

No fixture in any row. **The refusal path fires on a real eager-decode capture**,
with the right numbers.

**C1's margin is 4 %, and that is by construction rather than by luck.** Ceiling
16 against achieved decode concurrency 15.35 is what m1's prevention rule —
*choose a ceiling at least the load's concurrency* — produces at exactly its own
boundary: effective decode concurrency cannot exceed the offered concurrency, so
a ceiling set equal to it is correct and permanently near-tight. A reader seeing
`16 / 15.35` should read *"correct, at the bar"* and not *"nearly broken"*.

## Where this bar must NOT be pointed

**`profiling_mode_on`.** That line runs with CUDA graph **off by design** —
CONTRACT §1.1, because a graph launch hides the kernels the profiler exists to
see. The sealed `aiperf_profiled` carries no `--cuda-graph-*` flag at all, and
`check` correctly returns `ok: None` rather than `False` there. Wiring the bar
to that line would produce a guaranteed refusal on a correct capture.

The two call sites are **m2's `profiling_mode_off` bench** and **m5's two
arms** — engines that are meant to be serving normally.

**Measured, not requested.** The bar uses the concurrency the load *achieved*,
not `${max_conc}`, because what decides whether decode runs eager is the batch
the engine actually formed. A run that asked for 32 and achieved 7 is not
exceeding a ceiling of 8.

## Why it matters more than a normal bar

m1 established there is **no default**: the producing agent invents the ceiling
at every bring-up, and four real runs at tp=4 chose **16, 16, 8, 32**. So this
is not a setting somebody configured once wrongly — it is re-rolled per run, and
the decode difference is invisible in the environment record because nothing
records the ceiling there.

**What it is worth, measured by m2 on one node with one flag changed** — same
image, same node, same tp, same cards:

    max decode graph batch   8    ITL 42.15 ms     312 tok/s
    max decode graph batch  16    ITL  9.31 ms    1649 tok/s

**4.5x on decode, and `node`, `gpu_arch`, `image_id` and `model_path` are
identical across the two.** That is the whole argument for this file: the
package's strongest cross-handoff guard,
`check_environment.compare_fixed_across_inputs`, compares exactly those four
fields and cannot see a 4.5x engine.
"""
from __future__ import annotations

import itertools
import json
import pathlib

FLAG = "--cuda-graph-bs-decode"

#: Also accepted, because the flag has been spelled both ways across sglang
#: versions and a bar that silently finds neither would be the `items_schema`
#: shape again — present, and checking nothing.
FLAG_ALIASES = (FLAG, "--cuda-graph-max-bs")


def ceiling_from_argv(argv_path: pathlib.Path) -> tuple[int | None, str]:
    """The largest decode batch size the engine captured a graph for.

    Returns `(ceiling, why)`. `None` with a reason is **not** a pass — a caller
    that treats an unreadable ceiling as satisfied has built the thing this
    module exists to prevent.
    """
    if not argv_path.is_file():
        return None, f"no {argv_path.name} beside this measurement"
    toks = argv_path.read_text(encoding="utf-8", errors="replace").split()
    # **Graphs explicitly off beats a ceiling that is present but inert**, and it
    # has to be checked FIRST because both flags appear together.
    #
    # A real `profiling_mode_on` carries both: `assets/load/line.sh:85` adds
    # `--disable-cuda-graph` when `CAPTURE=1`, and the kit emits
    # `--cuda-graph-max-bs` **unconditionally** (`start_worker.sh`, and m1's
    # `deploy_kit.layout.yaml:591` says the kit "cannot express an override
    # without passing the flag twice"). Without this branch the search below
    # finds the ceiling, the bar passes, and the caller prints *"decode ran under
    # a captured graph"* about a line where graphs are off by design
    # (CONTRACT §1.1) — `ok=True`, no verdict changed, and a reassuring sentence
    # that is false.
    #
    # **The sealed corpus could not show this.** `aiperf_profiled` predates this
    # kit and carries no `--cuda-graph-*` flag at all, so it returned `None` and
    # reported "correctly unchecked" — the right answer for the wrong reason, on
    # the only artefact that existed. Predicted from these two files before rung
    # 2e's stage 2 dispatched, and corroborated independently by m1's comment and
    # by m5 reading the kit's `start_worker.sh`.
    #
    # `None` rather than a verdict: with graphs disabled, *did decode fit inside
    # the captured graph* has no answer, so **not applicable** is the honest
    # state — the same third state the rest of this module uses.
    if "--disable-cuda-graph" in toks:
        return None, ("the engine was started with --disable-cuda-graph — graphs are off by "
                      "design here, so there is no captured graph for decode to fit inside")
    for flag in FLAG_ALIASES:
        if flag not in toks:
            continue
        i = toks.index(flag)
        vals = list(itertools.takewhile(lambda t: not t.startswith("--"), toks[i + 1:]))
        nums = [int(v) for v in vals if v.isdigit()]
        if not nums:
            return None, f"{flag} is present with no numeric value"
        return max(nums), f"{flag} -> {nums if len(nums) > 1 else nums[0]}"
    return None, (f"none of {list(FLAG_ALIASES)} in the engine's command line — the engine "
                  "captured no decode graph. **Correct and expected for a "
                  "`profiling_mode_on` capture**, where CUDA graph is off by design "
                  "(CONTRACT §1.1: a graph launch hides the kernels the profiler exists to "
                  "see) — do not point this bar at that line. Anywhere else it means either "
                  "graphs are genuinely disabled or the flag was renamed again")


def decode_concurrency(aiperf: dict) -> tuple[float | None, str]:
    """The decode concurrency the load actually achieved.

    `effective_decode_concurrency` first: it is what decides whether a decode
    step exceeds the captured graph. `effective_concurrency` is the fallback for
    an export that predates it.

    **It is a `{unit, avg, p50, p90}` object, not a scalar**, exactly like every
    metric beside it. The first version of this function accepted only
    `int | float` — so it returned `None` on **every real artefact** while
    passing a fixture that wrote a bare number, which is §4.4.1 in the fixture
    of the check written to demonstrate §4.4.1. Caught by running against the
    sealed `aiperf_baseline` rather than against the battery; the battery was
    green and the bar would never have fired in production.

    A scalar is still accepted, because nothing costs less and an export that
    ever writes one should not silently disable the bar.
    """
    for key in ("effective_decode_concurrency", "effective_concurrency"):
        v = aiperf.get(key)
        if isinstance(v, dict):
            v = v.get("avg")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v), key
    return None, "the export carries neither effective_decode_concurrency nor effective_concurrency"


def check(argv_path: pathlib.Path, aiperf_path: pathlib.Path, label: str) -> dict:
    """Whether this measurement's decode fitted in its graph. One engine.

    The verdict is `ok: None` when either input is missing, with
    `unavailable_because` — never `True`. m5's `stock_vs_m2_block` uses the same
    three-state shape, and for the same reason: a reader must be able to tell
    *not applicable* from *not done*.
    """
    out: dict = {"label": label, "ok": None, "ceiling": None, "decode_concurrency": None}
    if not aiperf_path.is_file():
        out["unavailable_because"] = f"no {aiperf_path.name} for {label}"
        return out
    try:
        aiperf = json.loads(aiperf_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — the message is the finding
        out["unavailable_because"] = f"{aiperf_path.name} is unreadable: {type(exc).__name__}"
        return out

    ceiling, why_c = ceiling_from_argv(argv_path)
    conc, why_n = decode_concurrency(aiperf)
    out["ceiling_source"], out["concurrency_source"] = why_c, why_n
    if ceiling is None or conc is None:
        out["unavailable_because"] = "; ".join(x for x in (
            None if ceiling is not None else why_c,
            None if conc is not None else why_n) if x)
        return out

    out["ceiling"], out["decode_concurrency"] = ceiling, round(conc, 3)
    out["ok"] = ceiling >= conc
    if not out["ok"]:
        out["reason"] = (
            f"{label}: the engine captured decode graphs up to batch {ceiling} "
            f"({why_c}) and the load achieved decode concurrency {conc:.2f} — so decode "
            "exceeded the captured graph on essentially every step and the engine fell "
            "back to **eager decode**. Measured on this cluster: a 4.6x difference in "
            "decode speed from this cause alone, with an identical image and node.\n"
            "  Nothing in `environment.yaml` records the ceiling, so two engines this far "
            "apart are indistinguishable there — and because BOTH arms can share the "
            "fault, no comparison between them can find it (CONTRACT 4.6)."
        )
    return out


def check_pair(m2: dict, stock: dict) -> list[str]:
    """Reasons, for a caller grading two measurements at once.

    Kept separate from `check` so that a caller with one engine — m2's bench —
    uses exactly the same predicate as a caller with two.
    """
    return [r["reason"] for r in (m2, stock) if r.get("ok") is False and r.get("reason")]
