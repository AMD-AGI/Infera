#!/usr/bin/env python3
"""The stock-vs-m2 reconciliation, exercised on constructed inputs.

    python3 recon_cases.py

Three cases, run against the real `compare.stock_vs_m2_block` and the real
`graph_ceiling.check`. No node, no GPU, under a second.

## Why these three, and why the third is the point

`stock_vs_m2_block` asks whether m5's stock arm reproduces m2's
`profiling_mode_off` bench within `stock_vs_m2_tolerance` (0.10). It had never
been run against anything but sealed-corpus numbers, so nobody had seen it
refuse, and nobody had seen what a *wrong pass* looks like.

    1 agree      both engines healthy            -> must PASS
    2 disagree   one engine in eager decode      -> must REFUSE
    3 bad == bad BOTH engines in eager decode    -> PASSES, and must

**Case 3 is not a bug in the reconciliation.** It passes because the two arms
genuinely agree; they agree about a number that is 4.6x wrong. The block it
writes is *structurally identical* to case 1 — same keys, same three
`within_tolerance: True` — so no reader and no downstream consumer can tell them
apart. That is why the battery asserts case 3 **passes** and separately asserts
that `graph_ceiling` catches it: the two checks answer different questions and
only one of them can.

## The prediction that was wrong, kept because it was wrong

Written before the first run (`PREDICTIONS` below). Case 2 was predicted as
*"inter_token_latency breaches, the other two hold"*. In fact
**`output_token_throughput` breaches at -78.6 %, four times louder than ITL's
+30.8 %.** For eager decode, throughput is the detector and ITL is the
corroborator — the opposite emphasis to the one both m5 and the leader had been
using all afternoon in prose.

The battery encodes which row actually moves, so the next reader inherits the
measurement rather than the prose.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
LIB = HERE.parent
PKG = LIB.parent.parent

PREDICTIONS = """\
1 agree      -> ok=True,  3/3 within_tolerance
2 disagree   -> ok=False, inter_token_latency breaches, other two hold   <- WRONG
                actual: throughput -78.6% AND itl +30.8% both breach
3 bad==bad   -> ok=True,  3/3 within, block structurally identical to case 1
                and no comparison can separate them                      <- held
"""

#: A healthy engine and one in eager decode, from the numbers this cluster
#: really produced: m1's floor was calibrated at 32.5 ms ITL and rung 1's
#: deployed engine measured 42.51 ms.
GOOD = (193.6, 32.50, 120.0)
BAD = (41.5, 42.51, 121.0)


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def aiperf(tput: float, itl: float, ttft: float, decode_conc: float) -> dict:
    """An export in **production's** shape, which this fixture got wrong once.

    `effective_decode_concurrency` was written here as a bare float. AIPerf
    writes `{unit, avg, p50, p90}`, like every metric beside it — so
    `graph_ceiling.decode_concurrency` returned `None` on every real artefact
    while this battery stayed green. §4.4.1, in the fixture of the check built
    to demonstrate §4.4.1.

    Named here rather than only fixed, because the rule that catches the next
    one is m2's: **fix it everywhere the same convenience appears**, and the
    convenience here was "a number is easier to write than the object the
    producer emits".
    """
    return {
        "output_token_throughput": {"unit": "tokens/sec", "avg": tput},
        "inter_token_latency": {"unit": "ms", "avg": itl},
        "time_to_first_token": {"unit": "ms", "avg": ttft},
        "effective_decode_concurrency": {"unit": "requests", "avg": decode_conc,
                                         "p50": decode_conc, "p90": decode_conc},
    }


def build(m2_vals, stock_vals, m2_bs, stock_bs, decode_conc):
    """One evidence tree and one stock arm, in the layout each consumer globs.

    The `--cuda-graph-bs-decode` list is written the way the engine really
    writes it — a run of numbers after the flag, from `/proc/<pid>/cmdline` with
    nulls turned to newlines — so the parser is exercised on the real shape and
    not on a scalar somebody found convenient.
    """
    base = pathlib.Path(tempfile.mkdtemp(prefix="recon-"))
    ev = base / "evidence"
    part = ev / "items" / "result" / "bench_profiling_mode_off"
    (part / "env").mkdir(parents=True)
    (part / "profile_export_aiperf.json").write_text(json.dumps(aiperf(*m2_vals, decode_conc)))
    (part / "env" / "engine_argv.txt").write_text(_argv(m2_bs))

    st = base / "stock"
    r1 = st / "items" / "result" / "r1"
    r1.mkdir(parents=True)
    (st / "items" / "env").mkdir(parents=True)
    (r1 / "profile_export_aiperf.json").write_text(json.dumps(aiperf(*stock_vals, decode_conc)))
    (st / "items" / "env" / "engine_argv.txt").write_text(_argv(stock_bs))
    return st, ev, part, r1


def _argv(ceiling: int) -> str:
    sizes = [n for n in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128) if n <= ceiling]
    return "\n".join(["--served-model-name", "m", "--cuda-graph-bs-decode", *map(str, sizes),
                      "--tp-size", "8"]) + "\n"


CASES = [
    # name,                m2,   stock,                  m2_bs, stock_bs, conc, recon_ok, ceil_ok
    ("1 agree           ", GOOD, (193.0, 32.60, 120.4), 128, 128, 6.93, True, (True, True)),
    ("2 disagree        ", GOOD, BAD,                   128, 8, 15.7, False, (True, False)),
    ("3 bad == bad      ", BAD,  (41.4, 42.60, 121.3),  8, 8, 15.7, True, (False, False)),
]


def main() -> int:
    cmp_mod = _load("cmp", PKG / "assets" / "compare.py")
    gc = _load("gc", LIB / "graph_ceiling.py")

    print(PREDICTIONS)
    bad = []
    blocks = {}
    for name, m2v, stv, m2bs, stbs, conc, want_ok, want_ceil in CASES:
        st, ev, part, r1 = build(m2v, stv, m2bs, stbs, conc)
        block = cmp_mod.stock_vs_m2_block(st, str(ev), 0.10)
        blocks[name] = block
        rows = "  ".join(f"{r['metric'][:12]:12s}{r['rel_delta']:+.3f}/"
                         f"{'T' if r['within_tolerance'] else 'F'}" for r in block["metrics"])
        got_ceil = (
            gc.check(part / "env" / "engine_argv.txt", part / "profile_export_aiperf.json", "m2")["ok"],
            gc.check(st / "items" / "env" / "engine_argv.txt", r1 / "profile_export_aiperf.json", "stock")["ok"],
        )
        ok = block["ok"] is want_ok and got_ceil == want_ceil
        print(f"{'OK ' if ok else 'BAD'} {name} recon={str(block['ok']):5s} "
              f"ceiling(m2,stock)={got_ceil}  {rows}")
        if not ok:
            bad.append(f"{name}: recon={block['ok']} want {want_ok}; "
                       f"ceiling={got_ceil} want {want_ceil}")

    # **The assertion the whole battery exists for.** Not "case 3 fails" — case
    # 3 passes, and the point is that its output is indistinguishable from a
    # correct pass. If a future change makes these differ, this line tells the
    # reader the reconciliation gained the ability to see case 3, which would be
    # a real improvement and should not go unnoticed.
    a, c = blocks[CASES[0][0]], blocks[CASES[2][0]]
    same_keys = sorted(a) == sorted(c)
    same_verdicts = ([r["within_tolerance"] for r in a["metrics"]]
                     == [r["within_tolerance"] for r in c["metrics"]])
    mentions = "cuda" in json.dumps(c).lower()
    print(f"\ncase1 vs case3: same keys={same_keys} same verdicts={same_verdicts} "
          f"mentions the ceiling={mentions}")
    if not (same_keys and same_verdicts and not mentions):
        bad.append("case 1 and case 3 are no longer indistinguishable — if that is "
                   "deliberate, update this battery and say what changed")

    print()
    if bad:
        for b in bad:
            print("FAIL", b)
        return 1
    print(f"{len(CASES)}/{len(CASES)} as specified; case 3 passes the reconciliation and "
          "is caught only by graph_ceiling")
    return 0


if __name__ == "__main__":
    sys.exit(main())
