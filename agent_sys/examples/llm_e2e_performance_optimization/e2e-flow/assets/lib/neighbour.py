#!/usr/bin/env python3
"""The other tenants' occupancy at a moment, in the shape `round_noise.py` reads.

`round_noise.py` consumes `r["neighbour"]` per round and **nothing writes it** —
which is m5's own objection turned on the field itself: *I will not build a
comparison against a field nobody writes.* This is the producer.

T32, as m5 refined it. m3's `gpu.txt` is a `rocm-smi` **product-info** dump taken
once at bring-up, so the card *set* is recoverable and the *occupancy during the
measurement* is not. That second thing is what separates **"the artefact is
wrong"** from **"the producer's card had a neighbour"**, and two of the worst
numbers of 2026-09-04 needed exactly it:

* the DELIVERY-NOTE refusal blamed a patch for a neighbour;
* the sealed arms' `probe` read **2062 s** and re-measured at **37 s** on the
  same budget — **56x** — because those arms ran on a contended chassis.

Neither was recoverable afterwards. Sampling costs one `rocm-smi` per step.

`neighbour` IS A STRING, AND THAT IS A CONSTRAINT RATHER THAN A STYLE
    `round_noise.summarise` does `conditions = {r.get("neighbour") for r in
    rounds}` — a **set**. Measured: a dict there raises `TypeError: cannot use
    'dict' as a set element`, so a per-card mapping in that field crashes the
    consumer rather than informing it. So `neighbour` is a short canonical label
    that compares by equality, which is exactly what "did the window hold one
    condition throughout" needs, and the numbers live beside it in
    `neighbour_detail`, which `summarise` ignores.

OURS IS NOT NOISE
    A sample includes our own engine, which on a TP-8 deployment is every card.
    Passing `--ours` excludes the cards this deployment took, so the label
    describes **the neighbour** and not the subject. With `--ours` unset every
    card counts, which is the right default for a probe that does not know what
    it is standing next to.

USAGE
    rocm-smi --showmemuse --showuse --csv | neighbour.py --ours 4,5,6,7 --step load
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone

#: A card is "in use" above this. Same 5 % bar `nodeprobe.sh` uses for a free
#: card, and for the same reason: a card can hold a few hundred MB of somebody's
#: idle context without being in use, and the decision does not get better from
#: a tighter number. One bar in two tools beats two bars.
BUSY_PCT = 5


def parse(text: str) -> dict[str, dict[str, int]]:
    """`rocm-smi --showmemuse --showuse --csv` -> {card: {vram_pct, use_pct}}.

    Both columns from **one** invocation. Two calls would sample two moments and
    report them as one, which is the error this module exists to stop.
    """
    out: dict[str, dict[str, int]] = {}
    for row in csv.DictReader(line for line in text.splitlines() if line.strip()):
        dev = (row.get("device") or "").strip()
        if not dev.startswith("card"):
            continue

        def num(key: str) -> int:
            try:
                return int(float(row.get(key) or 0))
            except (TypeError, ValueError):
                return -1

        out[dev] = {"vram_pct": num("GPU Memory Allocated (VRAM%)"),
                    "use_pct": num("GPU use (%)")}
    return out


def record(text: str, ours: set[str], step: str) -> dict:
    """One sample, ready to merge into a round or a step entry."""
    cards = parse(text)
    theirs = {c: v for c, v in cards.items() if c not in ours}
    busy = sorted(c for c, v in theirs.items()
                  if v["vram_pct"] > BUSY_PCT or v["use_pct"] > BUSY_PCT)
    # The label is deliberately coarse and deterministic: two samples of the same
    # condition must compare equal, or `window_was_uniform` reports churn that is
    # only formatting. Card indices are in it because "a neighbour moved to
    # different cards" is a different condition from "the same neighbour".
    if not theirs:
        label = "unknown"                      # every card is ours; nothing to say
    elif not busy:
        label = "idle"
    else:
        label = "busy:" + ",".join(c.removeprefix("card") for c in busy)
    return {
        "neighbour": label,
        "neighbour_detail": {
            "step": step,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bar_pct": BUSY_PCT,
            "ours": sorted(ours),
            "cards": theirs,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="",
                    help="comma-separated card indices this deployment took; "
                         "excluded from the label, because ours is not noise")
    ap.add_argument("--step", default="", help="which step this sample belongs to")
    ap.add_argument("--append", default="",
                    help="append the record to this jsonl instead of stdout")
    a = ap.parse_args()
    ours = {f"card{i.strip()}" for i in a.ours.split(",") if i.strip()}
    rec = record(sys.stdin.read(), ours, a.step)
    line = json.dumps(rec)
    if a.append:
        with open(a.append, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    else:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
