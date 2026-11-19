"""Merge the per-GPU parts of a predict-then-run workload into one file per model.

usage: python ptr_merge.py <workload_dir>

wgen.py with CAND=cache:block writes part_<preset>__<cache>_<block>.json; a
preset run whole writes part_<preset>.json. Each becomes a_<preset>.json with
the rows of all its parts. ``load_s`` is one load, and ``sweep_s`` the GPU time
of every part's sweep plus the extra loads, so their sum is still the GPU time
of measuring the grid.
"""

import glob
import json
import os
import sys

d = sys.argv[1]
by_preset = {}
for f in sorted(glob.glob(os.path.join(d, "part_*.json"))):
    preset = os.path.basename(f)[len("part_") : -len(".json")].split("__")[0]
    by_preset.setdefault(preset, []).append(json.load(open(f)))
for preset, parts in by_preset.items():
    out = dict(parts[0])
    out["rows"] = [r for p in parts for r in p["rows"]]
    loads = [p["load_s"] for p in parts]
    out["load_s"] = max(loads)
    out["sweep_s"] = sum(p["sweep_s"] for p in parts) + sum(loads) - max(loads)
    out["parts"] = len(parts)
    json.dump(out, open(os.path.join(d, f"a_{preset}.json"), "w"), indent=1)
    print(preset, len(parts), "parts", len(out["rows"]), "rows")
