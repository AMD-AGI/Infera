"""Score the predict-then-run study against the measured candidates.

usage: python ptr_eval.py <ptr_dir> <out.json>

Per model and latency target: the best measured candidate that meets it, the
measured throughput and latency of the simulate and anchored picks, their
regret, and the released default (batch 1 in the model's own cache mode).
Also every candidate's latency error, the rank correlation of predicted and
measured throughput, and GPU-hours to measure the grid against what the
anchors cost.
"""

import glob
import json
import math
import os
import sys

d, out = sys.argv[1:3]
# The step sweep and its anchors: results/ on the cluster, workloads/ in the bench.
wdir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(d, "..")
pred = json.load(open(os.path.join(d, "predictions.json")))
G = pred["G"]
DEFAULT_MODE = {"dream_7B": "none"}
BAND = float(os.environ.get("BAND", "0.15"))


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for k, i in enumerate(order):
            r[i] = k
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb)


res = {"study_a": {}, "study_b": {}}
gpu_s_grid = gpu_s_anchor = gpu_s_confirm = 0.0
for preset, p in pred["study_a"].items():
    path = os.path.join(d, f"a_{preset}.json")
    if not os.path.exists(path):
        continue
    m = json.load(open(path))
    key = lambda r: (r["cache"], r["block"], r["batch"])  # noqa: E731
    meas = {key(r): r for r in m["rows"]}
    rows = []
    for c in p["candidates"]:
        r = meas.get(key(c))
        if r is None:
            continue
        rows.append({**c, **{k: r.get(k) for k in ("meas_ms", "oom", "peak_gb")}})
    ok = [r for r in rows if r.get("meas_ms")]
    tput = lambda r, k="meas_ms": r["batch"] * G / (r[k] / 1e3)  # noqa: E731
    err = {
        k: sum(abs(r[k] / r["meas_ms"] - 1) for r in ok) / len(ok) for k in ("sim_ms", "anc_ms")
    }
    rho = {
        k: spearman([tput(r, k) for r in ok], [tput(r) for r in ok]) for k in ("sim_ms", "anc_ms")
    }
    mode = DEFAULT_MODE.get(preset, ok[0]["cache"])
    default = next(r for r in ok if r["cache"] == mode and r["batch"] == 1)
    decisions = {}
    confirmed = set()
    for slo, picks in p["picks"].items():
        lim = None if slo == "None" else float(slo) * 1e3
        meets = [r for r in ok if lim is None or r["meas_ms"] <= lim]
        best = max(meets, key=tput) if meets else None
        dec = {"best": best and key(best), "best_tok_s": best and tput(best)}
        for mode_name, pk in picks.items():
            if pk is None:
                dec[mode_name] = {"pick": None, "feasible_measured": best is not None}
                continue
            r = meas.get((pk["cache"], pk["block"], pk["batch"]))
            if not r or not r.get("meas_ms"):
                dec[mode_name] = {"pick": list(pk.values()), "oom": True}
                continue
            met = lim is None or r["meas_ms"] <= lim
            dec[mode_name] = {
                "pick": [pk["cache"], pk["block"], pk["batch"]],
                "meas_ms": r["meas_ms"],
                "meets": met,
                "tok_s": tput(r),
                "regret": (1 - tput(r) / tput(best)) if best else None,
                "over_slo": (r["meas_ms"] / lim - 1) if lim and not met else 0.0,
            }
        dmet = lim is None or default["meas_ms"] <= lim
        dec["default"] = {"meets": dmet, "tok_s": tput(default)}
        # Two-stage: run only the candidates the anchored model puts within the
        # band of the target, keep the rest on their prediction.
        if lim is not None:
            band = [r for r in ok if abs(r["anc_ms"] / lim - 1) <= BAND]
            confirmed.update(key(r) for r in band)
            cand = [r for r in ok if r["anc_ms"] < lim * (1 - BAND)] + [
                r for r in band if r["meas_ms"] <= lim
            ]
            inband = {key(r) for r in band}
            score = lambda r: tput(r) if key(r) in inband else tput(r, "anc_ms")  # noqa: E731
            pk = max(cand, key=score) if cand else None
            runs_s = sum(3 * r["meas_ms"] / 1e3 for r in band) + (m["load_s"] if band else 0.0)
            if pk is None:
                dec["two_stage"] = {
                    "pick": None,
                    "feasible_measured": best is not None,
                    "runs": len(band),
                    "runs_s": runs_s,
                }
            else:
                met = pk["meas_ms"] <= lim
                dec["two_stage"] = {
                    "pick": list(key(pk)),
                    "meas_ms": pk["meas_ms"],
                    "meets": met,
                    "tok_s": tput(pk),
                    "regret": (1 - tput(pk) / tput(best)) if best else None,
                    "over_slo": (pk["meas_ms"] / lim - 1) if not met else 0.0,
                    "runs": len(band),
                    "runs_s": runs_s,
                }
        decisions[slo] = dec
    grid_s = m["sweep_s"] + m["load_s"]
    steps = json.load(open(os.path.join(wdir, f"{preset}.json")))
    anchors = json.load(open(os.path.join(wdir, "anchors", f"{preset}.json")))
    pts = {(a["batch"], a["q"], a["kv"]) for a in anchors["points"]}
    anchor_s = m["load_s"] + sum(
        7 * r["step_ms"] / 1e3 for r in steps["rows"] if (r["batch"], r["q"], r["kv"]) in pts
    )
    # A confirmation run is a warm-up and two timed generations, plus one load.
    confirm_s = (m["load_s"] if confirmed else 0.0) + sum(
        3 * meas[k]["meas_ms"] / 1e3 for k in confirmed
    )
    gpu_s_grid += grid_s
    gpu_s_anchor += anchor_s
    gpu_s_confirm += confirm_s
    res["study_a"][preset] = {
        "n": len(rows),
        "ran": len(ok),
        "oom": sum(1 for r in rows if r.get("oom")),
        "mape": err,
        "spearman": rho,
        "decisions": decisions,
        "grid_gpu_s": grid_s,
        "anchor_gpu_s": anchor_s,
        "confirm_gpu_s": confirm_s,
        "confirmed": len(confirmed),
        "rows": rows,
    }
res["gpu_hours"] = {
    "grid": gpu_s_grid / 3600,
    "anchors": gpu_s_anchor / 3600,
    "confirm": gpu_s_confirm / 3600,
}

# Study B
pb = pred["study_b"]
node, alone = {}, {}
for f in glob.glob(os.path.join(d, "b_*_tp*_r*.json")):
    mode = os.path.basename(f).split("_")[1]
    for r in json.load(open(f)):
        if r.get("meas_ms"):
            (node if mode == "node" else alone).setdefault((r["tp"], r["batch"]), []).append(
                r["meas_ms"]
            )
rows = []
for c in pb["candidates"]:
    k = (c["tp"], c["batch"])
    if k not in alone:
        continue
    # A node of 8 / TP replicas: every replica's latency, measured running together.
    lat = node.get(k) or alone[k]
    rows.append(
        {
            **c,
            "alone_ms": alone[k][0],
            "node_ms": max(lat),
            "node_tok_s": sum(c["batch"] * G / (x / 1e3) for x in lat),
        }
    )
dec_b = {}
for slo, picks in pb["picks"].items():
    lim = None if slo == "None" else float(slo) * 1e3
    meets = [r for r in rows if lim is None or r["node_ms"] <= lim]
    best = max(meets, key=lambda r: r["node_tok_s"]) if meets else None
    dec = {"best": best and (best["tp"], best["batch"]), "best_tok_s": best and best["node_tok_s"]}
    for mode_name, pk in picks.items():
        if pk is None:
            dec[mode_name] = {"pick": None, "feasible_measured": best is not None}
            continue
        r = next((r for r in rows if r["tp"] == pk["tp"] and r["batch"] == pk["batch"]), None)
        if r is None:
            dec[mode_name] = {"pick": [pk["tp"], pk["batch"]], "unmeasured": True}
            continue
        met = lim is None or r["node_ms"] <= lim
        dec[mode_name] = {
            "pick": [pk["tp"], pk["batch"]],
            "node_ms": r["node_ms"],
            "meets": met,
            "tok_s": r["node_tok_s"],
            "regret": (1 - r["node_tok_s"] / best["node_tok_s"]) if best else None,
        }
    dec_b[slo] = dec
res["study_b"] = {
    "rows": rows,
    "decisions": dec_b,
    "mape_alone": {
        k: sum(abs(r[k] / r["alone_ms"] - 1) for r in rows) / max(1, len(rows))
        for k in ("sim_ms", "anc_ms")
    },
    "mape_node": {
        k: sum(abs(r[k] / r["node_ms"] - 1) for r in rows) / max(1, len(rows))
        for k in ("sim_ms", "anc_ms")
    },
}
json.dump(res, open(out, "w"), indent=1)

for preset, v in res["study_a"].items():
    print(
        f"{preset:18} ran {v['ran']}/{v['n']} oom {v['oom']}  latency MAPE sim {v['mape']['sim_ms']:.1%} "
        f"anc {v['mape']['anc_ms']:.1%}  rho sim {v['spearman']['sim_ms']:.2f} anc {v['spearman']['anc_ms']:.2f}"
    )
    for slo, dd in v["decisions"].items():
        a = dd["anchored"]
        s = dd["simulate"]

        def fmt(x):
            if x.get("pick") is None:
                return "none" + (" (feasible!)" if x.get("feasible_measured") else "")
            if x.get("oom"):
                return f"{x['pick']} OOM"
            return f"{x['pick'][0][0]}{x['pick'][2]} {'ok' if x['meets'] else 'MISS'} regret {x['regret'] or 0:.1%}"

        t = dd.get("two_stage")
        t = f" | 2st {fmt(t)} runs {t['runs']}" if t else ""
        print(f"   slo {slo:>5}: best {dd['best']}  anc {fmt(a)} | sim {fmt(s)}{t}")
print("gpu hours", res["gpu_hours"])
if res["study_b"]["rows"]:
    print("study B MAPE alone", res["study_b"]["mape_alone"], "node", res["study_b"]["mape_node"])
    for slo, dd in res["study_b"]["decisions"].items():
        print("   B slo", slo, dd)
