#!/usr/bin/env python3
"""Verdict of a verification run from its summary ($LOG_ROOT/verify.out), with the criteria written down before the
runs (README "验证计划"). Per deployment: the files / env / start-up checks prove which arm actually ran; then the arm's
own criteria; then the aggregates (A vs B gates, C gates, N arms) and the two verdicts.

usage: judge.py VERIFY.out
"""
import collections
import re
import sys

LINE = re.compile(r"^\d\d:\d\d:\d\d (arm-[A-Za-z0-9-]+?)-(\d{4}T\d{4}) (files|env|start-up|smoke|probe|longctx [0-9:]+|lm-eval|audit|up failed[^:]*): ?(.*)$")
runs = collections.OrderedDict()
scores = []
for line in open(sys.argv[1], errors="replace"):
    line = line.rstrip("\n")
    if " score " in line and "| C104" in line:
        scores.append(line)
    m = LINE.match(line)
    if not m:
        continue
    arm, stamp, check, rest = m.groups()
    r = runs.setdefault((arm, stamp), {"arm": arm[4:], "longctx": []})
    if check.startswith("longctx"):
        r["longctx"].append((check.split()[1], rest))
    else:
        r[check.split()[0]] = rest

out = []


def mark(ok, what):
    out.append(("PASS" if ok else "FAIL") + "  " + what)
    return ok


def info(what):
    out.append("INFO  " + what)


def num(pat, s, cast=int, default=None):
    m = re.search(pat, s or "")
    return cast(m.group(1)) if m else default


def probe(r):
    s = r.get("probe", "")
    ok_n = num(r"correct=\d+/(\d+)", s)
    return {"n": num(r"n=(\d+)", s), "errors": num(r"errors=(\d+)", s), "ok_n": ok_n,
            "correct": num(r"\(([\d.]+)\)", s, float), "bad": num(r"bad_start=(\d+)", s),
            "garbled": num(r"garbled=(\d+)", s), "cap": num(r"hit_cap=(\d+)", s)}


def longctx_ok(r, label):
    good = True
    for spec, s in r["longctx"]:
        for kind in ("needle", "gsm8k"):
            m = re.search(kind + r": n=(\d+) correct=(\d+) garbled=(\d+)", s)
            if not m:
                good &= mark(False, f"{label} longctx {spec} {kind}: missing")
                continue
            n, c, g = map(int, m.groups())
            need = n - (max(1, n // 32) if kind == "needle" else max(2, n // 16))
            if spec.startswith("100000"):
                need = n - 1
            good &= mark(c >= need and g == 0, f"{label} longctx {spec} {kind} {c}/{n} (need >= {need}), garbled {g}")
    return good


FIX_ARMS = {"B-gate", "B-prod", "C-gate", "C-prod", "N1-nospec-fix", "N1a-nospec-audit"}
E11_ARMS = {"C0-e11", "C-gate", "C-prod", "N0-nospec", "N1-nospec-fix", "N1a-nospec-audit"}
agg = collections.defaultdict(lambda: collections.Counter())
for (arm, stamp), r in runs.items():
    a, label = r["arm"], f"{r['arm']}@{stamp}"
    out.append(f"--- {label}")
    if "up" in r and "files" not in r:
        mark(False, f"{label} did not start: {r.get('up', '')}")
        continue
    f = r.get("files", "")
    fix, dbg, org, rec = (num(p, f) for p in (r"fix=(\d+)", r"debug=(\d+)", r"orig=(\d+)", r"recipe=(\d+)"))
    if a == "N1a-nospec-audit":
        mark(fix == 5 and dbg == 1, f"{label} decode runs the fix + audit runner (fix={fix} debug={dbg})")
    elif a in FIX_ARMS:
        mark(fix == 6, f"{label} decode runs the fix (fix={fix} of 6)")
    else:
        mark(org == 6, f"{label} decode runs the image files (orig={org} of 6)")
    mark(rec == 5, f"{label} prefill runs the E10 recipe files (recipe={rec} of 5)")
    env = r.get("env", "")
    mark(("SGLANG_USE_ROCM700A=0" in env) == (a in E11_ARMS), f"{label} env ROCM700A as the arm says: {env.strip()}")
    st = r.get("start-up", "")
    dual = num(r"dual_graph_lines=(\d+)", st)
    spec = num(r"spec_args=(\d+)", st)
    if a.startswith("N"):
        mark(spec == 0, f"{label} no speculative decoding (spec_args={spec})")
        info(f"{label} DECODE-mode dual graphs kept: dual_graph_lines={dual}")
    elif a in FIX_ARMS:
        mark(dual == 0, f"{label} part 1: no dense/sparse pair for the verify graphs (dual_graph_lines={dual})")
    else:
        info(f"{label} image behaviour: dual_graph_lines={dual}")
    info(f"{label} capture: {re.sub(r'^.*capture=', '', st)}")
    smoke_ok = "6/6" in r.get("smoke", "")
    p = probe(r)
    info(f"{label} probe {r.get('probe', 'missing')}")
    if a in ("A-gate", "B-gate", "C-gate"):
        mark(smoke_ok, f"{label} smoke {r.get('smoke', 'missing')}")
        longctx_ok(r, label)
        agg[a]["runs"] += 1
        agg[a]["n"] += p["ok_n"] or 0
        agg[a]["bad"] += p["bad"] or 0
        agg[a]["garbled"] += p["garbled"] or 0
    elif a == "C0-e11":
        mark((p["garbled"] or 0) >= 10, f"{label} positive control reproduces E11 garble (garbled={p['garbled']} >= 10)")
    elif a == "N0-nospec":
        agg["N0"]["garbled"] += p["garbled"] or 0
        info(f"{label} positive control for part 2: garbled={p['garbled']} bad_start={p['bad']}")
    elif a in ("N1-nospec-fix", "N1a-nospec-audit"):
        mark(smoke_ok, f"{label} smoke {r.get('smoke', 'missing')}")
        mark(p["garbled"] == 0 and (p["bad"] or 0) <= 1, f"{label} garbled={p['garbled']} (0) bad_start={p['bad']} (<= 1)")
        longctx_ok(r, label)
        agg["N1"]["ok"] += 1
        if a == "N1a-nospec-audit":
            s = r.get("audit", "")
            nm, om = num(r"new_mismatch=(\d+)", s), num(r"old_mismatch=(\d+)", s)
            viol = [int(x) for x in re.findall(r" viol=(\d+)", s)]
            bviol = [int(x) for x in re.findall(r"bound_viol=(\d+)", s)]
            nob = [int(x) for x in re.findall(r"nobound=(\d+)", s)]
            dense = [int(x) for x in re.findall(r" dense=(\d+)", s)]
            mark(nm == 0 and num(r"replays=(\d+)", s, default=0) > 0, f"{label} audit: ranks agree on every replay (new_mismatch={nm})")
            mark(viol and max(viol) == 0, f"{label} audit: dense never with a sequence > index_topk (viol={viol})")
            mark(bviol and max(bviol) == 0 and nob and max(nob) == 0, f"{label} audit: bound always present and an upper bound (bound_viol={bviol}, nobound={nob})")
            mark(dense and max(dense) > 0, f"{label} audit: the dense variant is still used (dense={dense})")
            info(f"{label} audit: the per-rank rule would have mismatched on {om} replays of the same run")
            agg["N1a"]["audit_ok"] = int(nm == 0 and viol and max(viol) == 0 and max(bviol or [1]) == 0)
    elif a in ("B-prod", "C-prod"):
        mark(smoke_ok, f"{label} smoke {r.get('smoke', 'missing')}")
        mark((p["correct"] or 0) >= 0.96 and p["garbled"] == 0 and (p["bad"] or 0) <= 1,
             f"{label} probe correct={p['correct']} (>= 0.96) garbled={p['garbled']} (0) bad_start={p['bad']} (<= 1), errors={p['errors']}")
        longctx_ok(r, label)
        le = r.get("lm-eval", "")
        s_, f_ = num(r"strict ([\d.]+)", le, float, 0), num(r"flexible ([\d.]+)", le, float, 0)
        mark(s_ >= 0.96 and f_ >= 0.96, f"{label} lm-eval strict {s_} flexible {f_} (>= 0.96)")
    if (p["errors"] or 0) > 2:
        info(f"{label} {p['errors']} failed requests (transport / server errors, not counted as wrong output): read the log")

out.append("--- aggregates")
A, B, Cg = agg["A-gate"], agg["B-gate"], agg["C-gate"]
a_pos = A["bad"] + A["garbled"] >= 3
if A["runs"]:
    mark(a_pos, f"control A reproduces in this session: bad_start {A['bad']} + garbled {A['garbled']} over {A['n']} prompts (need >= 3)")
if B["runs"]:
    mark(B["bad"] <= 1 and B["garbled"] <= 1, f"B gates: bad_start {B['bad']} (<= 1), garbled {B['garbled']} (<= 1) over {B['n']} prompts, {B['runs']} gates")
if Cg["runs"]:
    mark(Cg["garbled"] == 0 and Cg["bad"] <= 1, f"C gates: garbled {Cg['garbled']} (0), bad_start {Cg['bad']} (<= 1) over {Cg['n']} prompts")
for s in scores:
    info(s)
fails = [o for o in out if o.startswith("FAIL")]
print("\n".join(out))
print("--- verdict")
print(f"{len(fails)} FAIL lines" + ("" if fails else " (all criteria met)"))
if A["runs"] and not a_pos:
    print("NOTE  control A did not reproduce the bad start: B's result says nothing about the bad start in this session")
