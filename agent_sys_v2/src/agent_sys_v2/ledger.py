"""The ledger (design §5.6): the results a project's tasks measured, read into one table with its baseline and best,
and into Y–X curves, one per series.

A result is a JSON file at the top of a task's workspace's results/ (results/*.json; the ledger does not read the
subdirectories, where raw data goes), with

  {"schema": "infera.result/v1", "label": "c64-spf", "metrics": {"input_tput_per_gpu": 14120.5, ...},
   "role": "baseline" | "candidate" | "reference", "round": 1, "series": "v0",
   "time": "2026-10-24T10:00:00Z", "change": "...", "workload": {"conc": 64, ...},
   "config": {...}, "verdict": {"status": "valid" | "void", "decision": "accept" | "reject" | "noise" | "rerun",
   "vs_baseline_pct": 15.4, "notes": "..."}, "evidence": {...}}

schema, label and metrics (numbers) are required; NaN and Infinity are not numbers. role reference is someone else's
number (an InferenceX curve): it is drawn and compared, never the baseline or the best. round is the project's round
the result belongs to, series the curve it is a point of (v0, v1 for the versions, r2-spf for an experiment,
ix:<entry> for a reference); the points of a series are told apart by workload.conc. project.yaml says how to read
them:

  ledger: {primary: input_tput_per_gpu, baseline: experiment/b0}    # baseline: a task or a file
  ledger: {primary: input_tput_per_gpu, x: p90_intvty, baseline_series: v0, working_point: {x: 30}}   # curves

The baseline is the latest valid result with the primary metric of the pinned task or file, else the latest such
result with role baseline. delta_pct is the primary metric against the baseline, gain the same in the better
direction (positive is better); with a baseline series, a result with a concurrency is compared with that series'
point at the same concurrency. Only results that count (`counts`) can be the best. With x (the curves' X metric;
x_higher_is_better, default true) every series is a curve: its points sorted by X, its frontier (the points no other
point of the series beats on both axes), and its value at the working point: {conc: C} = its points at concurrency C;
{x: X} = the best Y with X at least that (on the frontier, interpolated between two points), the "maximise throughput
at an interactivity SLA" of a project's goal. When the baseline series reaches the working point, the head compares
series there (curve_head): its baseline is that series, its best the series that counts with the best gain there.
Files are read again only when they change.

  "$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger check <dir or file> ... [--require-verdict]
  "$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger agentx <agentx_conc*.json or dir> ... --out "$AGENT_SYS_WORKSPACE/results"
      --series v0 [--round 1] [--role baseline] [--label PREFIX] [--change TEXT] [--time ISO]

check checks result files (a task's acceptance check): exit 1 and the reasons when one is wrong or there is none.
agentx writes one result per InferenceX AgentX output (agentx_conc<C>.json), with the metrics the InferenceX App
derives from it (ix_metrics), so our points and the official ones plot on one chart; it never overwrites a result
(a repeat at the same concurrency needs its own --label).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .model import iso, plain, write_json

SCHEMA = "infera.result/v1"
DECISIONS = ("accept", "reject", "noise", "rerun")
STATUSES = ("valid", "void")
ROLES = ("baseline", "candidate", "reference")
_cache = {}  # path -> ((mtime_ns, size), doc, errors)


def number(x):
    """A finite int or float, not a bool (nor an int too large for a float)."""
    try:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    except OverflowError:
        return False


def pct(value, base):
    """value against base in percent (2 decimals), None without both."""
    return (
        round((value - base) / abs(base) * 100, 2)
        if number(value) and number(base) and base
        else None
    )


def gain(delta, higher_is_better):
    """A delta in the better direction: positive is better; None without one."""
    if delta is None:
        return None
    return delta if higher_is_better else -delta or 0.0


def counts(r):
    """Whether a result can be the best: valid, ours (not a reference), and accepted, or a baseline nobody judged.
    Noise, a rerun, a rejection and a candidate not judged yet cannot."""
    return (
        r["status"] == "valid"
        and r["role"] != "reference"
        and (r["decision"] == "accept" or (r["role"] == "baseline" and not r["decision"]))
    )


def validate(doc, require_verdict=False):
    """The reasons a result document is wrong, [] when it is right."""
    if not isinstance(doc, dict):
        return ["not a JSON object"]
    errors = []
    if not str(doc.get("schema") or "").startswith("infera.result/"):
        errors.append(f'schema should be "{SCHEMA}"')
    if not str(doc.get("label") or "").strip():
        errors.append("label is missing")
    metrics = doc.get("metrics")
    if not isinstance(metrics, dict) or not any(number(v) for v in metrics.values()):
        errors.append("metrics should hold at least one number")
    role = doc.get("role", "candidate")
    if role not in ROLES:
        errors.append(f"role should be one of {', '.join(ROLES)}")
    n = doc.get("round", 0)
    if not (isinstance(n, int) and not isinstance(n, bool) and n >= 0):
        errors.append("round should be a whole number (the project's round)")
    if "series" in doc and not (isinstance(doc["series"], str) and doc["series"].strip()):
        errors.append("series should be a name (the curve the point belongs to)")
    workload = doc.get("workload", {})
    if not isinstance(workload, dict):
        errors.append("workload should be an object")
    elif "conc" in workload and not (number(workload["conc"]) and workload["conc"] > 0):
        errors.append("workload.conc should be a positive number (the concurrency)")
    verdict = {} if doc.get("verdict") is None else doc["verdict"]
    if not isinstance(verdict, dict):
        errors.append("verdict should be an object")
        verdict = {}
    if verdict.get("status", "valid") not in STATUSES:
        errors.append(f"verdict.status should be one of {', '.join(STATUSES)}")
    if verdict.get("decision") not in (None, *DECISIONS):
        errors.append(f"verdict.decision should be one of {', '.join(DECISIONS)}")
    if (
        require_verdict
        and role == "candidate"
        and not verdict.get("decision")
        and verdict.get("status") != "void"
    ):
        errors.append("verdict.decision is missing (a candidate needs the A/B decision, or void)")
    return errors


def _finite(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is not a number")
    return value


def read(path, require_verdict=False):
    """(doc, errors) of a file; doc None when it is not a result: some other JSON (no errors), or no JSON at all."""
    try:
        doc = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_finite, parse_float=_finite
        )
    except (OSError, UnicodeDecodeError, ValueError) as e:
        return None, [f"not JSON: {e}"]
    if not (isinstance(doc, dict) and str(doc.get("schema") or "").startswith("infera.result/")):
        return None, []  # some other JSON: not ours to judge
    return doc, validate(doc, require_verdict)


def load(path):
    """read(), cached by mtime and size."""
    try:
        st = path.stat()
    except OSError:
        return None, []
    key = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(path)
    if not hit or hit[0] != key:
        hit = _cache[path] = (key, *read(path))
    return hit[1], hit[2]


def json_files(path):
    """The *.json at the top of a directory (not in its subdirectories: raw data), or the file itself."""
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.glob("*.json") if p.is_file())
    return [path] if path.exists() else []


def working_point(value):
    """project.yaml's ledger.working_point -> {conc: C} or {x: X}; {} when it is neither."""
    value = value if isinstance(value, dict) else {}
    for key in ("conc", "x"):
        if number(value.get(key)):
            return {key: value[key]}
    return {}


def settings(project):
    cfg = plain(project.config.get("ledger")) or {}
    cfg = cfg if isinstance(cfg, dict) else {}
    return {
        "primary": str(cfg.get("primary") or ""),
        "higher_is_better": cfg.get("higher_is_better", True) is not False,
        "baseline": str(cfg.get("baseline") or ""),
        "x": str(cfg.get("x") or ""),
        "x_higher_is_better": cfg.get("x_higher_is_better", True) is not False,
        "baseline_series": str(cfg.get("baseline_series") or ""),
        "working_point": working_point(cfg.get("working_point")),
    }


def rows(project):
    """Every result of the project, oldest first, and the files that look like results but are wrong."""
    out, bad = [], []
    for t in project.tasks.values():
        for path in json_files(t.workspace / "results"):
            doc, errors = load(path)
            rel = str(path.relative_to(t.workspace))
            if errors:
                bad.append({"task": t.id, "file": rel, "errors": errors})
            if doc is None or errors:
                continue
            verdict = doc.get("verdict") or {}
            workload = doc.get("workload") or {}
            out.append(
                {
                    "task": t.id,
                    "title": t.title,
                    "type": t.type,
                    "file": rel,
                    "label": str(doc["label"]),
                    "time": str(doc.get("time") or iso(path.stat().st_mtime)),
                    "role": doc.get("role", "candidate"),
                    "round": doc.get("round"),
                    "series": str(doc.get("series") or "").strip(),
                    "conc": workload.get("conc"),
                    "change": str(doc.get("change") or ""),
                    "metrics": {k: v for k, v in doc["metrics"].items() if number(v)},
                    "workload": workload,
                    "status": verdict.get("status", "valid"),
                    "decision": verdict.get("decision") or "",
                    "notes": str(verdict.get("notes") or ""),
                }
            )
    return sorted(out, key=lambda r: (r["time"], r["task"], r["file"])), bad


def pick_baseline(table, cfg, primary):
    """(baseline, pinned): the latest valid result with the primary metric of the pinned task or file, else the
    latest such one with role baseline; pinned is False when ledger.baseline names one that has no such result."""
    usable = [r for r in table if r["status"] == "valid" and number(r["metrics"].get(primary))]
    pinned = cfg["baseline"]
    named = [
        r
        for r in usable
        if pinned in (r["task"], f"{r['task']}/{r['file']}", f"workspaces/{r['task']}/{r['file']}")
    ]
    base = next(reversed(named or [r for r in usable if r["role"] == "baseline"]), None)
    return base, bool(named or not pinned)


# ---------------------------------------------------------------- curves


def frontier(points, sx, sy):
    """The points no other one beats: at least as good on both axes and better on one (sx, sy: +1 when higher is
    better). Returns their indexes."""
    out = []
    for i, p in enumerate(points):
        beaten = any(
            q["x"] * sx >= p["x"] * sx
            and q["y"] * sy >= p["y"] * sy
            and (q["x"] * sx > p["x"] * sx or q["y"] * sy > p["y"] * sy)
            for q in points
        )
        if not beaten:
            out.append(i)
    return out


def at_x(front, x0, sx, sy):
    """The best Y of a frontier among X at least x0 (in the better direction): interpolated where x0 falls between
    two frontier points, the best Y overall when every point is past x0, None when none reaches it."""
    pts = sorted(front, key=lambda p: p["x"] * sx)
    past = [p for p in pts if p["x"] * sx >= x0 * sx]
    if not past:
        return None
    first = past[0]
    before = [p for p in pts if p["x"] * sx < x0 * sx]
    if not before or first["x"] == x0:
        best = max(past, key=lambda p: p["y"] * sy)
        return {"x": best["x"], "y": best["y"], "conc": best["conc"], "interpolated": False}
    a, b = before[-1], first
    y = a["y"] + (b["y"] - a["y"]) * (x0 - a["x"]) / (b["x"] - a["x"])
    return {"x": x0, "y": round(y, 6), "conc": None, "interpolated": True}


def at_conc(points, conc):
    """A series at one concurrency: the mean of its points there (repeats), None if it has none."""
    pts = [p for p in points if p["conc"] == conc]
    if not pts:
        return None
    mean = lambda k: round(sum(p[k] for p in pts) / len(pts), 6)  # noqa: E731
    return {"x": mean("x"), "y": mean("y"), "conc": conc, "n": len(pts), "interpolated": False}


def curves(table, cfg, primary, base_series):
    """The series as Y (primary) against X curves: [{name, role, round, counts, points, wp}], points sorted by X with
    their frontier flag; wp = the series at the working point with its delta and gain against the baseline series;
    counts: every point of it counts."""
    x = cfg["x"]
    if not x or not primary:
        return []
    sx, sy = (1 if cfg["x_higher_is_better"] else -1), (1 if cfg["higher_is_better"] else -1)
    groups = {}
    for r in table:
        vx, vy = r["metrics"].get(x), r["metrics"].get(primary)
        if r["series"] and r["status"] == "valid" and number(vx) and number(vy):
            groups.setdefault(r["series"], []).append(r)
    out = []
    for name, rs in groups.items():
        points = sorted(
            (
                {
                    "x": r["metrics"][x],
                    "y": r["metrics"][primary],
                    "conc": r["conc"],
                    "label": r["label"],
                    "task": r["task"],
                    "file": r["file"],
                    "time": r["time"],
                    "decision": r["decision"],
                }
                for r in rs
            ),
            key=lambda p: p["x"] * sx,
        )
        on = set(frontier(points, sx, sy))
        for i, p in enumerate(points):
            p["frontier"] = i in on
        roles = {r["role"] for r in rs}
        rounds = [r["round"] for r in rs if number(r["round"])]
        wp = cfg["working_point"]
        if "conc" in wp:
            value = at_conc(points, wp["conc"])
        elif "x" in wp:
            value = at_x([p for p in points if p["frontier"]], wp["x"], sx, sy)
        else:
            value = None
        out.append(
            {
                "name": name,
                "role": "reference"
                if roles == {"reference"}
                else "baseline"
                if name == base_series or "baseline" in roles
                else "candidate",
                "baseline": name == base_series,
                "round": max(rounds) if rounds else None,
                "time": max(r["time"] for r in rs),
                "counts": all(counts(r) for r in rs),
                "points": points,
                "wp": value,
            }
        )
    base = next((c for c in out if c["baseline"]), None)
    base_y = base["wp"]["y"] if base and base["wp"] else None
    for c in out:
        if c["wp"]:
            c["wp"]["delta_pct"] = pct(c["wp"]["y"], base_y)
            c["wp"]["gain"] = gain(c["wp"]["delta_pct"], cfg["higher_is_better"])
    # the baseline first, then ours by time, the references last
    return sorted(out, key=lambda c: (not c["baseline"], c["role"] == "reference", c["time"]))


def ledger(project):
    """The project's ledger: settings, the primary metric, baseline, best, rows with delta_pct, gain and the best so
    far, the curves, files in error."""
    cfg = settings(project)
    table, bad = rows(project)
    primary = cfg["primary"] or next((k for r in table for k in r["metrics"]), "")
    up = cfg["higher_is_better"]
    base, pinned = pick_baseline(table, cfg, primary)
    base_series = cfg["baseline_series"] or (base["series"] if base else "")
    # the baseline series at each concurrency: what a result at that concurrency is compared with
    by_conc = {}
    for r in table:
        if base_series and r["series"] == base_series and r["status"] == "valid":
            if r["conc"] is not None and number(r["metrics"].get(primary)):
                by_conc.setdefault(r["conc"], []).append(r["metrics"][primary])
    sign, best = (1 if up else -1), None
    for r in table:
        r["value"] = r["metrics"].get(primary)
        # on a curve: against the baseline series at the same concurrency, or nothing
        if r["conc"] is not None and by_conc:
            same = by_conc.get(r["conc"])
            r["delta_pct"] = pct(r["value"], sum(same) / len(same)) if same else None
        else:
            r["delta_pct"] = pct(r["value"], base and base["metrics"][primary])
        # the delta the right way up; none where it means nothing: a void measurement, or one inside the noise
        void = r["status"] == "void" or r["decision"] == "noise"
        r["gain"] = None if void else gain(r["delta_pct"], up)
        r["baseline"] = r is base
        if counts(r) and number(r["value"]):
            if not best or r["value"] * sign > best["value"] * sign:
                best = r
        r["best_so_far"] = best and best["value"]
    if by_conc:
        # values at different concurrencies do not compare: the best is the best gain over the baseline series at
        # its own concurrency (the baseline while none beats it), and there is no best so far
        best = max(
            (r for r in table if counts(r) and r["gain"] is not None),
            key=lambda r: (r["gain"], r["baseline"]),
            default=None,
        )
        for r in table:
            r["best_so_far"] = None
    series = curves(table, cfg, primary, base_series)
    base_curve = next((c for c in series if c["baseline"] and c["wp"]), None)
    if base_curve:
        # the head compares the series at the working point; one best across concurrencies would mean nothing
        gain_at_wp = lambda c: (c["wp"] or {}).get("gain")  # noqa: E731
        reached = [c for c in series if c["counts"] and gain_at_wp(c) is not None]
        baseline = brief_series(base_curve)
        best = brief_series(max(reached, key=gain_at_wp, default=None))
        for r in table:
            r["baseline"], r["best_so_far"] = r["series"] == base_series, None
    else:
        baseline, best = brief_row(base), brief_row(best)
    return {
        "primary": primary,
        "higher_is_better": up,
        "x": cfg["x"],
        "x_higher_is_better": cfg["x_higher_is_better"],
        "working_point": cfg["working_point"],
        # ledger.baseline names a task or file with no usable result (the head took the latest role baseline)
        "pin_missing": "" if pinned else cfg["baseline"],
        "curve_head": bool(base_curve),
        "baseline": baseline,
        "best": best,
        "count": len(table),
        "void": sum(r["status"] == "void" for r in table),
        "metrics": list(dict.fromkeys(k for r in table for k in r["metrics"])),
        "rows": table,
        "series": series,
        "errors": bad,
    }


def brief_row(r):
    return r and {
        k: r[k] for k in ("task", "title", "file", "label", "value", "delta_pct", "gain", "time")
    }


def brief_series(c):
    return c and {"series": c["name"], "value": c["wp"]["y"], **c["wp"]}


def head(full):
    """What the board's head shows of a ledger: primary metric, baseline, best and counts, without the rows."""
    return {k: v for k, v in full.items() if k not in ("rows", "metrics", "series")}


def task_briefs(full):
    """{task id: the result badge of its card}: how many results, the best delta among those that count besides the
    baseline (and its gain), whether they are all the baseline, the latest decision, and how many files are wrong."""
    by_task, out = {}, {}
    for r in full["rows"]:
        by_task.setdefault(r["task"], []).append(r)
    for tid, rs in by_task.items():
        best = max(
            (r for r in rs if counts(r) and not r["baseline"] and r["gain"] is not None),
            key=lambda r: r["gain"],
            default=None,
        )
        out[tid] = {
            "count": len(rs),
            "best_delta_pct": best and best["delta_pct"],
            "best_gain": best and best["gain"],
            "baseline": all(r["baseline"] for r in rs),
            "decision": rs[-1]["decision"],
            "label": rs[-1]["label"],
            "errors": 0,
        }
    for e in full["errors"]:
        empty = {
            "count": 0,
            "best_delta_pct": None,
            "best_gain": None,
            "baseline": False,
            "errors": 0,
        }
        out.setdefault(e["task"], empty)["errors"] += 1
    return out


# ---------------------------------------------------------------- InferenceX AgentX output -> result


def ix_metrics(agg):
    """The metrics of one agentx_conc<C>.json as the InferenceX App ingests it (its API names): for agentic rows the
    full-response ITL replaces the ITL when present, and interactivity is 1 / ITL of the same statistic; the
    e2e-normalized interactivity is the client's (the App recomputes the same formula). Throughput per GPU counts
    every GPU of the deployment (num_gpus: P and D), input including cache hits."""
    rm = agg["request_metrics"]
    lat = rm["latency"]
    itl = lat.get("full_response_itl") or lat["itl"]
    per_gpu = rm["throughput"]["per_gpu"]
    inv = lambda v: 1 / v if number(v) and v else None  # noqa: E731
    out = {
        "input_tput_per_gpu": per_gpu.get("input_tput_tps"),
        "tput_per_gpu": per_gpu.get("total_tput_tps"),
        "output_tput_per_gpu": per_gpu.get("output_tput_tps"),
        "p90_intvty": inv(itl.get("p90")),
        "median_intvty": inv(itl.get("p50")),
        "p90_e2e_norm_intvty": (lat.get("e2e_norm_intvty") or {}).get("p90"),
        "p90_ttft": (lat.get("ttft") or {}).get("p90"),
        "median_ttft": (lat.get("ttft") or {}).get("p50"),
        "p90_e2el": (lat.get("e2el") or {}).get("p90"),
        "p90_itl": itl.get("p90"),
    }
    return {k: round(v, 6) for k, v in out.items() if number(v)}


def topology(agg):
    """'P 1×TP1·PP4 → D 1×TP4·DCP4' for a PD deployment, 'TP8·EP8' for one engine."""

    def side(prefix):
        n = lambda k: agg.get(f"{prefix}_{k}") or 1  # noqa: E731
        parts = [f"TP{n('tp')}"] + [
            f"{name}{n(key)}"
            for name, key in (("PP", "pp"), ("DCP", "dcp_size"), ("EP", "ep"))
            if n(key) > 1
        ]
        if str(agg.get(f"{prefix}_dp_attention")).lower() == "true":
            parts.append("DPA")
        workers = n("num_workers")
        return ("" if workers == 1 else f"{workers}×") + "·".join(parts)

    if agg.get("disagg"):
        return f"P {side('prefill')} → D {side('decode')}"
    parts = [f"TP{agg.get('tp') or 1}"] + ([f"EP{agg['ep']}"] if (agg.get("ep") or 1) > 1 else [])
    return "·".join(parts)


def from_agentx(agg, source, series, round_=None, role="candidate", label="", change="", time=""):
    """One infera.result/v1 document from an agentx_conc<C>.json. Raises ValueError when it is not one."""
    try:
        conc = agg["conc"]
        metrics = ix_metrics(agg)
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"not an AgentX result (agentx_conc<C>.json): missing {e}") from None
    if not metrics:
        raise ValueError("not an AgentX result: no throughput or latency numbers")
    if "/" in (label or series):
        raise ValueError(f"{label or series!r}: a label or series names a file, so it has no /")
    accounting = agg.get("request_accounting") or {}
    throughput = agg["request_metrics"].get("throughput") or {}
    metrics |= {
        k: v
        for k, v in (
            ("requests", agg.get("num_requests_successful")),
            ("errors", accounting.get("records_error_dropped")),
        )
        if number(v)
    }
    doc = {
        "schema": SCHEMA,
        "label": f"{label or series}-c{conc}",
        "time": time,
        "role": role,
        **({"round": round_} if round_ is not None else {}),
        "series": series,
        "change": change,
        "metrics": metrics,
        "workload": {
            "client": "agentx",
            "conc": conc,
            "duration_s": throughput.get("duration_seconds"),
            "dataset": (agg.get("dataset") or {}).get("hf_dataset_name")
            or (agg.get("dataset") or {}).get("loader"),
            "num_gpus": agg.get("num_gpus"),
            "requests_total": agg.get("num_requests_total"),
        },
        "config": {
            "hw": agg.get("hw"),
            "image": agg.get("image"),
            "model": agg.get("model"),
            "framework": agg.get("framework"),
            "precision": agg.get("precision"),
            "spec_decoding": agg.get("spec_decoding"),
            "disagg": agg.get("disagg"),
            "topology": topology(agg),
            "kv_offloading": agg.get("kv_offloading"),
            "router": (agg.get("router") or {}).get("name")
            if isinstance(agg.get("router"), dict)
            else agg.get("router"),
        },
        "evidence": {"agentx": str(source)},
    }
    for part in ("workload", "config"):
        doc[part] = {k: v for k, v in doc[part].items() if v not in (None, "")}
    return doc


def convert_agentx(paths, out, series, round_=None, role="candidate", label="", change="", time=""):
    """Write a result into out for every agentx_conc*.json under paths. Returns (files written, errors)."""
    written, errors = [], []
    found = []
    for p in map(Path, paths):
        found += sorted(p.rglob("agentx_conc*.json")) if p.is_dir() else [p]
    if not found:
        errors.append(f"no agentx_conc*.json under {', '.join(map(str, paths))}")
    out = Path(out)
    for f in found:
        try:
            agg = json.loads(f.read_text(encoding="utf-8"))
            doc = from_agentx(
                agg,
                f.resolve(),
                series,
                round_,
                role,
                label,
                change,
                time or iso(f.stat().st_mtime),
            )
        except (OSError, UnicodeDecodeError, ValueError) as e:
            errors.append(f"{f}: {e}")
            continue
        target = out / f"{doc['label']}.json"
        # a repeat of the series at this concurrency must not replace the first
        if target.exists():
            errors.append(f"{f}: {target} exists already: give a repeat its own --label")
            continue
        write_json(target, doc)
        written.append(target)
    return written, errors


# ---------------------------------------------------------------- python -m agent_sys_v2.ledger check


def check(paths, require_verdict=False):
    """(all well, lines to print): there is at least one result file among paths (the files at the top of a
    directory), and every one is valid."""
    found, bad, lines = 0, 0, []
    for f in (f for p in paths for f in json_files(p)):
        doc, errors = read(f, require_verdict)
        found += doc is not None
        bad += bool(errors)
        lines += [f"{f}: {e}" for e in errors] or ([f"{f}: ok ({doc['label']})"] if doc else [])
    if not found:
        lines.append(f"no {SCHEMA} result under {', '.join(map(str, paths))}")
    return found > 0 and bad == 0, lines


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m agent_sys_v2.ledger",
        description="Check infera.result files; convert AgentX output.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="exit 1 unless every result file under the paths is valid")
    c.add_argument("paths", nargs="+")
    c.add_argument(
        "--require-verdict", action="store_true", help="a candidate must carry verdict.decision"
    )
    a = sub.add_parser(
        "agentx", help="write a result file for every agentx_conc*.json under the paths"
    )
    a.add_argument("paths", nargs="+")
    a.add_argument("--out", required=True, help="the results directory to write into")
    a.add_argument(
        "--series", required=True, help="the curve the points belong to, e.g. v0 or r2-spf"
    )
    a.add_argument("--round", type=int, help="the project's round")
    a.add_argument("--role", choices=ROLES, default="candidate")
    a.add_argument(
        "--label",
        default="",
        help="label prefix (default: the series); the label is <prefix>-c<conc>; give a repeat its own",
    )
    a.add_argument("--change", default="", help="one line: what differs from the baseline")
    a.add_argument(
        "--time", default="", help="ISO time of the measurement (default: the file's mtime)"
    )
    args = ap.parse_args(argv)
    if args.cmd == "agentx":
        written, errors = convert_agentx(
            args.paths,
            args.out,
            args.series,
            args.round,
            args.role,
            args.label,
            args.change,
            args.time,
        )
        print("\n".join([f"wrote {w}" for w in written] + errors))
        return 0 if written and not errors else 1
    ok, lines = check(args.paths, args.require_verdict)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
