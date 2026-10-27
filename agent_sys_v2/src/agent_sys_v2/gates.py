"""What an AgentX point says of its own measurement, and the gates an A/B passes before its numbers are judged
(design §5.6). ab_judge.py and `ledger verify` call the same functions.

The measurement of a point is read from its run directory, never from what an agent wrote: AIPerf's own export
(profile_export_aiperf.json, below the directory of agentx_conc<C>.json) gives the window (start_time, end_time: the
client's local clock, as written), the set duration (its profiling phase) and the server it measured (the node: the
host of its endpoint URL); runtime.env says whether InferenceX ran it in fast mode (AIPERF_EXPERIMENTAL_FAST=1, which
forces a 1200 s window whatever DURATION says) and its DURATION when AIPerf's export is not there; the AgentX output
itself has the measured duration. What cannot be read is written down as missing, never made up.

The gates, with the project's rules (project.yaml ledger, snapshotted when a run starts; the command line may only make
them stricter):
  formal      only a formal measurement is the input of a verdict: not fast mode, set to at least min_duration_s
              (3600) and measured for at least 95% of that (AIPerf's own coverage bar). The others are screening:
              in the ledger, marked, never an input.
  reproduce   the A/B's base arm reproduces the baseline series: its mean within reproduce_pct (5%) of the series'
              reference at that concurrency (ledger.reference(): the points the task that sealed the version
              measured, never this task's or another experiment's). Else the A/B is not judged: rerun (on a quieter
              node or hour, it may reproduce).
  timing      on one node no two inputs' windows overlap, and the inputs follow one another with no gap longer than
              max_arm_gap_min (60): two arms far apart in time measure the time as well.
  noise       the fixed part of the noise floor is the larger of noise_pct and the version's retest noise: 1.96 x
              sqrt(2) x the relative standard deviation of the sealing task's formal repeats on one node (the 95%
              band of the difference of two single measurements, as judge() computes for an A/B's own spread). When
              it is above accept_pct the measurement cannot tell a change of the bar's size: a result inside it is
              noise, resolvable false, left to the round's close (待集成); measuring another pair does not lower it.
A gate that cannot be checked (a result converted before windows were recorded; no reference at that concurrency) says
so: unchecked, with why. It never fails for that.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from . import ledger

DEFAULT_MIN_DURATION_S = (
    3600.0  # ledger.min_duration_s: a formal AgentX measurement is set to this at least
)
COVERAGE = 0.95  # AIPerf's own metric_duration_coverage bar: a window measured for 95% of its set duration counts
DEFAULT_REPRODUCE_PCT = 5.0  # ledger.reproduce_pct
DEFAULT_MAX_ARM_GAP_MIN = 60.0  # ledger.max_arm_gap_min
DEFAULT_MIN_REPEATS = (
    2  # ledger.min_repeats: formal measurements on one node at the working point, for the noise
)
RETEST_FACTOR = 1.96 * math.sqrt(
    2
)  # the 95% band of the difference of two single measurements, per unit of RSD
AIPERF_EXPORT = "profile_export_aiperf.json"
LOCAL = ("localhost", "127.0.0.1", "::1", "0.0.0.0")
OK, FAILED, UNCHECKED = "ok", "failed", "unchecked"
TO_INTEGRATE = (
    "待集成"  # the remark the round's close looks for (skill infera-experiment: --note 待集成)
)


# ---------------------------------------------------------------- the measurement of one point


def read_env(path):
    """{name: value} of a runtime.env (KEY=value lines; # comments)."""
    text = Path(path).read_bytes().decode("utf-8", "replace")
    pairs = (
        line.split("=", 1)
        for line in text.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    return {k.strip(): v.strip() for k, v in pairs}


def phase(export):
    """AIPerf's profiling phase of an export: {duration, concurrency, ...} or {}."""
    cfg = export.get("input_config") if isinstance(export.get("input_config"), dict) else {}
    for p in cfg.get("phases") or []:
        if isinstance(p, dict) and p.get("kind", p.get("name")) == "profiling":
            return p
    loadgen = cfg.get("loadgen") if isinstance(cfg.get("loadgen"), dict) else {}
    return {
        k: v
        for k, v in (
            ("duration", loadgen.get("benchmark_duration")),
            ("concurrency", loadgen.get("concurrency")),
        )
        if v is not None
    }


def find_export(run_dir, conc):
    """(path, None) of the AIPerf export of the point at conc below run_dir (three levels down at most: InferenceX puts
    it in aiperf_artifacts/ or a directory below it), or (None, why not)."""
    found = []
    for depth in range(4):
        for p in sorted(Path(run_dir).glob("/".join(["*"] * depth + [AIPERF_EXPORT]))):
            try:
                export = json.loads(p.read_bytes())
            except (OSError, ValueError):
                continue
            c = phase(export).get("concurrency") if isinstance(export, dict) else None
            if c in (None, conc):
                found.append(p)
    if not found:
        return None, f"no AIPerf export ({AIPERF_EXPORT}) of C{conc} below {Path(run_dir).name}/"
    if len(found) > 1:
        return (
            None,
            f"{len(found)} AIPerf exports of C{conc} below {Path(run_dir).name}/: which one is unknown",
        )
    return found[0], None


def when(text):
    """A naive datetime from AIPerf's ISO time (it writes the client's local time, without a zone; one with a zone is
    taken in UTC), None when it is not one."""
    text = str(text).strip()
    # what Python 3.10's fromisoformat does not read: a Z for UTC, a fraction of other than 3 or 6 digits
    text = re.sub(r"Z$", "+00:00", text)
    text = re.sub(r"\.(\d+)", lambda m: "." + (m[1] + "000000")[:6], text, count=1)
    try:
        t = datetime.fromisoformat(text)
    except ValueError:
        return None
    return t.astimezone(timezone.utc).replace(tzinfo=None) if t.tzinfo else t


def host_of(url):
    host = urlparse(str(url)).hostname or ""
    return "" if host in LOCAL else host


def measurement(agg, env, export):
    """What a point says of its own measurement, from its AgentX output (agg), runtime.env ({} without one) and AIPerf
    export ({} without one): {duration_s, benchmark_duration_s, fast, window, node, missing: {field: why}}; a field
    that cannot be read is absent, and missing says why."""
    out, missing = {}, {}
    measured = ((agg.get("request_metrics") or {}).get("throughput") or {}).get("duration_seconds")
    if ledger.number(measured):
        out["duration_s"] = measured
    else:
        missing["duration_s"] = (
            "the AgentX output has no request_metrics.throughput.duration_seconds"
        )
    set_to = phase(export).get("duration") if export else None
    if ledger.number(set_to):
        out["benchmark_duration_s"] = (
            set_to  # what AIPerf ran: fast mode forces 1200 whatever DURATION says
        )
    elif env.get("DURATION", "").isdigit() and env.get("AIPERF_EXPERIMENTAL_FAST") != "1":
        out["benchmark_duration_s"] = int(env["DURATION"])
    else:
        missing["benchmark_duration_s"] = (
            "neither AIPerf's export nor runtime.env (DURATION) says the duration it was set to"
        )
    if "AIPERF_EXPERIMENTAL_FAST" in env:
        out["fast"] = env["AIPERF_EXPERIMENTAL_FAST"] == "1"
    elif export or env:
        out["fast"] = (
            False  # the ATOM kit has no fast switch: a fast run there is a short DURATION, caught above
        )
    else:
        missing["fast"] = "no runtime.env and no AIPerf export beside the AgentX output"
    start, end = (export.get("start_time"), export.get("end_time")) if export else (None, None)
    if when(start) and when(end) and when(start) <= when(end):
        out["window"] = {"start": str(start), "end": str(end)}
    else:
        missing["window"] = (
            "AIPerf's export has no start_time and end_time"
            if export
            else "no AIPerf export beside the AgentX output"
        )
    cfg = (export.get("input_config") or {}) if export else {}
    urls = (cfg.get("endpoint") or {}).get("urls") if isinstance(cfg, dict) else None
    urls = urls if isinstance(urls, list) else []
    if not urls and env.get("AIPERF_SERVER_URL"):
        urls = [env["AIPERF_SERVER_URL"]]
    hosts = sorted({host_of(u) for u in urls} - {""})
    if hosts:
        out["node"] = ",".join(hosts)
    else:
        missing["node"] = (
            "the client measured localhost (no node name)"
            if urls
            else "neither AIPerf's export nor runtime.env names the server it measured"
        )
    if missing:
        out["missing"] = missing
    return out


def read_point(raw_path, agg, aiperf=None):
    """(measurement, evidence of the files it read) of the point whose AgentX output is raw_path. aiperf: the AIPerf
    export the conversion recorded (a path), else it is looked for below the run directory."""
    run_dir = Path(raw_path).parent
    env_path = run_dir / "runtime.env"
    env = read_env(env_path) if env_path.is_file() else {}
    export, why = (Path(aiperf), None) if aiperf else find_export(run_dir, agg.get("conc"))
    data = {}
    if export:
        try:
            raw = export.read_bytes()
            data = json.loads(raw)
            data = data if isinstance(data, dict) else {}
        except (OSError, ValueError) as e:
            why = f"AIPerf export {export.name} cannot be read: {e}"
            raw = None
    m = measurement(agg, env, data)
    if why and "window" in m.get("missing", {}):
        m["missing"]["window"] = why
    evidence = {}
    if export and data:
        evidence = {"file": export, "sha256": hashlib.sha256(raw).hexdigest()}
    return m, evidence


# ---------------------------------------------------------------- formal or screening


def screening(m, min_duration_s=DEFAULT_MIN_DURATION_S):
    """Why a measurement is screening rather than formal ('' when it is formal, or when nothing says either way)."""
    if m.get("fast") is True:
        return "AgentX fast mode (AIPERF_EXPERIMENTAL_FAST=1)"
    if m.get("client") == "agentx" and not ledger.number(m.get("duration_s")):
        # ledger agentx always writes the measured duration: without it, the measurement says nothing of itself
        return "an AgentX result that does not record its measured duration (workload.duration_s)"
    set_to, measured = m.get("benchmark_duration_s"), m.get("duration_s")
    if ledger.number(set_to) and set_to < min_duration_s:
        return f"set to {set_to:g} s, below the formal {min_duration_s:g} s"
    if ledger.number(measured) and measured < COVERAGE * min_duration_s:
        return f"measured {measured:g} s, below {COVERAGE:.0%} of the formal {min_duration_s:g} s"
    return ""


def checked_formal(m):
    """Whether a measurement says enough to call it formal (its duration, set or measured)."""
    return ledger.number(m.get("benchmark_duration_s")) or ledger.number(m.get("duration_s"))


# ---------------------------------------------------------------- the gates of an A/B


def pct_rsd(xs):
    mean = statistics.mean(xs)
    return statistics.stdev(xs) / abs(mean) * 100 if len(xs) > 1 and mean else 0.0


def reproduce(base_values, reference_values, reproduce_pct):
    """The base arm against the baseline series' reference at the A/B's concurrency."""
    if not reference_values:
        return {
            "status": UNCHECKED,
            "why": "the baseline series' reference has no point at this concurrency",
        }
    ref = statistics.mean(reference_values)
    off = ledger.pct(statistics.mean(base_values), ref)
    if off is None:
        return {
            "status": UNCHECKED,
            "why": f"the reference's mean is {ref}: no relative difference",
        }
    out = {
        "off_pct": off,
        "reference_mean": round(ref, 6),
        "reference_n": len(reference_values),
        "reproduce_pct": reproduce_pct,
    }
    if abs(off) > reproduce_pct:
        return {
            **out,
            "status": FAILED,
            "why": f"the base arm did not reproduce the baseline series (off by {off:+.2f}%, bar {reproduce_pct:g}%)",
        }
    return {**out, "status": OK}


def timing(items, max_gap_min):
    """items: [{file, arm, window, node}]: no two windows overlap on one node (or where a node is not known: they
    may be on one), and no gap between one window and the next to start is longer than max_gap_min. The windows that
    are known are checked even when one is not; the gate is unchecked only when nothing failed and something could not
    be checked."""
    known = [i for i in items if i.get("window")]
    unknown = [i["file"] for i in items if not i.get("window")]
    spans = sorted(
        ((when(i["window"]["start"]), when(i["window"]["end"]), i) for i in known),
        key=lambda s: s[0],
    )
    problems = []
    for k, (_s1, e1, a) in enumerate(spans):
        for s2, _e2, b in spans[k + 1 :]:
            if s2 >= e1:
                continue
            na, nb = a.get("node"), b.get("node")
            if not (na and nb):
                problems.append(
                    f"{a['file']} and {b['file']} overlap (a node not known: they may share it)"
                )
            elif na == nb:
                problems.append(f"{a['file']} and {b['file']} overlap on {na}")
    gap, latest = 0.0, None
    for s1, e1, _i in spans:
        if latest is not None:
            gap = max(gap, (s1 - latest).total_seconds() / 60)
        latest = e1 if latest is None else max(latest, e1)
    if gap > max_gap_min:
        problems.append(
            f"{gap:.0f} min pass between two of the measurements (at most {max_gap_min:g})"
        )
    nodes = sorted({i.get("node") or "?" for i in known})
    out = {"max_gap_min": round(gap, 1), "nodes": nodes}
    if problems:
        return {**out, "status": FAILED, "why": "; ".join(problems)}
    if unknown or len(known) < 2:
        why = (
            f"no measurement window for {', '.join(unknown)}"
            if unknown
            else "fewer than two windows"
        )
        return {**out, "status": UNCHECKED, "why": why}
    if len(nodes) > 1:
        out["note"] = (
            f"the arms were measured on {len(nodes)} nodes ({', '.join(nodes)}): the A/B carries their difference"
        )
    return {**out, "status": OK}


def retest(points):
    """The retest noise of formal points [(value, node)] of one stack at one concurrency: (pct, n) from the node with
    the most of them (the larger noise of equals), None with fewer than two on one node."""
    by_node = {}
    for value, node in points:
        by_node.setdefault(node or "?", []).append(value)
    best = None
    for vs in by_node.values():
        if len(vs) >= 2:
            cand = (len(vs), round(RETEST_FACTOR * pct_rsd(vs), 2))
            best = max(best, cand) if best else cand
    return (best[1], best[0]) if best else None


def same_node_groups(points):
    """{node: [values]} of formal points [(value, node)]."""
    out = {}
    for value, node in points:
        out.setdefault(node or "?", []).append(value)
    return out


def decide(v, g, project_accept_pct):
    """judge()'s numbers verdict v with the gates g applied: the decision a verdict is written with. Whether the
    measurement can tell a change of the bar's size is judged against the project's accept bar (its snapshot), not a
    bar raised on the command line: a stricter bar for one A/B must not turn 判不了 · 待集成 into plain noise."""
    notes, numbers = [], (v["decision"], v["notes"])
    for name, gate in (("reproduce", g.get("reproduce")), ("timing", g.get("timing"))):
        if gate and gate["status"] == FAILED:
            v["decision"] = "rerun"
            notes.append(
                gate["why"] if name == "reproduce" else f"the arms do not compare: {gate['why']}"
            )
    acc = g.get("accuracy") or {}
    if not notes and v["decision"] == "accept" and acc.get("status") == "worse":
        v["decision"] = "reject"
        notes.append(acc["why"])
    fixed = g["noise"]["floor_pct"]
    v["resolvable"] = fixed <= project_accept_pct
    if not notes and v["decision"] == "noise" and not v["resolvable"]:
        notes.append(
            f"this measurement cannot tell a change of the bar's size (noise floor {fixed:.2f}% > accept bar "
            f"{project_accept_pct:g}%) · {TO_INTEGRATE}"
        )
    unchecked = [
        k for k in ("reproduce", "timing", "noise") if (g.get(k) or {}).get("status") == UNCHECKED
    ]
    if unchecked:
        notes.append("unchecked: " + ", ".join(f"{k} ({g[k]['why']})" for k in unchecked))
    if (g.get("timing") or {}).get("note"):
        notes.append(g["timing"]["note"])
    if v["decision"] != numbers[0]:
        notes.append(f"on the numbers alone {numbers[0]}: {numbers[1]}")
    elif numbers[1]:
        notes.append(numbers[1])
    v["notes"] = "; ".join(notes)
    return v


def statuses(g):
    """{gate: status} of a gates record: what verify compares."""
    return {k: (g.get(k) or {}).get("status") for k in ("reproduce", "timing", "noise", "accuracy")}


# ---------------------------------------------------------------- reading what the gates need


def series_points(table, series, task):
    """The reference of a series for an A/B in task: ledger.reference() without the task's own points and other
    experiments' base arms (it has neither when a sealing task measured the series)."""
    return [
        r
        for r in ledger.reference(table, series)
        if r["task"] != task and not (r["type"] == "experiment" and r["role"] == "baseline")
    ]


def task_of(path, root):
    """'type/name' of the task whose workspace holds a result file (workspaces/<type>/<name>/results/x.json)."""
    try:
        parts = Path(path).resolve().relative_to(Path(root).resolve() / "workspaces").parts
    except ValueError:
        return ""
    return "/".join(parts[:2]) if len(parts) >= 4 else ""


def is_version(series):
    return bool(re.fullmatch(r"v\d+", series or ""))
