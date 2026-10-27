"""GSM8K, the one accuracy gate (design §5.6): an lm_eval run converted into an accuracy result, and the two rules it
is judged by.

The ATOM kit's engine/eval.sh runs lm_eval with --log_samples: every question's outcome is in its
samples_gsm8k_<time>.jsonl (one line per question and filter, exact_match 1 or 0). `ledger gsm8k` counts the questions
answered right under the strict-match filter and writes

  {"schema": "infera.gsm8k/v1", "label": "v0-full", "series": "v0", "correct": 1262, "total": 1319,
   "accuracy": 0.956785, "ci95": [0.944, 0.967], "filter": "strict-match", "wrong": [12, 40, ...],
   "evidence": {"samples": <path relative to the project directory>, "samples_sha256": ..., "results": ..., ...}}

into the task's results/ as <label>.gsm8k.json. It is no performance point: the ledger's curves never read it. A rule
counts the questions again from the samples file, whose digest must still be the one recorded.

The rules (project.yaml quality.gsm8k_margin, 0.02 by default):
  - an experiment's accept: every GSM8K of the candidate's series in its task (200 questions will do) against the
    version's, on the same questions: the 95% Newcombe interval of the difference must not lie wholly below 0
    (significantly worse);
  - a round's close that seals vN: vN's GSM8K over all 1319 questions against the previous version's, the interval's
    lower bound not below -gsm8k_margin.
A version's GSM8K is the one its sealing task measured (v0: the baseline task; vN: close/rN), the one with the most
questions when it has several.

Intervals: wilson() and newcombe() are copied from agent_sys/examples/llm_e2e_performance_optimization/e2e-flow/assets/
lib/eval_stats.py, which ported them from the kit's summarise_eval.py. 200 questions are a coin flipped 200 times:
at p=0.96 a run's interval is about ±2.8 points and the difference of two runs about ±4.2; at 1319 questions the
difference is about ±1.5. So 200 questions catch a gross loss only, and non-inferiority within 2 points needs the
full set.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "infera.gsm8k/v1"
FILTER = "strict-match"  # the answer after "#### ": a stack that garbles its format loses it
FULL = 1319  # the questions of GSM8K's test split
MIN_ACCEPT = 200  # the questions a candidate's GSM8K asks at least, before an accept
SAMPLES = re.compile(
    r"samples_gsm8k_\d.*\.jsonl"
)  # lm_eval's gsm8k task, not gsm8k_cot or another variant
DEFAULT_MARGIN = (
    0.02  # project.yaml quality.gsm8k_margin: how much worse a sealed version may be, at most
)
SUFFIX = ".gsm8k.json"
Z95 = 1.959963984540054


# ---------------------------------------------------------------- intervals (from v1's eval_stats.py)


def wilson(successes, n, z=Z95):
    """95% Wilson score interval for a proportion.

    Wilson and not Wald. The Wald interval has zero width at p=0 and p=1, which is exactly where an eval that is
    working perfectly or failing completely lands, and a zero-width interval says "certain" about a sample of 200.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe(succ_a, n_a, succ_b, n_b, z=Z95):
    """95% interval for (p_b - p_a), by Newcombe's composition of two Wilson intervals.

    Not a normal approximation to the difference: the two proportions are measured on samples that are small and
    often near 1.0, where the normal approximation understates the interval on the side that matters.
    """
    lo_a, hi_a = wilson(succ_a, n_a, z)
    lo_b, hi_b = wilson(succ_b, n_b, z)
    p_a = succ_a / n_a if n_a else 0.0
    p_b = succ_b / n_b if n_b else 0.0
    delta = p_b - p_a
    lower = delta - math.sqrt((p_b - lo_b) ** 2 + (hi_a - p_a) ** 2)
    upper = delta + math.sqrt((hi_b - p_b) ** 2 + (p_a - lo_a) ** 2)
    return (lower, upper)


# ---------------------------------------------------------------- lm_eval's samples -> counts


def samples_file(path):
    """The samples_gsm8k_*.jsonl of an lm_eval output directory (or the file itself). Raises ValueError unless there is
    exactly one."""
    path = Path(path)
    if path.is_file():
        return path
    found = (
        sorted(p for p in path.rglob("samples_gsm8k*.jsonl") if SAMPLES.fullmatch(p.name))
        if path.is_dir()
        else []
    )
    if not found:
        raise ValueError(
            f"no samples_gsm8k*.jsonl under {path}: run lm_eval with --log_samples (the kit's engine/eval.sh does)"
        )
    if len(found) > 1:
        raise ValueError(
            f"{len(found)} lm_eval runs under {path} ({', '.join(p.name for p in found)}): name the samples file"
        )
    return found[0]


def results_file(samples):
    """lm_eval's results_<time>.json of the same run as a samples file, None without one."""
    m = re.fullmatch(r"samples_gsm8k_?(.*)\.jsonl", Path(samples).name)
    stamp = m[1] if m else ""
    found = sorted(Path(samples).parent.glob(f"results_{stamp}.json" if stamp else "results*.json"))
    return found[0] if len(found) == 1 else None


def count(data):
    """{filter: {doc_id: right}} from the bytes of a samples file. Raises ValueError when it is not one."""
    out = {}
    for n, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            doc, em = row["doc_id"], row["exact_match"]
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(
                f"line {n} is not an lm_eval sample with doc_id and exact_match: {e}"
            ) from None
        if isinstance(doc, bool) or not isinstance(doc, int) or em not in (0, 1):
            raise ValueError(f"line {n}: doc_id {doc!r}, exact_match {em!r}")
        per = out.setdefault(str(row.get("filter") or FILTER), {})
        if doc in per:
            raise ValueError(
                f"line {n}: question {doc} appears twice under filter {row.get('filter')}"
            )
        per[doc] = em == 1
    if not out:
        raise ValueError("no samples in it")
    return out


def tally(per_doc, ids=None):
    """(right, asked) over the questions ids (all by default)."""
    ids = per_doc.keys() if ids is None else ids
    return sum(per_doc[i] for i in ids), len(ids)


def from_lm_eval(path, root, label, series, round_=None, time=""):
    """The accuracy result of an lm_eval run (its output directory or samples file). Raises ValueError."""
    from . import verdict

    samples = samples_file(path)
    data = samples.read_bytes()
    per = count(data)
    if FILTER not in per:
        raise ValueError(f"{samples.name} has no {FILTER} filter (it has {', '.join(per)})")
    right, asked = tally(per[FILTER])
    evidence = {
        "samples": verdict.rel(samples, root),
        "samples_sha256": hashlib.sha256(data).hexdigest(),
    }
    config = {}
    res = results_file(samples)
    if res:
        raw = res.read_bytes()
        try:
            summary = json.loads(raw)
            stated = summary["results"]["gsm8k"].get(f"exact_match,{FILTER}")
        except (ValueError, KeyError, TypeError, AttributeError):
            summary, stated = {}, None
        # lm_eval's own summary of the same run must say what the samples say
        if isinstance(stated, (int, float)) and not math.isclose(
            stated, right / asked, abs_tol=1e-9
        ):
            raise ValueError(
                f"{res.name} says exact_match {stated} under {FILTER}, its samples {right}/{asked}: not one run"
            )
        summary = summary if isinstance(summary, dict) else {}
        cfg = summary.get("config") if isinstance(summary.get("config"), dict) else {}
        args = cfg.get("model_args") if isinstance(cfg.get("model_args"), dict) else {}
        shots = summary.get("n-shot") if isinstance(summary.get("n-shot"), dict) else {}
        config = {
            "model": args.get("model") or summary.get("model_name"),
            "base_url": args.get("base_url"),
            "limit": cfg.get("limit"),
            "num_fewshot": shots.get("gsm8k"),
            "lm_eval_version": summary.get("lm_eval_version"),
        }
        config = {k: v for k, v in config.items() if v not in (None, "")}
        evidence |= {
            "results": verdict.rel(res, root),
            "results_sha256": hashlib.sha256(raw).hexdigest(),
        }
    lo, hi = wilson(right, asked)
    return {
        "schema": SCHEMA,
        "label": label,
        "time": time,
        **({"round": round_} if round_ is not None else {}),
        "series": series,
        "filter": FILTER,
        "correct": right,
        "total": asked,
        "accuracy": round(right / asked, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "wrong": sorted(i for i, ok in per[FILTER].items() if not ok),
        "other_filters": {
            f: dict(zip(("correct", "total"), tally(d))) for f, d in per.items() if f != FILTER
        },
        "config": config,
        "evidence": evidence,
    }


def convert(paths, out, label, series, round_=None, time=""):
    """Write <out>/<label>.gsm8k.json from one lm_eval run. Returns (files written, errors)."""
    from . import verdict
    from .model import iso, write_json

    if len(paths) != 1:
        return [], ["one lm_eval run per label: name one output directory or samples file"]
    if "/" in label or not label.strip():
        return [], [f"{label!r}: a label names a file, so it is a word with no /"]
    out = Path(out)
    target = out / f"{label}{SUFFIX}"
    if target.exists():
        return [], [f"{target} exists already: give another run its own --label"]
    try:
        samples = samples_file(paths[0])
        doc = from_lm_eval(
            samples,
            verdict.project_root(out),
            label,
            series,
            round_,
            time or iso(samples.stat().st_mtime),
        )
    except (OSError, UnicodeDecodeError, ValueError) as e:
        return [], [f"{paths[0]}: {e}"]
    write_json(target, doc)
    return [target], []


# ---------------------------------------------------------------- a result, counted again from its samples


def is_accuracy(doc):
    return isinstance(doc, dict) and doc.get("schema") == SCHEMA


def validate(doc):
    """The reasons an accuracy result is malformed, [] when it is not."""
    errors = []
    for key in ("correct", "total"):
        v = doc.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            errors.append(f"{key} should be a whole number")
    if not errors and doc["correct"] > doc["total"]:
        errors.append("correct is more than total")
    if not str(doc.get("label") or "").strip():
        errors.append("label is missing")
    if not str(doc.get("series") or "").strip():
        errors.append("series is missing (the stack it measured: v0, r1-spf ...)")
    if doc.get("filter", FILTER) != FILTER:
        errors.append(f"filter should be {FILTER}")
    ev = doc.get("evidence") if isinstance(doc.get("evidence"), dict) else {}
    if not (isinstance(ev.get("samples"), str) and len(str(ev.get("samples_sha256") or "")) == 64):
        errors.append("evidence.samples and samples_sha256 are missing: convert with ledger gsm8k")
    return errors


def recount(doc, root):
    """({doc_id: right} under the filter, problem): the questions counted again from the samples file the result was
    converted from; problem '' when they can be, and the counts match the file's."""
    from . import verdict

    ev = doc.get("evidence") if isinstance(doc.get("evidence"), dict) else {}
    if errors := validate(doc):
        return None, "; ".join(errors)
    path = verdict.resolve(ev["samples"], root)
    try:
        data = path.read_bytes()
    except OSError:
        return None, f"its lm_eval samples {ev['samples']} are gone"
    digest = hashlib.sha256(data).hexdigest()
    if digest != ev["samples_sha256"]:
        return None, (
            f"its lm_eval samples {ev['samples']} changed after the conversion (sha256 {digest[:12]}…, recorded "
            f"{ev['samples_sha256'][:12]}…)"
        )
    try:
        per = count(data).get(FILTER)  # whatever filter the file names: the gate is strict-match's
    except (UnicodeDecodeError, ValueError) as e:
        return None, f"its lm_eval samples {ev['samples']} cannot be read: {e}"
    if per is None:
        return None, f"its lm_eval samples have no {FILTER} filter"
    if tally(per) != (doc["correct"], doc["total"]):
        return None, (
            f"it says {doc['correct']}/{doc['total']}, its lm_eval samples {tally(per)[0]}/{tally(per)[1]}"
        )
    return per, ""


def compare(ver, cand):
    """The candidate's per-question outcomes against the version's, on the questions both asked: {right, asked of each,
    delta, ci95 (Newcombe, candidate - version)}."""
    common = sorted(set(ver) & set(cand))
    if not common:
        return None
    kv, n = tally(ver, common)
    kc, _n = tally(cand, common)
    lo, hi = newcombe(kv, n, kc, n)
    return {
        "questions": n,
        "version": [kv, n],
        "candidate": [kc, n],
        "delta": round((kc - kv) / n, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
    }


# ---------------------------------------------------------------- where a stack's GSM8K is


def results_of(workspace):
    """[(path, doc)] of the accuracy results at the top of a task's results/."""
    out = []
    for p in sorted(Path(workspace, "results").glob(f"*{SUFFIX}")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if is_accuracy(doc):
            out.append((p, doc))
    return out


def of_version(project, series):
    """(path, doc) of a version's GSM8K: the one its sealing task measured with the most questions (the latest of
    equals); None without one."""
    from . import ledger

    found = []
    for t in project.tasks.values():
        if not ledger.sealed_by({"type": t.type, "task": t.id}, series):
            continue
        for p, doc in results_of(t.workspace):
            if doc.get("series") == series and not validate(doc):
                found.append((doc["total"], doc["correct"] / max(doc["total"], 1), p, doc))
    # the most questions; of equals the best score: the strictest reference for what is compared with it
    return max(found, key=lambda f: f[:2])[2:] if found else None


def previous_version(project, n):
    """The version before vN that has a GSM8K: v(N-1), ... v0; None when none has."""
    for k in range(n - 1, -1, -1):
        if of_version(project, f"v{k}"):
            return f"v{k}"
    return None


def margin_of(config):
    """project.yaml's quality.gsm8k_margin: a number in [0, 1), else the default."""
    from .model import plain

    q = plain((config or {}).get("quality")) or {}
    v = q.get("gsm8k_margin") if isinstance(q, dict) else None
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v < 1
    return float(v) if ok else DEFAULT_MARGIN


def accept_gate(project, root, workspace, cand_series, version):
    """Whether an experiment's candidate may be accepted on accuracy: {status ok | worse | missing | unverifiable, why,
    and per file the comparison}. Every GSM8K in the task's results/ counts, whatever series it says (no picking; a
    base arm's own run must not be significantly worse either): it is a run of this task (its samples under the
    task's workspace, not the version's samples), and at least one of the candidate's series asks MIN_ACCEPT
    questions."""
    from . import verdict

    fix = (
        f'run engine/eval.sh on the candidate stack (EVAL_LIMIT={MIN_ACCEPT}), then "$AGENT_SYS_PYTHON" -m '
        f"agent_sys_v2.ledger gsm8k <run>/gsm8k --out <results> --label <label> --series {cand_series}"
    )
    found = []
    for p in sorted(Path(workspace, "results").glob(f"*{SUFFIX}")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as e:
            return {
                "status": "unverifiable",
                "why": f"{verdict.rel(p, root)} is not JSON: {e}",
                "fix": fix,
            }
        errors = validate(d) if is_accuracy(d) else [f'schema should be "{SCHEMA}"']
        if errors:
            return {
                "status": "unverifiable",
                "why": f"{verdict.rel(p, root)}: {'; '.join(errors)}",
                "fix": fix,
            }
        found.append((p, d))
    mine = [(p, d) for p, d in found if d.get("series") == cand_series and d["total"] >= MIN_ACCEPT]
    if not mine:
        asked = ", ".join(
            f"{verdict.rel(p, root)} ({d.get('series')}, {d['total']})" for p, d in found
        )
        return {
            "status": "missing",
            "why": f"no GSM8K of series {cand_series} over {MIN_ACCEPT}+ questions in the task's results/"
            + (f" (there are: {asked})" if asked else ""),
            "fix": fix,
        }
    base = of_version(project, version)
    if not base:
        return {
            "status": "missing",
            "why": f"version {version} has no GSM8K from its sealing task",
            "fix": f"the task that sealed {version} measures it (its full GSM8K, ledger gsm8k --series {version})",
        }
    ver, problem = recount(base[1], root)
    if problem:
        return {
            "status": "unverifiable",
            "why": f"{verdict.rel(base[0], root)}: {problem}",
            "fix": "",
        }
    out = {"status": "ok", "version": verdict.rel(base[0], root), "files": []}
    ws = Path(workspace).resolve()
    for p, d in found:
        samples = verdict.resolve(d["evidence"]["samples"], root).resolve()
        if (
            ws not in samples.parents
            or d["evidence"]["samples_sha256"] == base[1]["evidence"]["samples_sha256"]
        ):
            return {
                "status": "unverifiable",
                "why": f"{verdict.rel(p, root)} was converted from {d['evidence']['samples']}: not a run of this task "
                "(its lm_eval output belongs under the task's workspace)",
                "fix": fix,
            }
        cand, problem = recount(d, root)
        if problem:
            return {
                "status": "unverifiable",
                "why": f"{verdict.rel(p, root)}: {problem}",
                "fix": fix,
            }
        cmp = compare(ver, cand)
        if cmp is None:
            return {
                "status": "unverifiable",
                "why": f"{verdict.rel(p, root)} asked none of the questions {version}'s GSM8K asked",
                "fix": fix,
            }
        out["files"].append({"file": verdict.rel(p, root), **cmp})
        if cmp["ci95"][1] < 0 and out["status"] == "ok":
            out["status"] = "worse"
            out["why"] = (
                f"GSM8K is significantly worse: {cmp['candidate'][0]}/{cmp['questions']} against {version}'s "
                f"{cmp['version'][0]} on the same questions (difference {cmp['delta']:+.3f}, 95% "
                f"[{cmp['ci95'][0]:+.3f}, {cmp['ci95'][1]:+.3f}])"
            )
    return out


def seal_gate(project, root, workspace, version, margin):
    """Whether a round's close may seal version (vN) on accuracy: [(ok, line)] for the report. Every full GSM8K of vN
    in the task counts (no picking), each against the previous version's."""
    from . import verdict

    mine = [
        (p, d) for p, d in results_of(workspace) if d.get("series") == version and not validate(d)
    ]
    full = [(p, d) for p, d in mine if d["total"] >= FULL]
    if not full:
        asked = ", ".join(f"{verdict.rel(p, root)} ({d['total']})" for p, d in mine) or "none"
        return [
            (False, f"results/ has {version}'s GSM8K over all {FULL} questions (it has: {asked})")
        ]
    m = re.fullmatch(r"v(\d+)", version)
    prev = previous_version(project, int(m[1])) if m else None
    if not prev:
        return [(False, f"a version before {version} has a GSM8K to compare with (none has)")]
    pp, pd = of_version(project, prev)
    old, problem = recount(pd, root)
    if problem:
        return [
            (False, f"{verdict.rel(pp, root)} counts again from its lm_eval samples: {problem}")
        ]
    lines = []
    for p, d in full:
        new, problem = recount(d, root)
        if problem:
            lines.append(
                (False, f"{verdict.rel(p, root)} counts again from its lm_eval samples: {problem}")
            )
            continue
        cmp = compare(old, new)
        ok = cmp is not None and cmp["ci95"][0] >= -margin
        lines.append(
            (
                ok,
                f"GSM8K of {version} ({verdict.rel(p, root)}) is not worse than {prev}'s by more than "
                f"{margin:g}: "
                + (
                    f"{cmp['candidate'][0]} vs {cmp['version'][0]} of {cmp['questions']} questions, difference "
                    f"{cmp['delta']:+.4f}, 95% [{cmp['ci95'][0]:+.4f}, {cmp['ci95'][1]:+.4f}]"
                    if cmp
                    else f"no question in common with {verdict.rel(pp, root)}"
                ),
            )
        )
    return lines


def add_parser(sub):
    g = sub.add_parser(
        "gsm8k", help="write an accuracy result (<label>.gsm8k.json) from an lm_eval GSM8K run"
    )
    g.add_argument(
        "paths", nargs="+", help="lm_eval's output directory (or its samples_gsm8k_*.jsonl)"
    )
    g.add_argument("--out", required=True, help="the results directory to write into")
    g.add_argument("--label", required=True, help="the file is <label>.gsm8k.json")
    g.add_argument(
        "--series",
        required=True,
        help="the stack it measured: a version (v0) or an experiment's series",
    )
    g.add_argument("--round", type=int, help="the project's round")
    g.add_argument("--time", default="", help="ISO time (default: the samples file's mtime)")
    return g


def main_args(args):
    """`ledger gsm8k`: (exit code, lines)."""
    written, errors = convert(args.paths, args.out, args.label, args.series, args.round, args.time)
    return (0 if written else 1), [f"wrote {w}" for w in written] + errors
