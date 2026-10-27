"""A/B verdicts that can be recomputed from the AgentX output they were judged on (design §5.6).

Performance is judged on AgentX numbers only. A result `ledger agentx` converted carries its source in evidence:
the raw agentx_conc<C>.json (path relative to the project directory, and its sha256) and the run directory's
runtime.env. Only such results are inputs of a verdict; any other result is in the ledger for reference.

ab_judge.py (skill infera-experiment) judges with judge() below and writes into each candidate file a verdict with
its inputs: the metric and its direction, the X metric, the files of each arm (with the digest of their raw output)
and the values it read, the noise band and accept bar and where they came from. The bar and band are the project's
(project.yaml ledger.noise_pct and accept_pct, 2 and 3 by default); a run snapshots them into its launch.json, and
the command line may only make them stricter. A measurement is void for one of two reasons: more than 10% failed
requests (computed from the AgentX counts, not stated), or another reason written down with an evidence file.

  "$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger verify <dir or file> ...

checks every candidate result as `check --require-verdict` does, then recomputes its verdict from the source: the
inputs' raw files are there with the digest recorded, converting them again gives the metrics the result files hold,
every valid result of the same series and concurrency in an input's task is an input (no picking), the thresholds
are not looser than the run's snapshot (and project.yaml still says what the snapshot says), and judge() on the raw
numbers gives the same decision, vs_baseline_pct and noise floor. A failure is one of two kinds: the verdict does
not match the data (MISMATCH), or the verdict cannot be recomputed (UNVERIFIABLE: no AgentX source, a raw file
gone, no inputs); each says the problem, its context and how to fix it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from . import ledger
from .model import plain, read_json, read_text, runs_dir

DEFAULT_NOISE_PCT = ledger.DEFAULT_NOISE_PCT
DEFAULT_ACCEPT_PCT = ledger.DEFAULT_ACCEPT_PCT
FAILED_MAX_PCT = 10.0  # more failed requests than this voids a measurement
VOID_KINDS = ("failed_requests", "other")
# the project's rules a verdict is judged by: snapshotted into launch.json when a run starts
RULE_KEYS = (
    "primary",
    "higher_is_better",
    "x",
    "x_higher_is_better",
    "baseline_series",
    "noise_pct",
    "accept_pct",
)
# the runner hands a check the rules it snapshotted when the run started (launch.json is in the agent's reach)
RULES_ENV = "AGENT_SYS_RULES"
MISMATCH = "the verdict does not match the data"
UNVERIFIABLE = "the verdict cannot be recomputed"
REJUDGE = (
    "judge again with ab_judge.py --write (skill infera-experiment); never edit a verdict by hand"
)


# ---------------------------------------------------------------- the judgment


def rsd_pct(xs):
    """Relative standard deviation in percent; 0 for a single value or equal ones."""
    if len(set(xs)) < 2:
        return 0.0
    mean = statistics.mean(xs)
    if not mean:
        raise ValueError("an arm's values differ around a mean of 0: no relative spread")
    return statistics.stdev(xs) / abs(mean) * 100


def compare(base, cand, higher_is_better, noise_pct):
    """(gain in percent, its noise floor) of cand against base."""
    if not base or not cand:
        raise ValueError("both arms need at least one value")
    if not all(ledger.number(v) for v in base + cand):
        raise ValueError("every value must be a finite number")
    mb, mc = statistics.mean(base), statistics.mean(cand)
    if mb == 0:
        raise ValueError("the baseline's mean is 0: no relative difference")
    spread = 1.96 * math.sqrt(rsd_pct(base) ** 2 / len(base) + rsd_pct(cand) ** 2 / len(cand))
    return ledger.gain(ledger.pct(mc, mb), higher_is_better), max(noise_pct, spread)


def judge(
    base,
    cand,
    higher_is_better=True,
    noise_pct=DEFAULT_NOISE_PCT,
    accept_pct=DEFAULT_ACCEPT_PCT,
    x=None,
):
    """The verdict of cand against base, on the numbers alone. x: (baseline X values, candidate X values, X higher is
    better) or None. A gain inside the noise floor (the larger of noise_pct and 1.96 x the relative standard error of
    the difference) is noise, a loss beyond it a reject; an accept needs the gain past the bar and the floor, the
    medians moving the way the means do, and X not worse beyond its own floor; anything else asks for a rerun."""
    gain, floor = compare(base, cand, higher_is_better, noise_pct)
    x_gain, x_floor = compare(*x, noise_pct) if x else (0.0, 0.0)
    mb, mc = statistics.mean(base), statistics.mean(cand)
    median_agrees = (
        len(base) < 2
        or len(cand) < 2
        or (statistics.median(cand) - statistics.median(base)) * (mc - mb) > 0
    )
    if abs(gain) < floor:
        decision, note = "noise", f"|{gain:+.2f}%| is inside the noise floor {floor:.2f}%"
    elif gain < 0:
        decision, note = "reject", f"worse by {-gain:.2f}% (floor {floor:.2f}%)"
    elif gain < max(accept_pct, floor):
        decision = "rerun"
        note = f"{gain:+.2f}% is between the noise floor and the {accept_pct:g}% bar: run again"
    elif not median_agrees:
        decision, note = "rerun", "the medians disagree with the means: run again"
    elif x and x_gain <= -x_floor:
        decision = "rerun"
        note = f"X is worse by {-x_gain:.2f}% (floor {x_floor:.2f}%): compare at equal X (the working point)"
    else:
        decision = "accept"
        note = f"{gain:+.2f}% past the {accept_pct:g}% bar and the floor {floor:.2f}%"
    return {
        "status": "valid",
        "decision": decision,
        "vs_baseline_pct": ledger.pct(mc, mb),
        "noise_floor_pct": round(floor, 2),
        **({"x_gain_pct": x_gain} if x else {}),
        "n_baseline": len(base),
        "n_candidate": len(cand),
        "mean_baseline": mb,
        "mean_candidate": mc,
        "notes": note,
    }


def failed_pct(metrics):
    """The failed requests of an AgentX result in percent of all it sent (errors / (successful + errors)); None
    without both counts."""
    ok, bad = metrics.get("requests"), metrics.get("errors")
    if not (ledger.number(ok) and ledger.number(bad)) or ok + bad <= 0:
        return None
    return round(bad / (ok + bad) * 100, 2)


# ---------------------------------------------------------------- the project's rules


def rules(config):
    """The rules a verdict is judged by, from project.yaml (its ledger section): RULE_KEYS."""
    cfg = ledger.settings_from(config)
    return {k: cfg[k] for k in RULE_KEYS}


def project_root(path):
    """The project directory a file belongs to: $AGENT_SYS_PROJECT when the file is under it, else the nearest
    directory above it with a project.yaml; None outside any project."""
    path = Path(path).resolve()
    env = os.environ.get("AGENT_SYS_PROJECT")
    if env and (Path(env).resolve() == path or Path(env).resolve() in path.parents):
        return Path(env).resolve()
    return next((d for d in path.parents if (d / "project.yaml").is_file()), None)


def project_rules(root):
    """(rules, where they came from) of the project at root; the defaults without a project."""
    if root is None:
        return rules({}), "defaults (no project.yaml found)"
    from .model import load_yaml

    try:
        config = plain(load_yaml(read_text(Path(root) / "project.yaml")))
    except ValueError as e:
        raise ValueError(f"{root}/project.yaml cannot be read: {e}") from None
    return rules(config), "project.yaml"


def rel(path, root):
    """path relative to the project directory when it is inside it, else absolute."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(root).resolve())) if root else str(path)
    except ValueError:
        return str(path)


def resolve(text, root):
    path = Path(str(text))
    return path if path.is_absolute() or root is None else Path(root) / path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(workspace):
    """(the rules snapshotted by the task's run, the run's id): what the runner hands the check it runs now
    ($AGENT_SYS_RULES, from the launch.json it read before the agent started), else the launch.json of the run whose
    check runs now ($AGENT_SYS_CHECK_RUN) or of the latest run of the workspace that took one; (None, '') when no run
    did."""
    from .procs import CHECK_MARK

    try:
        given = json.loads(os.environ.get(RULES_ENV) or "null")
    except ValueError:
        given = None
    if isinstance(given, dict):
        return given, "in flight"
    workspace = Path(workspace).resolve()
    mark = os.environ.get(CHECK_MARK)
    candidates = (
        [Path(mark)]
        if mark and Path(mark).resolve().parent == runs_dir(workspace).resolve()
        else []
    )
    d = runs_dir(workspace)
    candidates += sorted(d.iterdir(), reverse=True) if d.is_dir() else []
    for run in candidates:
        cfg = read_json(run / "launch.json")
        if isinstance(cfg.get("ledger"), dict):
            return cfg["ledger"], run.name
    return None, ""


# ---------------------------------------------------------------- the AgentX source of a result


def source(doc):
    """(raw file as recorded, its sha256) of a result `ledger agentx` wrote; None for any other result."""
    ev = doc.get("evidence") if isinstance(doc, dict) else None
    if not isinstance(ev, dict):
        return None
    raw, digest = ev.get("agentx"), ev.get("agentx_sha256")
    if isinstance(raw, str) and raw and isinstance(digest, str) and len(digest) == 64:
        return raw, digest
    return None


def reconvert(doc, root):
    """(the metrics converting the raw file again gives, the raw file's conc, problems [(kind, problem, fix)])."""
    src = source(doc)
    if not src:
        return (
            None,
            None,
            [
                (
                    UNVERIFIABLE,
                    "it has no AgentX source (evidence.agentx and agentx_sha256)",
                    CONVERT,
                )
            ],
        )
    raw = resolve(src[0], root)
    try:
        data = raw.read_bytes()
    except OSError:
        return None, None, [(UNVERIFIABLE, f"its raw AgentX output {src[0]} is gone", KEEP_RAW)]
    if hashlib.sha256(data).hexdigest() != src[1]:
        return (
            None,
            None,
            [
                (
                    MISMATCH,
                    f"its raw AgentX output {src[0]} changed after the conversion "
                    f"(sha256 {hashlib.sha256(data).hexdigest()[:12]}…, recorded {src[1][:12]}…)",
                    KEEP_RAW,
                )
            ],
        )
    try:
        again = ledger.from_agentx(json.loads(data), raw, "s")
    except (UnicodeDecodeError, ValueError) as e:
        return (
            None,
            None,
            [(UNVERIFIABLE, f"its raw AgentX output {src[0]} cannot be converted: {e}", KEEP_RAW)],
        )
    return again["metrics"], again["workload"].get("conc"), []


CONVERT = (
    'convert the AgentX output with "$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger agentx <run dir> --out <results> '
    "(skill infera-bench); a result written any other way is in the ledger for reference only"
)
KEEP_RAW = (
    "keep the run directory the result was converted from where it was, unchanged; if it is lost, measure again "
    "and convert the new output"
)


def same(a, b):
    return ledger.number(a) and ledger.number(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def metric_diffs(doc, metrics):
    """Where a result's metrics differ from what its raw output gives: ['name: file F, raw R']."""
    held = doc.get("metrics") or {}
    return [
        f"{k}: file {held.get(k)!r}, raw {v!r}"
        for k, v in metrics.items()
        if not same(held.get(k), v)
    ]


# ---------------------------------------------------------------- void


def void_problems(doc, root):
    """Why a void verdict is not one of the two kinds: [(kind, problem, fix)], [] when it is."""
    verdict = doc.get("verdict") or {}
    void = verdict.get("void")
    kind = void.get("kind") if isinstance(void, dict) else None
    fix = (
        "void with ab_judge.py --void failed-requests (when AgentX counts more than 10% failed requests) or "
        '--void other --reason "<why>" --evidence <file>'
    )
    if kind not in VOID_KINDS:
        return [
            (MISMATCH, f"verdict.void.kind is {kind!r}, not one of {', '.join(VOID_KINDS)}", fix)
        ]
    if kind == "other":
        out = []
        if not str(void.get("reason") or "").strip():
            out.append((MISMATCH, "a void of kind other has no reason", fix))
        evidence = str(void.get("evidence") or "").strip()
        if not evidence or not resolve(evidence, root).is_file():
            out.append(
                (
                    MISMATCH,
                    f"the evidence file of the void ({evidence or 'none'}) is not there",
                    fix,
                )
            )
        return out
    metrics, _conc, problems = reconvert(doc, root)
    if problems:
        return problems
    pct = failed_pct(metrics)
    if pct is None or pct <= FAILED_MAX_PCT:
        return [
            (
                MISMATCH,
                f"voided for failed requests, but AgentX counts {pct if pct is not None else 'no'}% failed "
                f"(the bar is more than {FAILED_MAX_PCT:g}%)",
                fix,
            )
        ]
    return []


# ---------------------------------------------------------------- the inputs of a verdict


def conc_of(doc, root):
    """A result's concurrency: its raw AgentX output's when it has one that can be read (the file's own field can be
    edited), else workload.conc."""
    metrics, conc, _problems = reconvert(doc, root) if source(doc) else (None, None, [])
    return conc if metrics is not None else (doc.get("workload") or {}).get("conc")


def peers(path, conc, root):
    """The results at the top of the results/ directory of path whose concurrency is conc, of any series and role
    (those fields can be edited to hide one): [(path, doc)]."""
    out = []
    for p in ledger.json_files(Path(path).parent):
        other, errors = ledger.load(p)
        if other is not None and not errors and conc_of(other, root) == conc:
            out.append((p, other))
    return out


def arm(files, metric, x_metric, root):
    """The inputs of one arm from its result files: [{file, agentx, agentx_sha256, value, x}]. Raises ValueError
    with every reason a file cannot be an input."""
    out, why = [], []
    for f in files:
        path = Path(f)
        doc, errors = ledger.read(path)
        if doc is None or errors:
            why.append(f"{f}: {'; '.join(errors) or 'not an infera.result file'}")
            continue
        if (doc.get("verdict") or {}).get("status") == "void":
            why.append(f"{f}: it is void, so it is no input")
            continue
        metrics, _conc, problems = reconvert(doc, root)
        if problems:
            why += [f"{f}: {p} ({fix})" for _k, p, fix in problems]
            continue
        if diffs := metric_diffs(doc, metrics):
            why.append(f"{f}: its metrics differ from its raw AgentX output ({'; '.join(diffs)})")
            continue
        if (pct := failed_pct(metrics)) is not None and pct > FAILED_MAX_PCT:
            why.append(f"{f}: {pct}% of its requests failed: void it (--void failed-requests)")
            continue
        value = metrics.get(metric)
        xv = metrics.get(x_metric) if x_metric else None
        if not ledger.number(value) or (x_metric and not ledger.number(xv)):
            why.append(
                f"{f}: metrics.{metric if not ledger.number(value) else x_metric} is not a number"
            )
            continue
        src = source(doc)
        out.append(
            {
                "file": rel(path, root),
                "agentx": src[0],
                "agentx_sha256": src[1],
                "value": value,
                **({"x": xv} if x_metric else {}),
            }
        )
    if why:
        raise ValueError("\n".join(why))
    return out


def picked(files, root):
    """The results that should be inputs and are not: every valid result at the inputs' concurrency in the results/ of
    an input's task, whatever its series or role (one A/B per concurrency per task). files: both arms. ['file (series
    s, C c)']."""
    have = {resolve(f, root).resolve() for f in files}
    missing = {}
    for f in files:
        path = resolve(f, root)
        doc, _errors = ledger.load(path)
        if not doc:
            continue
        conc = conc_of(doc, root)
        for p, other in peers(path, conc, root):
            void = (other.get("verdict") or {}).get("status") == "void"
            if not void and p.resolve() not in have:
                missing[p.resolve()] = f"{rel(p, root)} (series {other.get('series')}, C{conc})"
    return list(missing.values())


def void_peers(files, root):
    """The void results among the inputs' peers whose void is not one of the two kinds: [(file, problems)]."""
    seen, out = set(), []
    for f in files:
        path = resolve(f, root)
        doc, _errors = ledger.load(path)
        for p, other in peers(path, conc_of(doc, root), root) if doc else []:
            if p.resolve() in seen or (other.get("verdict") or {}).get("status") != "void":
                continue
            seen.add(p.resolve())
            if problems := void_problems(other, root):
                out.append((rel(p, root), problems))
    return out


def concurrencies(files, root):
    """{conc: [files]} of result files, by their raw AgentX output's concurrency."""
    out = {}
    for f in files:
        doc, _errors = ledger.load(resolve(f, root))
        if doc:
            out.setdefault(conc_of(doc, root), []).append(str(f))
    return out


# ---------------------------------------------------------------- verify


class Finding:
    def __init__(self, kind, problem, context="", fix=""):
        self.kind, self.problem, self.context, self.fix = kind, problem, context, fix

    def lines(self):
        out = [f"  [{self.kind}] {self.problem}"]
        if self.context:
            out.append(f"      context: {self.context}")
        if self.fix:
            out.append(f"      fix: {self.fix}")
        return out

    def as_dict(self):
        return {
            "kind": self.kind,
            "problem": self.problem,
            "context": self.context,
            "fix": self.fix,
        }


def verify(path, root=None, snap=None, current=None):
    """The findings of a result's verdict against its source, [] when it holds. root: the project directory
    (project_root by default); snap: the rules the run snapshotted (the task's latest run's by default, else
    project.yaml's); current: project.yaml's rules now, compared with the snapshot (the acceptance check: the rules
    must not change during the run)."""
    path = Path(path)
    root = project_root(path) if root is None else Path(root).resolve()
    doc, errors = ledger.read(path, require_verdict=True)
    if doc is None:
        return [
            Finding(
                UNVERIFIABLE,
                "; ".join(errors) or "not an infera.result file",
                "",
                "write the result with ledger agentx and the verdict with ab_judge.py --write; never by hand",
            )
        ]
    if errors:
        return [
            Finding(UNVERIFIABLE, e, fix="fix the file (skill infera-bench), then judge it again")
            for e in errors
        ]
    verdict = doc.get("verdict") or {}
    if not judged(doc):
        return []  # nothing to recompute: a baseline, a reference
    role = doc.get("role", "candidate")
    if role != "candidate" and verdict.get("status") != "void":
        # a decision is a candidate's: a role changed afterwards must not hide it from the recomputation
        return [
            Finding(
                MISMATCH,
                f"role is {role!r} but the file carries an A/B decision ({verdict.get('decision')!r}); a decision is "
                "a candidate's",
                "",
                "put role back to candidate (ledger agentx writes it), then " + REJUDGE,
            )
        ]
    if verdict.get("status") == "void":
        return [
            Finding(
                k, p, f"verdict.void: {json.dumps(verdict.get('void'), ensure_ascii=False)}", fix
            )
            for k, p, fix in void_problems(doc, root)
        ]
    inputs = verdict.get("inputs")
    if not isinstance(inputs, dict):
        return [
            Finding(
                UNVERIFIABLE,
                "the verdict has no inputs (verdict.inputs): it was not written by ab_judge.py "
                "from AgentX results",
                f"decision {verdict.get('decision')}",
                REJUDGE,
            )
        ]
    out = []
    if snap is None:
        workspace = path.resolve().parent.parent
        snap, run = snapshot(workspace)
        snap_from = (
            f"the snapshot of run {run}" if snap else "project.yaml (no run snapshotted the rules)"
        )
    else:
        snap_from = "the run's snapshot"
    # the baseline series a run judged against is the run's: without one, any (the ledger marks it, judged_against)
    run_series = (snap or {}).get("baseline_series") or ""
    # a key the snapshot lacks (a run from before it was taken, or an edited launch.json) is project.yaml's
    snap = {**project_rules(root)[0], **{k: v for k, v in (snap or {}).items() if k in RULE_KEYS}}
    if current is not None:
        changed = [k for k in RULE_KEYS if current.get(k) != snap.get(k)]
        if changed:
            out.append(
                Finding(
                    MISMATCH,
                    "project.yaml's ledger rules changed during the run: "
                    + ", ".join(f"{k} {snap.get(k)!r} -> {current.get(k)!r}" for k in changed),
                    f"{snap_from} is what this run is judged by",
                    "put project.yaml's ledger section back; only the user changes the project's rules "
                    "(a stricter bar for one A/B goes on ab_judge.py's command line)",
                )
            )
    out += rule_findings(inputs, {**snap, "baseline_series": run_series}, snap_from)
    arms = {}
    for side in ("baseline", "candidate"):
        items = inputs.get(side)
        if not isinstance(items, list) or not items or not all(isinstance(i, dict) for i in items):
            out.append(Finding(UNVERIFIABLE, f"verdict.inputs.{side} lists no files", "", REJUDGE))
            continue
        arms[side] = items
    if len(arms) < 2:
        return out
    me = rel(path, root)
    if me not in [i.get("file") for i in arms["candidate"]]:
        out.append(
            Finding(
                MISMATCH,
                f"this file ({me}) is not among the candidate inputs of its own verdict",
                f"candidates: {', '.join(str(i.get('file')) for i in arms['candidate'])}",
                REJUDGE,
            )
        )
    values = {}
    for side, items in arms.items():
        files = [str(i.get("file")) for i in items]
        if len(set(files)) < len(files):
            out.append(
                Finding(MISMATCH, f"the {side} inputs list a file twice", ", ".join(files), REJUDGE)
            )
        ys, xs = [], []
        for item in items:
            y, x, found = input_value(item, inputs, root, side)
            out += found
            ys.append(y)
            xs.append(x)
        values[side] = (ys, xs)
    every = [str(i.get("file")) for items in arms.values() for i in items]
    out += arm_findings(every, root)
    if any(f.kind == UNVERIFIABLE for f in out) or any(
        v is None for ys, _ in values.values() for v in ys
    ):
        return out
    x_metric = inputs.get("x_metric") or ""
    try:
        again = judge(
            values["baseline"][0],
            values["candidate"][0],
            inputs.get("higher_is_better") is not False,
            float(inputs.get("noise_pct")),
            float(inputs.get("accept_pct")),
            (
                values["baseline"][1],
                values["candidate"][1],
                inputs.get("x_higher_is_better") is not False,
            )
            if x_metric
            else None,
        )
    except (TypeError, ValueError) as e:
        out.append(Finding(UNVERIFIABLE, f"the verdict cannot be recomputed: {e}", "", REJUDGE))
        return out
    for key in ("decision", "vs_baseline_pct", "noise_floor_pct"):
        if not (verdict.get(key) == again[key] or same(verdict.get(key), again[key])):
            out.append(
                Finding(
                    MISMATCH,
                    f"verdict.{key} is {verdict.get(key)!r}; recomputed from the raw AgentX output it is {again[key]!r}",
                    f"baseline {values['baseline'][0]}, candidate {values['candidate'][0]} ({inputs.get('metric')}), "
                    f"noise {inputs.get('noise_pct')}%, bar {inputs.get('accept_pct')}%",
                    REJUDGE,
                )
            )
    return out


def arm_findings(files, root):
    """What is wrong with the inputs as a whole (files: both arms): two concurrencies, a result left out, a void peer
    for no valid reason."""
    out = []
    by_conc = concurrencies(files, root)
    if len(by_conc) > 1:
        out.append(
            Finding(
                MISMATCH,
                "the inputs are at different concurrencies: "
                + "; ".join(f"C{c}: {', '.join(fs)}" for c, fs in by_conc.items()),
                "both arms of an A/B are measured at one concurrency",
                REJUDGE,
            )
        )
    if missing := picked(files, root):
        out.append(
            Finding(
                MISMATCH,
                "the inputs leave out valid results at the same concurrency: " + ", ".join(missing),
                "every valid result at the A/B's concurrency in an input's task counts, whatever its series or "
                "role (a void one does not)",
                "judge again with all of them, or void the spoiled ones (failed requests, or other with a reason "
                "and evidence); screening points go under results/<subdirectory>/ or log/",
            )
        )
    for f, problems in void_peers(files, root):
        out += [
            Finding(
                k, f"{f}: {p}", "a void result is left out of the A/B only for a valid reason", fix
            )
            for k, p, fix in problems
        ]
    return out


def judged(doc):
    """Whether a result carries a verdict to recompute, whatever its role: a decision, inputs, or a void."""
    verdict = doc.get("verdict") if isinstance(doc, dict) else None
    return isinstance(verdict, dict) and bool(
        verdict.get("decision") or verdict.get("inputs") or verdict.get("status") == "void"
    )


def rule_findings(inputs, snap, snap_from):
    """The verdict's metric, directions and thresholds against the rules the run is judged by."""
    out = []
    want = {"higher_is_better": snap.get("higher_is_better") is not False}
    if snap.get("primary"):
        want["metric"] = snap["primary"]
    if snap.get("x"):
        want |= {
            "x_metric": snap["x"],
            "x_higher_is_better": snap.get("x_higher_is_better") is not False,
        }
    for key, value in want.items():
        if inputs.get(key) != value:
            out.append(
                Finding(
                    MISMATCH,
                    f"verdict.inputs.{key} is {inputs.get(key)!r}; the project judges by {value!r}",
                    f"the rules: {snap_from}",
                    REJUDGE + " (it reads the metrics from project.yaml)",
                )
            )
    if snap.get("baseline_series") and inputs.get("baseline_series") != snap["baseline_series"]:
        out.append(
            Finding(
                MISMATCH,
                f"the baseline arm is of series {inputs.get('baseline_series')!r}; the project's baseline "
                f"series is {snap['baseline_series']!r}",
                f"the rules: {snap_from}",
                "measure the base arm as a repeat of the baseline series (ledger agentx --series "
                f"{snap['baseline_series']} --role baseline), then " + REJUDGE,
            )
        )
    for key in ("noise_pct", "accept_pct"):
        have, floor = inputs.get(key), snap.get(key)
        if not ledger.number(have):
            out.append(
                Finding(MISMATCH, f"verdict.inputs.{key} is {have!r}, not a number", "", REJUDGE)
            )
        elif key == "noise_pct" and ledger.number(floor) and have != floor:
            # a wider band is not stricter: it turns a reject into noise as well as an accept
            out.append(
                Finding(
                    MISMATCH,
                    f"verdict.inputs.noise_pct is {have:g}%, not the project's {floor:g}%",
                    f"the rules: {snap_from}",
                    "the noise band is the project's; " + REJUDGE,
                )
            )
        elif key == "accept_pct" and ledger.number(floor) and have < floor:
            out.append(
                Finding(
                    MISMATCH,
                    f"verdict.inputs.{key} is {have:g}%, looser than the project's {floor:g}%",
                    f"the rules: {snap_from}",
                    f"a {key} can only be raised on the command line; " + REJUDGE,
                )
            )
    return out


def input_value(item, inputs, root, side):
    """(Y, X, findings) of one input: the values its raw AgentX output gives, after checking the file, its source and
    what the verdict recorded of it. Y None when it cannot be read."""
    out = []
    name = str(item.get("file"))
    path = resolve(name, root)
    doc, errors = ledger.load(path) if path.is_file() else (None, ["it is gone"])
    if doc is None or errors:
        return (
            None,
            None,
            [
                Finding(
                    UNVERIFIABLE,
                    f"{side} input {name}: {'; '.join(errors) or 'not a result'}",
                    "",
                    "put the result file back, or " + REJUDGE,
                )
            ],
        )
    if (doc.get("verdict") or {}).get("status") == "void":
        out.append(
            Finding(MISMATCH, f"{side} input {name} is void, so it cannot be an input", "", REJUDGE)
        )
    src = source(doc)
    if src and (item.get("agentx"), item.get("agentx_sha256")) != src:
        out.append(
            Finding(
                MISMATCH,
                f"{side} input {name}: the verdict recorded raw output {item.get('agentx')} "
                f"({str(item.get('agentx_sha256'))[:12]}…), the file says {src[0]} ({src[1][:12]}…)",
                "",
                REJUDGE,
            )
        )
    metrics, conc, problems = reconvert(doc, root)
    out += [Finding(k, f"{side} input {name}: {p}", "", fix) for k, p, fix in problems]
    if metrics is None:
        return None, None, out
    if diffs := metric_diffs(doc, metrics):
        out.append(
            Finding(
                MISMATCH,
                f"{side} input {name}: its metrics differ from its raw AgentX output",
                "; ".join(diffs),
                "convert the raw output again (ledger agentx); never edit a result's "
                "numbers, then " + REJUDGE,
            )
        )
    if conc != (doc.get("workload") or {}).get("conc"):
        out.append(
            Finding(
                MISMATCH,
                f"{side} input {name}: workload.conc is {(doc.get('workload') or {}).get('conc')}"
                f", its raw AgentX output says {conc}",
                "",
                "convert the raw output again",
            )
        )
    if (pct := failed_pct(metrics)) is not None and pct > FAILED_MAX_PCT:
        out.append(
            Finding(
                MISMATCH,
                f"{side} input {name}: {pct}% of its requests failed",
                "",
                "void it (ab_judge.py --void failed-requests) and " + REJUDGE,
            )
        )
    metric, x_metric = inputs.get("metric"), inputs.get("x_metric") or ""
    y = metrics.get(metric)
    x = metrics.get(x_metric) if x_metric else None
    if not ledger.number(y) or (x_metric and not ledger.number(x)):
        out.append(
            Finding(
                UNVERIFIABLE,
                f"{side} input {name}: its raw output has no {metric if not ledger.number(y) else x_metric}",
                "",
                REJUDGE,
            )
        )
        return None, None, out
    for key, want in (("value", y), *((("x", x),) if x_metric else ())):
        if not same(item.get(key), want):
            out.append(
                Finding(
                    MISMATCH,
                    f"{side} input {name}: the verdict read {key} {item.get(key)!r}, its raw "
                    f"AgentX output gives {want!r}",
                    "",
                    REJUDGE,
                )
            )
    return y, x, out


def verify_paths(paths, current_rules=True):
    """(all well, lines to print) for every result under paths: check --require-verdict, then verify()."""
    found, bad, lines, verdicts = 0, 0, [], 0
    for f in (f for p in paths for f in ledger.json_files(p)):
        doc, _errors = ledger.read(f)
        if doc is None and not _errors:
            continue  # some other JSON
        found += 1
        verdicts += bool(doc and judged(doc))
        root = project_root(f)
        try:
            current = project_rules(root)[0] if current_rules and root else None
            findings = verify(f, root, current=current)
        except ValueError as e:
            findings = [Finding(UNVERIFIABLE, str(e))]
        if findings:
            bad += 1
            kinds = sorted({x.kind for x in findings})
            lines.append(f"{f}: FAIL ({'; '.join(kinds)})")
            for x in findings:
                lines += x.lines()
        else:
            label = doc.get("label") if doc else ""
            decision = ((doc or {}).get("verdict") or {}).get("decision") or (doc or {}).get(
                "role", ""
            )
            lines.append(f"{f}: ok ({label}, {decision})")
    if not found:
        lines.append(f"no {ledger.SCHEMA} result under {', '.join(map(str, paths))}")
    elif not verdicts:
        lines.append(
            f"no A/B verdict under {', '.join(map(str, paths))}: judge the candidates with ab_judge.py --write "
            "(skill infera-experiment), or void them"
        )
    return found > 0 and bad == 0 and verdicts > 0, lines


# ---------------------------------------------------------------- the ledger's view: verified or not, cached


_cache = {}  # path -> (stamps of what it read, findings)


def _stamps(paths):
    out = []
    for p in paths:
        try:
            st = Path(p).stat()
            out.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(p), None, None))
    return tuple(out)


def _deps(path, doc, root):
    """The files a verification of doc reads: the file, its inputs, their raw outputs and results/ directories, the
    task's runs (the snapshot)."""
    runs = runs_dir(path.resolve().parent.parent)
    deps = [path, runs, Path(root or path.parent) / "project.yaml"]
    deps += sorted(runs.glob("*/launch.json")) if runs.is_dir() else []
    verdict = (doc.get("verdict") or {}) if isinstance(doc, dict) else {}
    void = verdict.get("void") if isinstance(verdict.get("void"), dict) else {}
    if void.get("evidence"):
        deps.append(resolve(str(void["evidence"]), root))
    inputs = verdict.get("inputs") if isinstance(verdict.get("inputs"), dict) else {}
    dirs = {path.parent}
    for side in ("baseline", "candidate"):
        for item in inputs.get(side) or []:
            if isinstance(item, dict):
                dirs.add(resolve(str(item.get("file")), root).parent)
    for d in dirs:
        # every result there (a peer may be any of them) and its raw output
        deps.append(d)
        for q in ledger.json_files(d):
            deps.append(q)
            other, _e = ledger.load(q)
            if other and (src := source(other)):
                deps.append(resolve(src[0], root))
    return deps


def check_row(path, root):
    """[findings as dicts] of a result's verdict for the ledger, [] when it holds; cached by the mtimes of the files
    it read."""
    path = Path(path)
    hit = _cache.get(path)
    if hit and _stamps([p for p, *_ in hit[0]]) == hit[0]:
        return hit[1]
    doc, _errors = ledger.load(path)
    try:
        findings = [f.as_dict() for f in verify(path, root)]
    except ValueError as e:
        findings = [Finding(UNVERIFIABLE, str(e)).as_dict()]
    _cache[path] = (_stamps(_deps(path, doc, root)), findings)
    return findings
