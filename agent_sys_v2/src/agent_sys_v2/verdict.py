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
import re
import statistics
from pathlib import Path

from . import gates, gsm8k, ledger
from .model import plain, read_json, read_text, runs_dir

DEFAULT_NOISE_PCT = ledger.DEFAULT_NOISE_PCT
DEFAULT_ACCEPT_PCT = ledger.DEFAULT_ACCEPT_PCT
FAILED_MAX_PCT = 10.0  # more failed requests than this voids a measurement
FORMAL = (
    "only a formal AgentX measurement (not fast mode, DURATION at least ledger.min_duration_s) is judged; screening "
    "points stay in the ledger, marked"
)
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
    "min_duration_s",
    "reproduce_pct",
    "max_arm_gap_min",
    "min_repeats",
    "gsm8k_margin",
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
    x_noise_pct=None,
):
    """The verdict of cand against base, on the numbers alone. x: (baseline X values, candidate X values, X higher is
    better) or None. A gain inside the noise floor (the larger of noise_pct and 1.96 x the relative standard error of
    the difference) is noise, a loss beyond it a reject; an accept needs the gain past the bar and the floor, the
    medians moving the way the means do, and X not worse beyond its own floor; anything else asks for a rerun.
    noise_pct is the floor's fixed part (with the gates: the larger of the band and the version's retest noise, which
    is Y's); X's floor starts from x_noise_pct (noise_pct by default)."""
    gain, floor = compare(base, cand, higher_is_better, noise_pct)
    x_band = noise_pct if x_noise_pct is None else x_noise_pct
    x_gain, x_floor = compare(*x, x_band) if x else (0.0, 0.0)
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
    """The rules a verdict is judged by, from project.yaml (its ledger section, and quality.gsm8k_margin): RULE_KEYS."""
    cfg = {**ledger.settings_from(config), "gsm8k_margin": gsm8k.margin_of(config)}
    return {k: cfg[k] for k in RULE_KEYS}


THRESHOLDS = (
    "noise_pct",
    "accept_pct",
    "min_duration_s",
    "reproduce_pct",
    "max_arm_gap_min",
    "min_repeats",
    "gsm8k_margin",
)


def run_thresholds(workspace, config):
    """The thresholds a task's checks judge by: the run's snapshot (snapshot(): what the runner hands the check, else
    the latest launch.json), project.yaml's for a key it lacks. A task allowed to write project.yaml (close: version
    and baseline series) cannot loosen a bar for its own check."""
    snap, _run = snapshot(workspace)
    cfg = rules(config)
    return {k: (snap or {}).get(k, cfg[k]) for k in THRESHOLDS}


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


def measured(doc, root):
    """(what its run directory says of a result's measurement, problems [(kind, problem, fix)]): gates.read_point()
    on its raw AgentX output, with the runtime.env and AIPerf export the conversion recorded, which must be unchanged.
    A result converted before AIPerf's export was recorded is read with the export found below its run directory now
    (or none). ({}, problems) when the raw output cannot be read; a result with no AgentX source has its own
    workload's word."""
    src = source(doc)
    if not src:
        return dict(doc.get("workload") or {}), []
    raw = resolve(src[0], root)
    try:
        agg = json.loads(raw.read_bytes())
    except (OSError, ValueError):
        return {}, [(UNVERIFIABLE, f"its raw AgentX output {src[0]} cannot be read", KEEP_RAW)]
    ev = doc.get("evidence") or {}
    problems = []
    env = ev.get("runtime_env") if isinstance(ev.get("runtime_env"), dict) else {}
    for name, recorded, digest in (
        ("runtime.env", env.get("file"), env.get("sha256")),
        ("AIPerf export", ev.get("aiperf"), ev.get("aiperf_sha256")),
    ):
        if not (isinstance(recorded, str) and isinstance(digest, str)):
            continue
        try:
            data = resolve(recorded, root).read_bytes()
        except OSError:
            problems.append((UNVERIFIABLE, f"its {name} {recorded} is gone", KEEP_RAW))
            continue
        if hashlib.sha256(data).hexdigest() != digest:
            problems.append(
                (MISMATCH, f"its {name} {recorded} changed after the conversion", KEEP_RAW)
            )
    if problems:
        return {}, problems
    aiperf = ev.get("aiperf") if isinstance(ev.get("aiperf"), str) else None
    m, _evidence = gates.read_point(
        raw, agg if isinstance(agg, dict) else {}, aiperf and resolve(aiperf, root)
    )
    return m, []


MEASURE_KEYS = ("duration_s", "benchmark_duration_s", "fast", "window", "node")


def measure_diffs(doc, m):
    """Where a result's workload says other than its run directory of the measurement: ['key: file F, raw R'] (a key
    the file does not have is not compared: a result converted before it was recorded)."""
    held = doc.get("workload") or {}
    return [
        f"workload.{k}: file {held[k]!r}, raw {m.get(k)!r}"
        for k in MEASURE_KEYS
        if k in held and held[k] != m.get(k) and not same(held[k], m.get(k))
    ]


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


def peers(path, conc, root, min_duration_s=ledger.DEFAULT_MIN_DURATION_S):
    """The formal results at the top of the results/ directory of path whose concurrency is conc, of any series and
    role (those fields can be edited to hide one): [(path, doc)]. A screening one (fast mode, too short: read from its
    run directory) is no peer."""
    out = []
    for p in ledger.json_files(Path(path).parent):
        other, errors = ledger.load(p)
        if other is None or errors or conc_of(other, root) != conc:
            continue
        m, problems = measured(other, root)
        # screening only when its run directory and its own file both say so (and they agree): an edited file or an
        # edited run directory leaves it a peer
        own = gates.screening(ledger.measured_by(other), min_duration_s)
        if (
            not problems
            and own
            and gates.screening(m, min_duration_s)
            and not measure_diffs(other, m)
        ):
            continue
        out.append((p, other))
    return out


def arm(files, metric, x_metric, root, min_duration_s=ledger.DEFAULT_MIN_DURATION_S):
    """The inputs of one arm from its result files: [{file, agentx, agentx_sha256, value, x}], each formal (not fast
    mode, at least min_duration_s). Raises ValueError with every reason a file cannot be an input."""
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
        m, problems = measured(doc, root)
        if problems:
            why += [f"{f}: {p} ({fix})" for _k, p, fix in problems]
            continue
        if diffs := measure_diffs(doc, m):
            why.append(f"{f}: its workload differs from its run directory ({'; '.join(diffs)})")
            continue
        if reason := gates.screening(m, min_duration_s):
            why.append(f"{f}: a screening measurement ({reason}), never an input: {FORMAL}")
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
                "measured": {k: m[k] for k in ("window", "node") if k in m},
            }
        )
    if why:
        raise ValueError("\n".join(why))
    return out


def picked(files, root, min_duration_s=ledger.DEFAULT_MIN_DURATION_S):
    """The results that should be inputs and are not: every valid formal result at the inputs' concurrency in the
    results/ of an input's task, whatever its series or role (one A/B per concurrency per task). files: both arms.
    ['file (series s, C c)']."""
    have = {resolve(f, root).resolve() for f in files}
    missing = {}
    for f in files:
        path = resolve(f, root)
        doc, _errors = ledger.load(path)
        if not doc:
            continue
        conc = conc_of(doc, root)
        for p, other in peers(path, conc, root, min_duration_s):
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


def verify(path, root=None, snap=None, current=None, project=None):
    """The findings of a result's verdict against its source, [] when it holds. root: the project directory
    (project_root by default); snap: the rules the run snapshotted (the task's latest run's by default, else
    project.yaml's); current: project.yaml's rules now, compared with the snapshot (the acceptance check: the rules
    must not change during the run); project: the scanned project, when the caller has it (the gates read its
    ledger and GSM8K results)."""
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
    got = {}
    for side, items in arms.items():
        files = [str(i.get("file")) for i in items]
        if len(set(files)) < len(files):
            out.append(
                Finding(MISMATCH, f"the {side} inputs list a file twice", ", ".join(files), REJUDGE)
            )
        got[side] = []
        for item in items:
            one, found = input_value(item, inputs, root, side)
            out += found
            got[side].append(one)
    every = [str(i.get("file")) for items in arms.values() for i in items]
    min_s = inputs.get("min_duration_s")
    out += arm_findings(
        every, root, min_s if ledger.number(min_s) else ledger.DEFAULT_MIN_DURATION_S
    )
    if any(f.kind == UNVERIFIABLE for f in out) or any(
        one is None for ones in got.values() for one in ones
    ):
        return out
    x_metric = inputs.get("x_metric") or ""
    try:
        th = thresholds_of(inputs, got, snap)
        again = judge_ab(
            got["baseline"],
            got["candidate"],
            th,
            Project(root, project),
            gates.task_of(path, root),
            str(inputs.get("baseline_series") or ""),
            str(doc.get("series") or ""),
            path.resolve().parent.parent,
            (
                [i["x"] for i in got["baseline"]],
                [i["x"] for i in got["candidate"]],
                inputs.get("x_higher_is_better") is not False,
            )
            if x_metric
            else None,
        )
    except (TypeError, ValueError) as e:
        out.append(Finding(UNVERIFIABLE, f"the verdict cannot be recomputed: {e}", "", REJUDGE))
        return out
    accuracy = again["gates"]["accuracy"]
    if again["decision"] == "accept" and accuracy["status"] in ("missing", "unverifiable"):
        out.append(
            Finding(
                UNVERIFIABLE,
                f"an accept needs the candidate's GSM8K: {accuracy['why']}",
                "GSM8K is the accuracy gate of every accept (skill infera-quality-gate)",
                accuracy.get("fix") or REJUDGE,
            )
        )
    # a verdict written before the gates has no resolvable and no gates: its decision is still recomputed with them
    legacy = "gates" not in verdict and not any(k in inputs for k in GATE_KEYS)
    for key in (
        "decision",
        "vs_baseline_pct",
        "noise_floor_pct",
        *(() if legacy else ("resolvable",)),
    ):
        if not (verdict.get(key) == again[key] or same(verdict.get(key), again[key])):
            out.append(
                Finding(
                    MISMATCH,
                    f"verdict.{key} is {verdict.get(key)!r}; recomputed from the raw AgentX output it is {again[key]!r}",
                    f"baseline {[i['value'] for i in got['baseline']]}, candidate "
                    f"{[i['value'] for i in got['candidate']]} ({inputs.get('metric')}), noise floor "
                    f"{again['gates']['noise']['floor_pct']}%, bar {inputs.get('accept_pct')}%"
                    + "".join(
                        f"; {k}: {again['gates'][k].get('why')}"
                        for k in ("reproduce", "timing", "accuracy")
                        if again["gates"][k].get("status") not in ("ok", "unchecked")
                    ),
                    REJUDGE,
                )
            )
    recorded = gates.statuses(verdict.get("gates") or {})
    for k, status in () if legacy else gates.statuses(again["gates"]).items():
        if recorded.get(k) != status and not (
            k == "accuracy" and status in ("missing", "unverifiable")
        ):
            out.append(
                Finding(
                    MISMATCH,
                    f"verdict.gates.{k} is {recorded.get(k)!r}; recomputed it is {status!r}",
                    str(again["gates"][k].get("why") or ""),
                    REJUDGE,
                )
            )
    return out


def thresholds_of(inputs, got, snap):
    """The thresholds a verdict was judged with (its inputs'; a verdict written before the gates lacks theirs: the
    run's), and the A/B's concurrency. Raises ValueError."""
    th = {
        "metric": inputs.get("metric"),
        "higher_is_better": inputs.get("higher_is_better") is not False,
    }
    for key in ("noise_pct", "accept_pct", *GATE_KEYS):
        value = inputs[key] if key in inputs or key not in GATE_KEYS else snap.get(key)
        if not ledger.number(value):
            raise ValueError(f"verdict.inputs.{key} is {value!r}, not a number")
        th[key] = float(value)
    # the project's bar (the run's snapshot), whatever stricter one the command line gave: resolvable is judged by it
    th["project_accept_pct"] = float(snap["accept_pct"])
    concs = {i["conc"] for ones in got.values() for i in ones}
    th["conc"] = concs.pop() if len(concs) == 1 else None
    return th


def arm_findings(files, root, min_duration_s=ledger.DEFAULT_MIN_DURATION_S):
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
    shas = {}
    for f in files:
        doc, _errors = ledger.load(resolve(f, root))
        if doc and (src := source(doc)):
            shas.setdefault(src[1], []).append(str(f))
    if twice := [fs for fs in shas.values() if len(fs) > 1]:
        out.append(
            Finding(
                MISMATCH,
                "one AgentX run is counted more than once: "
                + "; ".join(", ".join(fs) for fs in twice),
                "each input is a measurement of its own (the same raw output converted under two labels is one)",
                REJUDGE,
            )
        )
    if missing := picked(files, root, min_duration_s):
        out.append(
            Finding(
                MISMATCH,
                "the inputs leave out valid results at the same concurrency: " + ", ".join(missing),
                "every valid formal result at the A/B's concurrency in an input's task counts, whatever its series "
                "or role (a void one and a screening one do not)",
                "judge again with all of them, or void the spoiled ones (failed requests, or other with a reason "
                "and evidence)",
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


# ---------------------------------------------------------------- the gates (gates.py), as ab_judge and verify apply them


class Project:
    """What the gates read of a project: its tasks and its ledger table (None and [] outside a project, or when it
    cannot be read: the gates that need them are unchecked)."""

    def __init__(self, root, project=None):
        self.root = root
        self.project = project
        if project is None and root is not None:
            from .model import AgentSysError
            from .scan import load_project

            try:
                self.project = load_project(root)
            except (AgentSysError, OSError, ValueError):
                self.project = None
        self.table = ledger.rows(self.project)[0] if self.project else []

    def path(self, r):
        return self.project.tasks[r["task"]].workspace / r["file"]


def formal_points(ctx, table_rows, metric, min_duration_s):
    """[{file, task, conc, value, node}] of the formal AgentX results among ledger rows, read from their raw output;
    and [why] for the ones left out because they have no AgentX source or it does not hold."""
    out, left_out = [], []
    for r in table_rows:
        path = ctx.path(r)
        doc, _errors = ledger.load(path)
        if not doc:
            continue
        if not source(doc):
            left_out.append(f"{rel(path, ctx.root)}: it has no AgentX source")
            continue
        metrics, conc, problems = reconvert(doc, ctx.root)
        m, more = measured(doc, ctx.root) if not problems else ({}, [])
        if problems or more:
            left_out.append(f"{rel(path, ctx.root)}: {(problems or more)[0][1]}")
            continue
        if gates.screening(m, min_duration_s) or not ledger.number(metrics.get(metric)):
            continue
        out.append(
            {
                "file": rel(path, ctx.root),
                "task": r["task"],
                "conc": conc,
                "value": metrics[metric],
                "node": m.get("node") or "",
            }
        )
    return out, left_out


def retest_noise(ctx, series, conc, metric, min_duration_s):
    """The retest noise of the version a series is (gates.retest), from the formal repeats on one node its sealing task
    measured: at conc, else at the concurrency it repeated most; a version whose sealing task repeated nothing falls
    back to the version before it, down to v0. None when none did."""
    m = re.fullmatch(r"v(\d+)", series or "")
    for s in [f"v{k}" for k in range(int(m[1]), -1, -1)] if m else [series]:
        sealed = [r for r in ledger.reference(ctx.table, s) if ledger.sealed_by(r, s)]
        pts, _left_out = formal_points(ctx, sealed, metric, min_duration_s)
        groups = {}
        for p in pts:
            # one task's repeats on one node at one concurrency (two sealing tasks are two stacks' worth of drift)
            groups.setdefault((p["task"], p["conc"], p["node"]), []).append(p)
        best = None
        for (_task, c, node), ps in groups.items():
            got = gates.retest([(p["value"], node) for p in ps])
            if got:
                key = (c == conc, got[1], got[0])
                if best is None or key > best[0]:
                    best = (key, c, node, ps, got)
        if best:
            _key, c, node, ps, (noise, n) = best
            return {
                "retest_pct": noise,
                "n": n,
                "from": f"{ps[0]['task']}: {n} formal measurements of {s} at C{c} on {node or 'an unnamed node'}",
                "files": [p["file"] for p in ps],
            }
    return None


def assess(base, cand, th, ctx, task, base_series):
    """The gates of an A/B (gates.py) on its inputs: base and cand are [{file, value, measured: {window, node}}] read
    from the raw output; th the thresholds (noise_pct, accept_pct, min_duration_s, reproduce_pct, max_arm_gap_min)."""
    metric = th["metric"]
    conc = th["conc"]
    g = {}
    ref_rows = gates.series_points(ctx.table, base_series, task) if base_series else []
    refs, left_out = formal_points(ctx, ref_rows, metric, th["min_duration_s"])
    refs = [p for p in refs if p["conc"] == conc]
    g["reproduce"] = gates.reproduce(
        [i["value"] for i in base], [p["value"] for p in refs], th["reproduce_pct"]
    )
    g["reproduce"]["reference"] = [p["file"] for p in refs]
    if left_out:
        g["reproduce"]["left_out"] = left_out
    items = [
        {"file": i["file"], "arm": side, **(i.get("measured") or {})}
        for side, arm_items in (("baseline", base), ("candidate", cand))
        for i in arm_items
    ]
    g["timing"] = gates.timing(items, th["max_arm_gap_min"])
    rn = retest_noise(ctx, base_series, conc, metric, th["min_duration_s"]) if base_series else None
    fixed = max(th["noise_pct"], rn["retest_pct"] if rn else 0.0)
    g["noise"] = {
        "status": gates.OK if rn else gates.UNCHECKED,
        "noise_pct": th["noise_pct"],
        "floor_pct": round(fixed, 2),
        **(
            rn
            or {
                "why": f"the sealing tasks of {base_series or 'the baseline series'} measured no formal repeats "
                "on one node"
            }
        ),
    }
    return g


def judge_ab(base, cand, th, ctx, task, base_series, cand_series, workspace, x=None):
    """The verdict of an A/B with its gates, as ab_judge writes it and verify recomputes it: judge() on the numbers
    with the noise floor's fixed part from the gates, then the gates applied (gates.decide); an accept on the numbers
    also needs the candidate's GSM8K (gsm8k.accept_gate)."""
    g = assess(base, cand, th, ctx, task, base_series)
    v = judge(
        [i["value"] for i in base],
        [i["value"] for i in cand],
        th["higher_is_better"],
        g["noise"]["floor_pct"],
        th["accept_pct"],
        x,
        x_noise_pct=th["noise_pct"],
    )
    failed = any(g[k]["status"] == gates.FAILED for k in ("reproduce", "timing"))
    if v["decision"] == "accept" and not failed:
        g["accuracy"] = (
            gsm8k.accept_gate(ctx.project, ctx.root, workspace, cand_series, base_series)
            if ctx.project
            else {"status": "missing", "why": "no project to find the GSM8K results in", "fix": ""}
        )
    else:
        g["accuracy"] = {"status": gates.UNCHECKED, "why": "only an accept needs it"}
    v = gates.decide(v, g, th.get("project_accept_pct", th["accept_pct"]))
    v["gates"] = g
    return v


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
    for key in ("noise_pct", "accept_pct", *GATE_KEYS):
        have, floor = inputs.get(key), snap.get(key)
        if key in GATE_KEYS and key not in inputs:
            continue  # a verdict from before the gates: recomputed with the run's
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
        elif ledger.number(floor) and (
            have < floor if key in STRICTER_UP else key in STRICTER_DOWN and have > floor
        ):
            unit = UNITS[key]
            out.append(
                Finding(
                    MISMATCH,
                    f"verdict.inputs.{key} is {have:g}{unit}, looser than the project's {floor:g}{unit}",
                    f"the rules: {snap_from}",
                    f"a {key} can only be made stricter on the command line; " + REJUDGE,
                )
            )
    return out


GATE_KEYS = (
    "min_duration_s",
    "reproduce_pct",
    "max_arm_gap_min",
)  # the inputs' thresholds the gates added
STRICTER_UP = ("accept_pct", "min_duration_s")  # a higher one is stricter
STRICTER_DOWN = ("reproduce_pct", "max_arm_gap_min")  # a lower one is stricter
UNITS = {"accept_pct": "%", "min_duration_s": " s", "reproduce_pct": "%", "max_arm_gap_min": " min"}


def input_value(item, inputs, root, side):
    """(the input as its raw output gives it: {file, value, x, measured, conc}, findings) of one input of a verdict,
    after checking the file, its source, its measurement and what the verdict recorded of it; None for the input when
    it cannot be read."""
    out = []
    name = str(item.get("file"))
    path = resolve(name, root)
    doc, errors = ledger.load(path) if path.is_file() else (None, ["it is gone"])
    if doc is None or errors:
        return None, [
            Finding(
                UNVERIFIABLE,
                f"{side} input {name}: {'; '.join(errors) or 'not a result'}",
                "",
                "put the result file back, or " + REJUDGE,
            )
        ]
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
        return None, out
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
    m, problems = measured(doc, root)
    out += [Finding(k, f"{side} input {name}: {p}", "", fix) for k, p, fix in problems]
    if problems:
        return None, out
    if diffs := measure_diffs(doc, m):
        out.append(
            Finding(
                MISMATCH,
                f"{side} input {name}: its workload differs from its run directory",
                "; ".join(diffs),
                "convert the raw output again (ledger agentx); never edit a result, then "
                + REJUDGE,
            )
        )
    min_s = inputs.get("min_duration_s")
    min_s = min_s if ledger.number(min_s) else ledger.DEFAULT_MIN_DURATION_S
    if reason := gates.screening(m, min_s):
        out.append(
            Finding(
                MISMATCH,
                f"{side} input {name} is a screening measurement ({reason}), never an input",
                "",
                FORMAL + "; measure it formally, then " + REJUDGE,
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
        return None, out
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
    measured_part = {k: m[k] for k in ("window", "node") if k in m}
    return {"file": name, "value": y, "x": x, "measured": measured_part, "conc": conc}, out


def accuracy_problem(path):
    """What is wrong with a GSM8K result file, '' when nothing is."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        return f"not JSON: {e}"
    if not gsm8k.is_accuracy(doc):
        return f'schema should be "{gsm8k.SCHEMA}"'
    if errors := gsm8k.validate(doc):
        return "; ".join(errors)
    return gsm8k.recount(doc, project_root(path))[1]


def verify_paths(paths, current_rules=True):
    """(all well, lines to print) for every result under paths: check --require-verdict, then verify()."""
    found, bad, lines, verdicts = 0, 0, [], 0
    scanned = {}
    for f in (f for p in paths for f in ledger.json_files(p)):
        if f.name.endswith(gsm8k.SUFFIX):
            # a GSM8K result: well formed, and its counts are its lm_eval samples' (an accept reads it)
            problem = accuracy_problem(f)
            if problem:
                bad += 1
                lines.append(f"{f}: FAIL ({UNVERIFIABLE})")
                lines += Finding(
                    UNVERIFIABLE, problem, "", "convert lm_eval's output again with ledger gsm8k"
                ).lines()
            else:
                lines.append(f"{f}: ok (GSM8K)")
            continue
        doc, _errors = ledger.read(f)
        if doc is None and not _errors:
            continue  # some other JSON
        found += 1
        verdicts += bool(doc and judged(doc))
        root = project_root(f)
        try:
            current = project_rules(root)[0] if current_rules and root else None
            if doc and judged(doc) and root not in scanned:
                scanned[root] = Project(root).project
            findings = verify(f, root, current=current, project=scanned.get(root))
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


def _deps(path, doc, root, project=None):
    """The files a verification of doc reads: the file, its inputs, their raw outputs and results/ directories, the
    task's runs (the snapshot); for the gates, every task's results/ (the reference, the retest noise, GSM8K) with
    the run directories and lm_eval samples they were converted from, and the task files (their types)."""
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
    if project is not None:
        dirs |= {t.workspace / "results" for t in project.tasks.values()}
        deps += [t.path for t in project.tasks.values()]
        # a task added later (another sealing task, its results) changes the reference: the directories that list them
        for top in (Path(project.path) / "tasks", Path(project.path) / "workspaces"):
            deps += (
                [top, *sorted(d for d in top.iterdir() if d.is_dir())] if top.is_dir() else [top]
            )
            deps += (
                [t.workspace for t in project.tasks.values()] if top.name == "workspaces" else []
            )
    for d in sorted(dirs):
        # every result there (a peer may be any of them) and what it was converted from
        deps.append(d)
        for q in ledger.json_files(d):
            deps.append(q)
            other, _e = ledger.load(q)
            ev = (other or {}).get("evidence") if isinstance(other, dict) else None
            if other and (src := source(other)):
                deps.append(resolve(src[0], root))
                deps.append(resolve(src[0], root).parent)
            if isinstance(ev, dict):
                for key in ("aiperf",):
                    if isinstance(ev.get(key), str):
                        deps.append(resolve(ev[key], root))
                if isinstance(ev.get("runtime_env"), dict) and isinstance(
                    ev["runtime_env"].get("file"), str
                ):
                    deps.append(resolve(ev["runtime_env"]["file"], root))
            if q.name.endswith(gsm8k.SUFFIX):
                try:
                    acc = json.loads(q.read_text(encoding="utf-8"))
                    deps.append(resolve(acc["evidence"]["samples"], root))
                except (OSError, ValueError, KeyError, TypeError):
                    pass
    return deps


def check_row(path, root, project=None):
    """[findings as dicts] of a result's verdict for the ledger, [] when it holds; cached by the mtimes of the files
    it read."""
    path = Path(path)
    hit = _cache.get(path)
    if hit and _stamps([p for p, *_ in hit[0]]) == hit[0]:
        return hit[1]
    doc, _errors = ledger.load(path)
    try:
        findings = [f.as_dict() for f in verify(path, root, project=project)]
    except ValueError as e:
        findings = [Finding(UNVERIFIABLE, str(e)).as_dict()]
    _cache[path] = (_stamps(_deps(path, doc, root, project)), findings)
    return findings
