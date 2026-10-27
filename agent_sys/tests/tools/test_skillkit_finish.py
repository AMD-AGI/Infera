"""`skillkit` run path against the runtime's contract: the output phase is all
or nothing, composites see real kinds, the environment row is the runtime's,
and a run never starts from a stale or unresolvable record.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

AGENT_SYS = Path(__file__).resolve().parents[2]
LIB = AGENT_SYS / "tools" / "skill" / "lib"
sys.path.insert(0, str(LIB))

from skillkit import cli, hstore, pkg, runner, scaffold, wf  # noqa: E402
from skillkit import validate as vmod  # noqa: E402

from tests.tools.test_skillkit import _pair_package, _plan  # noqa: E402

A = "11111111-1111-1111-1111-111111111111"
B = "22222222-2222-2222-2222-222222222222"
SECRET = 'echo "token = \'hunter2hunter2\'" > "$AGENT_SYS_OUTPUT_B'


def _content(where: Path, name: str) -> Path:
    d = where / name
    (d / "items").mkdir(parents=True, exist_ok=True)
    (d / "README.md").write_text(f"# {name}\n\n## Purpose\nx.\n")
    (d / "items" / "content").write_text("x\n")
    return d


def _verdict(value: str) -> str:
    return (
        "python3 -c \"import json;ids=json.load(open('inputs.json'));"
        f"json.dump({{i:{value} for i in ids}},open('verdict.json','w'))\"\n"
    )


def _pair_run(tmp_path: Path, *, b_secret: bool = False) -> tuple[Path, dict]:
    dest = _pair_package(tmp_path)
    if b_secret:
        entry = dest / "assets" / "make.task" / "entry.sh"
        entry.write_text(entry.read_text().replace('echo original > "$AGENT_SYS_OUTPUT_B', SECRET))
    run = tmp_path / "run"
    runner.init(run, dest, {})
    info = runner.prepare(run, "make")
    assert runner.run_program(run, "make") == 0
    return run, info


def _run_leaves(run: Path, *leaves: str) -> None:
    for leaf in leaves:
        runner.prepare(run, leaf)
        assert runner.run_program(run, leaf) == 0
        assert all(c.passed for c in runner.finish(run, leaf))


@pytest.mark.parametrize(("second", "expect"), [("True", True), ("False", False)])
def test_composite_members_run_on_kinds_and_reduce(tmp_path, second, expect):
    dest = _pair_package(tmp_path)
    p = pkg.load(dest)
    plain = {k: v for k, v in p.validators["check_a"].items() if k not in ("body", "args")}
    plain["inputs"] = ["a"]
    for name, value in (("m1", "True"), ("m2", second)):
        (dest / name).mkdir()
        (dest / name / "readme.md").write_text("# x\n")
        (dest / name / "entry.sh").write_text(_verdict(value))
        body = {"readme": f"{name}/readme.md", "entry": f"{name}/entry.sh"}
        p.validators[name] = {**plain, "name": name, "body": body}
    p.validators["comp"] = {**plain, "name": "comp", "members": ["m1", "m2"], "reduce": "all"}
    o = vmod.run_one(p, "comp", {"a": (A, _content(tmp_path, "ca"))}, tmp_path / "w")[A]
    assert o.why_unreached == "" and o.result is expect, o


def test_agent_without_env_takes_the_producer_row(tmp_path):
    dest = _pair_package(tmp_path)
    (dest / "assets" / "v_pair.validator" / "check.py").write_text(
        "import json,os\nfrom pathlib import Path\n"
        "ids=json.loads(Path('inputs.json').read_text())\n"
        "Path('verdict.json').write_text(json.dumps({h: 'FOO' in os.environ "
        "and 'PATH' in os.environ for h in ids}))\n"
    )
    p = pkg.load(dest)
    assert not (p.agents["runner"] or {}).get("env")
    p.validators["v_pair"]["agent"] = "runner"
    targets = {"a": (A, _content(tmp_path, "ca")), "b": (B, _content(tmp_path, "cb"))}
    res = vmod.run_one(p, "v_pair", targets, tmp_path / "w", source_env={"FOO": "bar"})
    assert all(o.result is True for o in res.values()), res


def test_unresolvable_validator_agent_is_refused(tmp_path):
    dest = _pair_package(tmp_path)
    steps = dest / "steps" / "make.yaml"
    steps.write_text(
        steps.read_text().replace("  name: v_pair\n", "  name: v_pair\n  agent: ghost\n", 1)
    )
    p = pkg.load(dest)  # the documents load ...
    with pytest.raises(pkg.PackageError, match="ghost"):  # ... agent_sys's `show` does not
        runner.init(tmp_path / "run", dest, {})
    o = vmod.run_one(p, "v_pair", {"a": (A, _content(tmp_path, "ca"))}, tmp_path / "w")[A]
    assert o.result is None and "does not resolve" in o.why_unreached


def test_failed_phase_publishes_nothing_and_retry_reruns_every_validator(tmp_path):
    run, info = _pair_run(tmp_path, b_secret=True)
    first = {c.kind: c for c in runner.finish(run, "make")}
    assert not first["b"].passed and not first["a"].passed and first["a"].held
    assert hstore.latest_published(run / "handoffs", "a") is None  # nothing a consumer can stage
    assert runner.open_run(run)[0]["leaves"]["make"]["status"] == "output_invalid"
    on_b = sorted(o.validator for o in first["b"].outcomes)
    assert "check_a" in on_b  # the validator registered on a that lists b

    Path(info["outputs"]["b"]["content"], "items", "content").write_text("original\n")
    second = {c.kind: c for c in runner.finish(run, "make")}
    assert sorted(o.validator for o in second["b"].outcomes) == on_b
    assert all(c.passed for c in second.values()), [vmod.render(c) for c in second.values()]
    for k in ("a", "b"):
        slot = hstore.latest_published(run / "handoffs", k)
        assert {r["validator"] for r in hstore.verdicts(slot)} >= {"v_pair"}
    assert runner.open_run(run)[0]["leaves"]["make"]["status"] == "succeeded"


def test_unvalidated_output_is_covered_by_the_phase_fold(tmp_path):
    p = pkg.load(_pair_package(tmp_path))
    p.kinds["b"]["validators"] = []
    for v in ("check_a", "v_pair"):
        p.validators[v]["inputs"] = ["a"]
    targets = {"a": (A, _content(tmp_path, "ca")), "b": (B, _content(tmp_path, "cb"))}
    checks = vmod.check_targets(p, targets, tmp_path / "w", bound=["check_a"])
    assert vmod.phase_passed(checks), [vmod.render(c) for c in checks.values()]
    assert not checks["b"].outcomes and checks["b"].covered
    alone = vmod.check_targets(p, {"b": targets["b"]}, tmp_path / "w2", bound=[])
    assert not alone["b"].passed  # a phase that ran nothing is unchecked


def test_consumer_waits_for_the_recorded_state_then_validates_its_inputs(tmp_path, capsys):
    dest = tmp_path / "pkg"
    scaffold.scaffold(_plan(tmp_path), dest)
    steps = dest / "steps" / "review.yaml"
    steps.write_text(
        steps.read_text().replace("  validators: []\n", "  validators:\n  - check_note_has_word\n")
    )
    run = tmp_path / "run"
    runner.init(run, dest, {})
    _run_leaves(run, "publish_inputs", "write_note")
    good_note = hstore.latest_published(run / "handoffs", "note").content
    runner.prepare(run, "write_note")  # a new attempt: upstream is not done any more
    with pytest.raises(runner.NotReady, match="upstream has not succeeded"):
        runner.prepare(run, "review")
    # The workflow's gate reads the same record, at the path `status --json` prints.
    assert "`leaves.<leaf>.status`" in wf.emit(pkg.load(dest), run)
    capsys.readouterr()
    assert cli.main(["status", "--json", "--run", str(run)]) == 0
    assert json.loads(capsys.readouterr().out)["leaves"]["write_note"]["status"] == "prepared"

    runner.finish(run, "write_note")  # its program never ran: empty output
    with pytest.raises(runner.NotReady, match="write_note: output_invalid"):
        runner.prepare(run, "review")
    # Seeding over a failed producer is the skip-ahead.
    assert runner.seed(run, "note", good_note).passed
    state, p = runner.open_run(run)
    write_note = next(x for x in p.leaves() if x.name == "write_note")
    assert runner.leaf_status(state, write_note) == "seeded"
    assert "note" in runner.prepare(run, "review")["inputs"]
    # The consumer's own phase validator ran as its input phase, on record.
    rows = hstore.verdicts(hstore.latest_published(run / "handoffs", "note"))
    assert rows[-1]["validator"] == "check_note_has_word"
    assert rows[-1]["environment"]["task_label"] == "review#1"
    assert list((run / "validation").glob("review.input*/check_note_has_word"))


@pytest.mark.parametrize("stop", ["--timeout", "SIGINT", "SIGTERM", "SIGHUP"])
def test_a_stopped_run_program_kills_the_programs_group(tmp_path, stop):
    """However the wait ends, nothing entry.sh started keeps writing into the
    output slot `finish` then seals."""
    run, _info = _pair_run(tmp_path)
    tick = tmp_path / "tick"
    loop = f"while true; do echo x >> {tick}; sleep 0.2; done"
    runner.prepare(run, "make")
    zone = Path(runner.open_run(run)[0]["leaves"]["make"]["zone"])
    (zone / "package" / "assets" / "make.task" / "entry.sh").write_text(f"({loop}) &\n{loop}\n")
    env = {**os.environ, "PYTHONPATH": f"{AGENT_SYS}:{LIB}"}
    cmd = [sys.executable, str(LIB / "astk.py"), "run-program", "--run", str(run), "--leaf", "make"]
    p = subprocess.Popen(cmd + (["--timeout", "1"] if stop == "--timeout" else []), env=env)
    try:
        deadline = time.monotonic() + 20
        while not tick.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        if stop != "--timeout":
            p.send_signal(getattr(signal, stop))
        assert p.wait(timeout=10) != 0
        time.sleep(0.5)
        n1 = len(tick.read_text().splitlines())
        time.sleep(1.5)
        assert len(tick.read_text().splitlines()) == n1, "entry.sh still writing"
    finally:
        subprocess.run(["pkill", "-KILL", "-f", str(zone)], check=False)


def test_finish_refuses_after_a_failed_prepare(tmp_path, monkeypatch):
    run, _info = _pair_run(tmp_path)

    def boom(*_a, **_k):
        raise ValueError("staging failed")

    monkeypatch.setattr(runner, "stage_package", boom)
    with pytest.raises(ValueError, match="staging failed"):
        runner.prepare(run, "make")
    rec = runner.open_run(run)[0]["leaves"]["make"]
    assert rec["status"] == "prepare_failed" and "outputs" not in rec
    with pytest.raises(pkg.PackageError, match="not prepared"):
        runner.finish(run, "make")
    assert hstore.latest_published(run / "handoffs", "a") is None
    monkeypatch.undo()
    _run_leaves(run, "make")  # recovers with a new attempt
    with pytest.raises(pkg.PackageError, match="succeeded"):
        runner.finish(run, "make")  # a succeeded attempt is not judged again


def test_overlapping_prepares_finish_only_the_last_attempt(tmp_path, monkeypatch):
    """An earlier `prepare` whose staging lands after a later one's must not be
    judged (or published) under the later attempt's label."""
    run = tmp_path / "run"
    runner.init(run, _pair_package(tmp_path), {})
    stage, entered, release = runner._stage, threading.Event(), threading.Event()

    def slow_first(*args):
        if args[-1] == 0:  # attempt index 0: the first prepare
            entered.set()
            release.wait()
        return stage(*args)

    monkeypatch.setattr(runner, "_stage", slow_first)
    first = threading.Thread(target=runner.prepare, args=(run, "make"))
    first.start()
    entered.wait()
    runner.prepare(run, "make")
    release.set()
    first.join()
    runner.run_program(run, "make")
    with pytest.raises(pkg.PackageError, match="not the current attempt"):
        runner.finish(run, "make")


def test_latest_published_is_the_highest_version_per_id(tmp_path):
    """Per id the store's `latest`, even when a lower version was sealed later."""
    specs = pkg.load(_pair_package(tmp_path)).kinds
    store = tmp_path / "handoffs"
    v0, v1 = (hstore.import_content(store, "b", specs, _content(tmp_path, "b")) for _ in "01")
    assert hstore.seal(v1, specs, producer="p") is None
    assert hstore.seal(v0, specs, producer="p") is None
    os.utime(v0.vdir / "manifest.yaml", (time.time() + 60,) * 2)
    assert hstore.latest_published(store, "b").version == v1.version
