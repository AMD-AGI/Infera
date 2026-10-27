"""GSM8K, the accuracy gate (gsm8k.py): lm_eval's samples converted and counted again, the intervals, what 200 and
1319 questions can tell apart, and the two rules: an accept's candidate not significantly worse, a sealed version
not worse than the one before by more than the margin."""

import json

import pytest
from measuring import FIXTURES, gsm8k, lm_eval_run

from agent_sys_v2 import gsm8k as g
from agent_sys_v2 import ledger, ops
from agent_sys_v2.scan import load_project


def test_a_real_lm_eval_run_converts(tmp_path):
    """The real samples of the fixture (20 questions, two filters): counted under strict-match, checked against
    lm_eval's own summary, its files recorded with their digests."""
    out = tmp_path / "results"
    assert (
        ledger.main(
            [
                "gsm8k",
                str(FIXTURES / "lm_eval"),
                "--out",
                str(out),
                "--label",
                "gate",
                "--series",
                "v0",
            ]
        )
        == 0
    )
    doc = json.loads((out / "gate.gsm8k.json").read_text())
    assert (doc["schema"], doc["filter"], doc["correct"], doc["total"]) == (
        "infera.gsm8k/v1",
        "strict-match",
        19,
        20,
    )
    assert doc["other_filters"] == {"flexible-extract": {"correct": 19, "total": 20}}
    assert len(doc["wrong"]) == 1 and doc["ci95"][0] < 0.95 < doc["ci95"][1]
    assert doc["config"]["limit"] == 20.0 and doc["config"]["base_url"].endswith(
        "/v1/chat/completions"
    )
    assert (
        doc["evidence"]["samples"].endswith(".jsonl")
        and len(doc["evidence"]["samples_sha256"]) == 64
    )
    per, problem = g.recount(doc, None)
    assert problem == "" and g.tally(per) == (19, 20)
    # never overwritten; one run per label
    assert (
        ledger.main(
            [
                "gsm8k",
                str(FIXTURES / "lm_eval"),
                "--out",
                str(out),
                "--label",
                "gate",
                "--series",
                "v0",
            ]
        )
        == 1
    )
    # ledger check reads it as a GSM8K result, not a point of the curves
    assert ledger.check([out]) == (
        False,
        [f"{out}/gate.gsm8k.json: ok (GSM8K gate)", f"no infera.result/v1 result under {out}"],
    )


def test_a_summary_that_disagrees_with_its_samples_is_refused(tmp_path):
    run = lm_eval_run(tmp_path / "run", 50, {1, 2})
    res = next(run.rglob("results_*.json"))
    d = json.loads(res.read_text())
    d["results"]["gsm8k"]["exact_match,strict-match"] = 1.0
    res.write_text(json.dumps(d))
    written, errors = g.convert([run], tmp_path / "out", "x", "v0")
    assert not written and "not one run" in errors[0]
    # two runs in one directory: which one is unknown
    lm_eval_run(tmp_path / "run2", 10, set())
    out = tmp_path / "run2/__models__GLM-5.2-MXFP4"
    # another task of the family (gsm8k_cot) is not GSM8K's run
    (out / "samples_gsm8k_cot_2026-10-27T09-00-00.000000.jsonl").write_text("")
    assert g.convert([tmp_path / "run2"], tmp_path / "out", "y", "v0")[0]
    (out / "samples_gsm8k_2026-10-27T09-00-00.000000.jsonl").write_text("")
    assert "2 lm_eval runs" in g.convert([tmp_path / "run2"], tmp_path / "out", "z", "v0")[1][0]


def test_the_intervals_are_v1s():
    """Wilson and Newcombe as v1's eval_stats.py computes them, and what they can tell apart at p = 0.96."""
    lo, hi = g.wilson(192, 200)
    assert (round(lo, 4), round(hi, 4)) == (0.9231, 0.9796)
    assert g.wilson(0, 0) == (0.0, 1.0)
    # 200 against 200: about ±4.2 points; 1319 against 1319: about ±1.5
    lo, hi = g.newcombe(192, 200, 192, 200)
    assert round(-lo, 3) == round(hi, 3) == 0.042
    lo, hi = g.newcombe(0.96 * 1319, 1319, 0.96 * 1319, 1319)
    assert round(hi, 4) == 0.0152
    # so a full run passes a 2-point margin when it is as good, and fails once it is about half a point worse
    assert lo > -0.02 and g.newcombe(0.96 * 1319, 1319, 0.955 * 1319, 1319)[0] < -0.02


@pytest.fixture
def proj(tmp_path):
    """v0's full GSM8K from the baseline task, an experiment's candidate over 200 questions."""
    pdir = ops.new_project(tmp_path, "glm", "infera-perf")
    (pdir / "tasks/experiment/r1-spf.md").write_text("---\nparent: plan/r1\n---\n")
    gsm8k(pdir / "workspaces/baseline/r1", "v0-full", "v0", 1319, range(0, 1319, 25))  # 1266: 0.960
    return pdir


def gate(pdir, series="r1-spf"):
    return g.accept_gate(
        load_project(pdir), pdir, pdir / "workspaces/experiment/r1-spf", series, "v0"
    )


def test_an_accept_needs_a_candidate_not_significantly_worse(proj):
    ws = proj / "workspaces/experiment/r1-spf"
    assert gate(proj)["status"] == "missing"
    gsm8k(ws, "r1-spf-200", "r1-spf", 200, range(3, 200, 25))  # 192: 0.96
    got = gate(proj)
    assert (
        got["status"] == "ok"
        and got["version"] == "workspaces/baseline/r1/results/v0-full.gsm8k.json"
    )
    # on the same questions: v0's first 200 (8 wrong there), not its 1319
    assert got["files"][0]["version"] == [192, 200] and got["files"][0]["questions"] == 200
    # every GSM8K of the candidate's series counts: a second, much worse one is not left out
    gsm8k(ws, "r1-spf-200b", "r1-spf", 200, range(0, 200, 8))  # 175
    got = gate(proj)
    assert (
        got["status"] == "worse" and "significantly worse: 175/200 against v0's 192" in got["why"]
    )
    # its samples edited after the conversion
    (ws / "results/r1-spf-200b.gsm8k.json").unlink()
    samples = next((ws / "log/gsm8k/r1-spf-200").rglob("samples_*.jsonl"))
    samples.write_text(samples.read_text().replace('"exact_match": 0.0', '"exact_match": 1.0'))
    got = gate(proj)
    assert got["status"] == "unverifiable" and "changed after the conversion" in got["why"]


def test_sealing_needs_the_full_set_within_the_margin(proj):
    project = lambda: load_project(proj)  # noqa: E731
    cws = proj / "workspaces/close/r1"
    lines = g.seal_gate(project(), proj, cws, "v1", 0.02)
    assert lines == [(False, "results/ has v1's GSM8K over all 1319 questions (it has: none)")]
    gsm8k(cws, "v1-200", "v1", 200, set())
    assert (
        "(it has: workspaces/close/r1/results/v1-200.gsm8k.json (200))"
        in g.seal_gate(project(), proj, cws, "v1", 0.02)[0][1]
    )
    # as good as v0: the interval's lower bound is above -2 points
    gsm8k(cws, "v1-full", "v1", 1319, range(0, 1319, 25))
    lines = g.seal_gate(project(), proj, cws, "v1", 0.02)
    assert all(ok for ok, _ in lines) and "not worse than v0's by more than 0.02" in lines[-1][1]
    # 2.7 points worse: refused, with the numbers
    (cws / "results/v1-full.gsm8k.json").unlink()
    gsm8k(cws, "v1-full2", "v1", 1319, range(0, 1319, 15))
    ok, line = g.seal_gate(project(), proj, cws, "v1", 0.02)[-1]
    assert not ok and "1231 vs 1266 of 1319 questions" in line
    # a stricter margin from project.yaml
    assert g.margin_of({"quality": {"gsm8k_margin": 0.01}}) == 0.01
    assert g.margin_of({"quality": {"gsm8k_margin": "big"}}) == g.DEFAULT_MARGIN


def test_a_malformed_accuracy_result_is_named(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "x.gsm8k.json").write_text(
        json.dumps({"schema": g.SCHEMA, "label": "x", "series": "v0", "correct": 5, "total": 3})
    )
    ok, lines = ledger.check([out])
    text = "\n".join(lines)
    assert not ok and "correct is more than total" in text and "evidence.samples" in text


def test_what_counts_as_the_candidates_gsm8k(proj):
    """At least 200 questions; a run of this task (not the version's run converted again); counted under
    strict-match whatever the file says; every GSM8K in the task counts, whatever series it says, and a file that is
    no longer a GSM8K result is refused rather than left out."""
    ws = proj / "workspaces/experiment/r1-spf"
    gsm8k(ws, "r1-spf-5", "r1-spf", 5, set())
    assert gate(proj)["status"] == "missing" and "over 200+ questions" in gate(proj)["why"]
    (ws / "results/r1-spf-5.gsm8k.json").unlink()
    # the version's own lm_eval run converted as the candidate's
    own = proj / "workspaces/baseline/r1/log/gsm8k/v0-full"
    assert (
        ledger.main(
            [
                "gsm8k",
                str(own),
                "--out",
                str(ws / "results"),
                "--label",
                "copy",
                "--series",
                "r1-spf",
            ]
        )
        == 0
    )
    got = gate(proj)
    assert got["status"] == "unverifiable" and "not a run of this task" in got["why"]
    (ws / "results/copy.gsm8k.json").unlink()
    # a bad run, then its filter field edited to flexible-extract with better counts: still counted under strict-match
    bad = gsm8k(ws, "r1-spf-bad", "r1-spf", 200, range(0, 200, 6))  # 166
    d = json.loads(bad.read_text())
    bad.write_text(json.dumps({**d, "filter": "flexible-extract", "correct": 192}))
    assert (
        gate(proj)["status"] == "unverifiable"
        and "filter should be strict-match" in gate(proj)["why"]
    )
    bad.write_text(json.dumps(d))
    gsm8k(ws, "r1-spf-200", "r1-spf", 200, range(3, 200, 25))
    assert gate(proj)["status"] == "worse"
    # hidden by another series, or by a schema of its own: no
    bad.write_text(json.dumps({**d, "series": "v0"}))
    assert gate(proj)["status"] == "worse"
    bad.write_text(json.dumps({**d, "schema": "x"}))
    assert gate(proj)["status"] == "unverifiable"
    # and ledger verify reads every GSM8K file of the results it checks
    from agent_sys_v2 import verdict

    ok, lines = verdict.verify_paths([ws / "results"])
    assert not ok and any('schema should be "infera.gsm8k/v1"' in line for line in lines)
