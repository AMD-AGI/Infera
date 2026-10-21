"""`tools/skill/lib/skillkit` — the toolkit behind the agent_sys Claude Code skills.

Each test drives the toolkit the way a SKILL.md does, against agent_sys's own
loader and handoff code, and checks the artefact it leaves, not only its exit.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

AGENT_SYS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_SYS / "tools" / "skill" / "lib"))

from skillkit import audit, history, hstore, pkg, runner, scaffold, wf  # noqa: E402
from skillkit import validate as vmod  # noqa: E402
from skillkit.cli import main as astk  # noqa: E402

HELLO = AGENT_SYS / "examples" / "01_hello_world"
E2E = AGENT_SYS / "examples" / "llm_e2e_performance_optimization" / "e2e-flow"
E2E_VARS = {
    k: "x"
    for k in (
        "jobid",
        "node",
        "node_ip",
        "model_name",
        "model_path",
        "image",
        "container",
        "scratch_root",
        "work_root",
    )
}


# -- history ------------------------------------------------------------------ #


def _session(tmp_path: Path) -> Path:
    project = tmp_path / "projects" / "-repo"
    project.mkdir(parents=True)
    recs = [
        {"type": "ai-title", "aiTitle": "build the thing"},
        {
            "type": "user",
            "timestamp": "2026-10-01T01:00:00Z",
            "cwd": "/repo",
            "gitBranch": "dev",
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": "please build it"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-10-01T01:01:00Z",
            "message": {
                "content": [
                    {"type": "text", "text": "Building now."},
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {
                            "command": "curl -H 'Authorization: Bearer abcdefghijklmnopqrstuv' x; export API_KEY=supersecret123",
                            "description": "fetch",
                        },
                    },
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {"file_path": "/repo/a.py", "content": "x"},
                    },
                ]
            },
        },
        {
            "type": "user",
            "timestamp": "2026-10-01T01:02:00Z",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "is_error": True, "content": "boom"}
                ]
            },
        },
        {
            "type": "assistant",
            "isSidechain": True,
            "timestamp": "2026-10-01T01:03:00Z",
            "message": {"content": [{"type": "text", "text": "SIDECHAIN TEXT"}]},
        },
        {
            "type": "user",
            "isCompactSummary": True,
            "timestamp": "2026-10-01T02:00:00Z",
            "message": {"content": "Summary: it was built with sk-ant-api03-ABCDEFGHIJKLMNOP"},
        },
        {"type": "attachment", "timestamp": "2026-10-01T02:00:01Z", "attachment": {"type": "x"}},
    ]
    path = project / "11111111-2222-3333-4444-555555555555.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\nnot json\n")
    return path


def test_history_digest_keeps_the_work_and_redacts_secrets(tmp_path):
    path = _session(tmp_path)
    info = history.session_info(path)
    assert (info.title, info.human_prompts, info.compactions) == ("build the thing", 1, 1)
    text = history.render(info, history.digest(path, history.DigestOptions()))
    assert "please build it" in text and "Building now." in text
    assert "COMPACTION SUMMARY" in text and "tool error: boom" in text
    assert "Write /repo/a.py" in text and "/repo/a.py (1)" in text
    assert "SIDECHAIN TEXT" not in text
    for secret in ("abcdefghijklmnopqrstuv", "supersecret123", "ABCDEFGHIJKLMNOP"):
        assert secret not in text
    assert history.resolve_sessions(path.parent, ["1111"]) == [path]
    parts = history.split(text, 200)
    assert "".join(parts) == text and len(parts) > 1


# -- scaffold ----------------------------------------------------------------- #


def _plan(tmp_path: Path) -> dict:
    fixture = tmp_path / "fixture"
    (fixture / "items").mkdir(parents=True)
    (fixture / "README.md").write_text(
        "# card\n\n## Purpose\nWhich model.\n\n## Schema\nA JSON object.\n"
    )
    (fixture / "items" / "text.json").write_text(json.dumps({"model": "m", "tp": 1}))
    custom = tmp_path / "check_note_has_word.py"
    custom.write_text(
        "import json\nfrom pathlib import Path\n"
        "ids = json.loads(Path('inputs.json').read_text())\n"
        "m = json.loads(Path('materials.json').read_text())\n"
        "Path('verdict.json').write_text(json.dumps({h: 'model' in "
        "(Path(m[h]) / 'items' / 'content').read_text() for h in ids}))\n"
    )
    return {
        "package": "demo",
        "description": "Read a card, write a note about it.",
        "goal": "Read the model card and write a note.",
        "tasks": [
            {
                "name": "write_note",
                "kind": "program",
                "goal": "Write a note from the card.",
                "inputs": ["card"],
                "outputs": ["note"],
                "readme": "# write_note\n\nWrites a note.\n",
                "entry": 'mkdir -p "$AGENT_SYS_OUTPUT_NOTE/items"\n'
                'printf "# note\\n\\n## Purpose\\nA note.\\n" > "$AGENT_SYS_OUTPUT_NOTE/README.md"\n'
                'echo "the model is $(cat "$AGENT_SYS_INPUT_CARD/items/text.json")" '
                '> "$AGENT_SYS_OUTPUT_NOTE/items/content"\n',
            },
            {
                "name": "review",
                "kind": "ai",
                "goal": "Review the note.",
                "inputs": ["note"],
                "outputs": ["review"],
                "readme": "# review\n\nReview the note; write items/content.\n",
            },
        ],
        "kinds": [
            {
                "name": "card",
                "content_type": "structured_text",
                "description": "a card",
                "fixture": str(fixture),
                "check": {"json_files": {"items/text.json": ["model", "tp"]}},
            },
            {
                "name": "note",
                "content_type": "text",
                "description": "a note",
                "check": {"require_items": ["content"]},
                "validators": [
                    {
                        "name": "check_note_has_word",
                        "brief": "the note names the model",
                        "dimension": "trustworthiness",
                        "strength": "strong",
                        "cost": "seconds",
                        "logic_source": "external_static",
                        "check_py": str(custom),
                    }
                ],
            },
            {
                "name": "review",
                "content_type": "text",
                "description": "a review",
                "check": {"require_items": ["content"], "min_lines": {"items/content": 2}},
            },
        ],
    }


@pytest.fixture
def scaffolded(tmp_path) -> Path:
    dest = tmp_path / "pkg"
    scaffold.scaffold(_plan(tmp_path), dest)
    return dest


def test_scaffold_writes_a_package_agent_sys_accepts(scaffolded):
    rc, out = pkg.show(scaffolded)
    assert rc == 0, out
    p = pkg.load(scaffolded)
    assert not p.is_leaf("main")
    leaves = p.leaves()
    assert [x.name for x in leaves] == ["publish_inputs", "write_note", "review"]
    assert leaves[1].deps == ["publish_inputs"] and leaves[2].deps == ["write_note"]
    assert p.validators_for("write_note", "note") == ["check_note", "check_note_has_word"]
    entry = scaffolded / "assets" / "write_note.task" / "entry.sh"
    assert os.access(entry, os.X_OK)
    assert (scaffolded / "assets" / "fixture" / "card" / "items" / "text.json").is_file()


def test_scaffold_refuses_a_plan_agent_sys_would_not_load(tmp_path):
    plan = _plan(tmp_path)
    plan["tasks"][1]["inputs"] = ["nowhere"]
    plan["goal"] = "x" * 101
    with pytest.raises(scaffold.PlanError) as err:
        scaffold.scaffold(plan, tmp_path / "bad")
    assert "nowhere" in str(err.value) and "goal" in str(err.value)
    assert not (tmp_path / "bad").exists()


# -- handoffs and the generic validator --------------------------------------- #


def test_generic_validator_passes_real_and_refuses_skeleton_empty_and_secret(scaffolded, tmp_path):
    p = pkg.load(scaffolded)
    good = tmp_path / "good"
    (good / "items").mkdir(parents=True)
    (good / "README.md").write_text("# r\n\n## Purpose\nA review of the note.\n")
    (good / "items" / "content").write_text("line one\nline two\n")
    check = vmod.check_kind(p, "review", good, tmp_path / "w1")
    assert check.passed, vmod.render(check)

    skeleton = tmp_path / "skel"
    hstore.template(skeleton, "text")
    assert not vmod.check_kind(p, "review", skeleton, tmp_path / "w2").passed
    empty = tmp_path / "empty"
    empty.mkdir()
    refused = vmod.check_kind(p, "review", empty, tmp_path / "w3")
    assert not refused.passed and refused.precheck and not refused.outcomes

    (good / "items" / "content").write_text("token = 'hunter2hunter2'\nline two\n")
    leaked = vmod.check_kind(p, "review", good, tmp_path / "w4")
    assert not leaked.passed
    assert "secret" in leaked.outcomes[0].report


def test_precheck_is_the_store_seal_and_the_gate(tmp_path):
    c = tmp_path / "c"
    hstore.template(c, "reproducible")
    problems = hstore.precheck(c, "reproducible", None)
    assert any("README" in x for x in problems)
    (c / "items" / "command").chmod(0o644)
    assert any("not an executable file" in x for x in hstore.precheck(c, "reproducible", None))


# -- the runner ----------------------------------------------------------------- #


def test_runner_executes_a_package_and_publishes_validated_handoffs(scaffolded, tmp_path):
    run = tmp_path / "run"
    runner.init(run, scaffolded, {})
    for leaf in ("publish_inputs", "write_note"):
        runner.prepare(run, leaf)
        assert runner.run_program(run, leaf) == 0
        checks = runner.finish(run, leaf)
        assert all(c.passed for c in checks), [vmod.render(c) for c in checks]

    with pytest.raises(runner.NotReady, match="not ready"):
        _unready(scaffolded, tmp_path)

    info = runner.prepare(
        run, "review"
    )  # the AI leaf: brief + env, the work is "done by hand" here
    brief = Path(info["brief"]).read_text()
    assert "## What this task must deliver" in brief and "Review the note" in brief
    out = Path(info["outputs"]["review"]["content"])
    (out / "items").mkdir()
    (out / "README.md").write_text("# review\n\n## Purpose\nChecks the note.\n")
    (out / "items" / "content").write_text("only one line\n")
    first = runner.finish(run, "review")
    assert not first[0].passed  # min_lines floor of 2
    (out / "items" / "content").write_text("line one\nline two\n")
    second = runner.finish(run, "review")
    assert second[0].passed

    slot = hstore.latest_published(run / "handoffs", "review")
    manifest = (slot.vdir / "manifest.yaml").read_text()
    assert hstore.digest(slot.content) in manifest
    rows = hstore.verdicts(slot)
    assert [r["result"] for r in rows] == [
        True
    ]  # the seal creates validation.yaml; the refusal is in state
    state, _ = runner.open_run(run)
    assert [bool(x["failed"]) for x in state["leaves"]["review"]["finish_log"]] == [True, False]
    # agent_sys's own store reads what the runner published: digest verified, verdicts parse.
    from handoff import verdict as verdict_mod  # noqa: PLC0415
    from handoff.store import FilesystemStore  # noqa: PLC0415

    store = FilesystemStore(run / "handoffs")
    content = store.copy_out(slot.hid, slot.version, tmp_path / "copied")
    assert sorted(content.items) == ["content"]
    assert [v.result for v in verdict_mod.read(slot.vdir / "validation.yaml")] == [True]
    report = runner.closure_report(run)
    assert report["complete"], report
    script = wf.emit(pkg.load(scaffolded), run)
    assert script.startswith("export const meta") and '"deps": [\n      "write_note"' in script
    assert astk(["workflow", "--run", str(run), "--out", str(tmp_path / "wf.js")]) == 0
    assert (tmp_path / "wf.js").read_text().count('"done": true') == 3

    # A second run that seeds `note` from the first: the producer counts as done.
    run2 = tmp_path / "run2"
    runner.init(run2, scaffolded, {})
    note = hstore.latest_published(run / "handoffs", "note")
    assert runner.seed(run2, "note", note.content).passed
    state, p2 = runner.open_run(run2)
    status = {x.name: runner.leaf_status(state, x, run2 / "handoffs") for x in p2.leaves()}
    assert status["write_note"] == "seeded" and status["review"] == "pending"


def _unready(scaffolded: Path, tmp_path: Path) -> None:
    fresh = tmp_path / "fresh"
    runner.init(fresh, scaffolded, {})
    runner.prepare(fresh, "write_note")


def test_hello_world_program_leaf_through_the_cli(tmp_path, capsys):
    run = tmp_path / "run"
    assert astk(["run-init", "--run", str(run), "--package", str(HELLO)]) == 0
    assert astk(["prepare", "--run", str(run), "--leaf", "env_facts"]) == 0
    assert astk(["run-program", "--run", str(run), "--leaf", "env_facts"]) == 0
    assert astk(["finish", "--run", str(run), "--leaf", "env_facts"]) == 0
    assert "[PASS] check_env_facts" in capsys.readouterr().out
    assert astk(["closure", "--run", str(run)]) == 1  # env_check (AI) has not run


def test_e2e_flow_graph_and_workflow(tmp_path):
    p = pkg.load(E2E, E2E_VARS)
    leaves = {x.name: x for x in p.leaves()}
    assert "rank" in leaves["identify"].deps
    assert "build_workset" in leaves["optimize_kernel"].deps
    assert not any(x.external_inputs for x in leaves.values())
    script = wf.emit(p, tmp_path)
    for name in leaves:
        assert f'"name": "{name}"' in script


# -- audit ---------------------------------------------------------------------- #


def test_audit_blocks_secrets_and_asks_about_big_files(tmp_path):
    d = tmp_path / "d"
    (d / "logs").mkdir(parents=True)
    (d / "ok.sh").write_text('export HF_TOKEN="$HF_TOKEN"\n')
    (d / "bad.txt").write_text("password: correct-horse-battery\n")
    (d / "logs" / "run.log").write_text("x\n")
    big = d / "big.bin"
    with big.open("wb") as fh:
        fh.truncate(5 << 20)
    a = audit.audit(d)
    assert a.secrets == ["bad.txt:1"]
    assert any("big.bin" in x for x in a.ask_user) and any("log" in x for x in a.ask_user)
    assert stat.S_ISREG(big.stat().st_mode)


# -- fixes from dogfooding the skills ------------------------------------------- #


def test_project_dir_maps_every_non_alphanumeric(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "projects" / "-x-agent-sys").mkdir(parents=True)
    assert history.project_dir_for("/x/agent_sys").name == "-x-agent-sys"
    with pytest.raises(FileNotFoundError):
        history.project_dir_for("/nowhere/at_all")


def test_generic_check_quoted_placeholder_regex_floor_and_dir_lines(tmp_path):
    sys.path.insert(0, str(AGENT_SYS / "tools" / "skill" / "lib" / "skillkit" / "templates"))
    import handoff_check  # noqa: PLC0415

    root = tmp_path / "c"
    (root / "items" / "readers").mkdir(parents=True)
    (root / "README.md").write_text(
        "# r\n\n## Purpose\nThe seal refuses `[More Information Needed]`.\n"
    )
    (root / "items" / "readers" / "a.md").write_text("see handoff/store.py:12\n")
    problems, _ = handoff_check.check(root, {"forbid_placeholders": True})
    assert problems == []
    problems, _ = handoff_check.check(
        root,
        {
            "min_matches": {"items/**": {"pattern": r"\w+\.py:\d+", "n": 2, "why": "cite"}},
            "min_lines": {"items/readers": 3},
        },
    )
    assert any("matches" in p for p in problems) and any("non-blank lines" in p for p in problems)


def test_seal_is_idempotent_and_unset_vars_are_reported(scaffolded, tmp_path, capsys):
    good = tmp_path / "good"
    (good / "items").mkdir(parents=True)
    (good / "README.md").write_text("# r\n\n## Purpose\nA review.\n")
    (good / "items" / "content").write_text("one\ntwo\n")
    store = tmp_path / "store"
    argv = ["seal", "--package", str(scaffolded), "--kind", "review", "--content", str(good)]
    argv += ["--store", str(store), "--work", str(tmp_path / "zones")]
    assert astk(argv) == 0 and astk(argv) == 0
    assert "already published" in capsys.readouterr().out
    assert len(hstore.slots(store, "review")) == 1

    plan = _plan(tmp_path / "p2")
    plan["tasks"][1]["env"] = {"TOPIC": "${topic}"}
    (tmp_path / "p2").mkdir(exist_ok=True)
    scaffold.scaffold(plan, tmp_path / "pkg2")
    with pytest.raises(pkg.PackageError, match="topic"):
        pkg.load(tmp_path / "pkg2")
    p = pkg.load(tmp_path / "pkg2", allow_unset=True)
    assert vmod.check_kind(p, "review", good, tmp_path / "w").passed


# -- fixes from the review --------------------------------------------------------- #


def _pair_package(tmp_path: Path) -> Path:
    """One leaf writing two kinds; `v_pair` judges both at once and mutates its input."""
    check = tmp_path / "v_pair.py"
    check.write_text(
        "import json\nfrom pathlib import Path\n"
        "ids = json.loads(Path('inputs.json').read_text())\n"
        "m = json.loads(Path('materials.json').read_text())\n"
        "for h in ids:\n    (Path(m[h]) / 'items' / 'content').write_text('tampered')\n"
        "Path('verdict.json').write_text(json.dumps({h: len(ids) == 2 for h in ids}))\n"
    )
    entry = "".join(
        f'mkdir -p "$AGENT_SYS_OUTPUT_{k.upper()}/items"\n'
        f'printf "# {k}\\n\\n## Purpose\\nx.\\n" > "$AGENT_SYS_OUTPUT_{k.upper()}/README.md"\n'
        f'echo original > "$AGENT_SYS_OUTPUT_{k.upper()}/items/content"\n'
        for k in ("a", "b")
    )
    plan = {
        "package": "pair",
        "goal": "Write a pair.",
        "description": "pair",
        "tasks": [
            {
                "name": "make",
                "kind": "program",
                "goal": "Write a and b.",
                "inputs": [],
                "outputs": ["a", "b"],
                "readme": "# make\n",
                "entry": entry,
            }
        ],
        "kinds": [
            {
                "name": "a",
                "content_type": "text",
                "description": "a",
                "validators": [
                    {
                        "name": "v_pair",
                        "brief": "both arms",
                        "dimension": "completeness",
                        "strength": "strong",
                        "cost": "seconds",
                        "check_py": str(check),
                    }
                ],
            },
            {"name": "b", "content_type": "text", "description": "b"},
        ],
    }
    dest = tmp_path / "pair"
    scaffold.scaffold(plan, dest)
    # v_pair also lists `b`, as check_measurement_order lists both arms.
    steps = dest / "steps" / "make.yaml"
    steps.write_text(
        steps.read_text().replace("inputs:\n  - a\n  tags", "inputs:\n  - a\n  - b\n  tags")
    )
    return dest


def test_multi_input_validator_sees_all_outputs_and_cannot_mutate_them(tmp_path):
    dest = _pair_package(tmp_path)
    assert pkg.load(dest).validators["v_pair"]["inputs"] == ["a", "b"]
    run = tmp_path / "run"
    runner.init(run, dest, {})
    runner.prepare(run, "make")
    assert runner.run_program(run, "make") == 0
    checks = runner.finish(run, "make")
    assert all(c.passed for c in checks), [vmod.render(c) for c in checks]
    for k in ("a", "b"):
        slot = hstore.latest_published(run / "handoffs", k)
        assert (slot.content / "items" / "content").read_text() == "original\n"


def test_scaffold_reproducible_kind_and_force_rules(tmp_path):
    plan = _plan(tmp_path)
    plan["kinds"][2]["content_type"] = "reproducible"
    plan["kinds"][2]["check"] = {"require_items": ["result", "env", "command"]}
    dest = tmp_path / "pkg"
    scaffold.scaffold(plan, dest)
    rc, out = pkg.show(dest)
    assert rc == 0, out
    scaffold.scaffold(plan, dest, force=True)  # a fixture already in place is fine
    plan["tasks"][1]["name"] = "renamed"
    with pytest.raises(scaffold.PlanError, match="stale generated file"):
        scaffold.scaffold(plan, dest, force=True)
    assert not (dest / "steps" / "renamed.yaml").exists()  # refused before writing
    plan["kinds"].append({"name": "orphan", "content_type": "text", "description": "x"})
    with pytest.raises(scaffold.PlanError, match="neither produced nor consumed"):
        scaffold.scaffold(plan, tmp_path / "other")


def test_secret_patterns_are_one_list_and_catch_realistic_secrets():
    from skillkit import secretscan  # noqa: PLC0415

    sys.path.insert(0, str(AGENT_SYS / "tools" / "skill" / "lib" / "skillkit" / "templates"))
    import handoff_check  # noqa: PLC0415

    assert handoff_check.PATTERNS == secretscan.PATTERNS
    for text in (
        "--api-key abcdef123456",
        "export OPENAI_API_KEY=abc123def456ghi",
        "https://bob:hunter22@example.com/x",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        '"Ocp-Apim-Subscription-Key": "abcd1234efgh"',
        "token=$HF_TOKEN password: correct-horse-battery",
    ):
        assert secretscan.looks_secret(text), text
    for text in (
        "max_tokens: 2000",
        "total_tokens=1234567890",
        "token=$HF_TOKEN",
        'k=os.environ["K"]',
    ):
        assert not secretscan.looks_secret(text), text
    assert secretscan.redact("token=$HF_TOKEN") == "token=$HF_TOKEN"


def test_digest_skips_messages_no_person_typed(tmp_path):
    path = _session(tmp_path)
    extra = [
        {
            "type": "user",
            "timestamp": "2026-10-01T03:00:00Z",
            "message": {"content": "[Request interrupted by user]"},
        },
        {
            "type": "user",
            "timestamp": "2026-10-01T03:01:00Z",
            "message": {"content": "<task-notification>done</task-notification>"},
        },
    ]
    with path.open("a") as fh:
        fh.write("\n".join(json.dumps(r) for r in extra) + "\n")
    assert history.session_info(path).human_prompts == 1
    opts = history.DigestOptions(until="2026-10-01")
    assert "please build it" in history.render(
        history.session_info(path), history.digest(path, opts)
    )


def test_emitted_workflow_parses_as_javascript(scaffolded, tmp_path):
    import shutil as sh  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    node = sh.which("node")
    if node is None:
        pytest.skip("node is not installed")
    run = tmp_path / "it's a run"
    runner.init(run, scaffolded, {})
    script = wf.emit(pkg.load(scaffolded), run)
    body = script.replace("export const meta", "const meta", 1)
    src = tmp_path / "wf.mjs"
    src.write_text(f"async function main() {{\n{body}\n}}\n")
    proc = subprocess.run([node, "--check", str(src)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_skillkit_uses_the_runtime_functions_not_copies():
    """The pieces that must agree with agent_sys are agent_sys's own objects."""
    import agent.gate  # noqa: PLC0415
    import agent.runner  # noqa: PLC0415
    import env_mgr.grants  # noqa: PLC0415
    import handoff.store  # noqa: PLC0415
    import validator.phase  # noqa: PLC0415

    assert pkg.env_name is env_mgr.grants.env_name
    assert (
        hstore.runnable is agent.gate.runnable
        and hstore.FilesystemStore is handoff.store.FilesystemStore
    )
    assert runner.outputs_brief is agent.runner.outputs_brief
    assert vmod.LeafValidator is validator.phase.LeafValidator
    assert vmod.declare_materials is validator.phase.declare_materials
