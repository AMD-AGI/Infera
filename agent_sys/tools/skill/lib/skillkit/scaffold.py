"""Turn a small task plan into a loadable agent_sys task package.

The plan is what the task-from-history skill writes after reading the history;
everything agent_sys requires that the plan need not repeat is derived here:

* `main` is always a NON-LEAF wrapping the leaves (a leaf `main` crashes `show`);
* each kind gets at least one validator, and each validator names the kind;
* grants: `read` for every input, `write` for every output; `main` writes all;
* subgraph `froms` = the derived handoff edges + the plan's ordering `froms`;
* an input kind no task produces is published by a `publish_inputs` program
  leaf that copies a fixture (agent_sys's documented route for an external
  handoff — `examples/03_kernelforge_one_kernel`), and is validated on
  publication exactly as if a task had produced it.

Plan (YAML):

    package: my_task                # name used in descriptions
    description: one line
    goal: <= 100 chars              # main's goal
    tasks:                          # listing order = execution order
      - name: build_thing
        kind: ai | program
        goal: <= 100 chars
        description: one line
        inputs: [kind, ...]
        outputs: [kind, ...]
        froms: [task, ...]          # optional ordering-only edges
        readme: |                   # ai: the prompt (system prompt) | program: what it does
        entry: |                    # program only: POSIX sh body (set -eu is added)
        env: {VAR: value}           # optional, reaches this task's body only
        resources: {gpu: 1}         # optional
    kinds:
      - name: thing_result
        content_type: reproducible | code | structured_text | text
        description: one line
        items: [extra required item names]
        check: {handoff_check args}   # see templates/handoff_check.py
        calibration: |                # why each bar is where it is (measured on the real result);
                                      # rendered into the validator readme, survives --force
        fixture: /abs/content/dir     # only for an input no task produces
        validators:                   # optional hand-written validators, in addition
          - name: check_thing_reruns
            brief: one line
            dimension: completeness | usability | trustworthiness
            strength: strong | long_term_strong | weak
            cost: seconds | minutes | gpu_hours
            logic_source: external_static | external_dynamic
            args: {...}
            check_py: /abs/path/check.py   # copied to assets/<name>.validator/check.py
            calibration: |                 # why the bar is where it is
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

from skillkit import hstore
from skillkit import pkg as pkg_mod

TEMPLATES = Path(__file__).resolve().parent / "templates"
GOAL_MAX = 100
CONTENT_TYPES = ("reproducible", "code", "structured_text", "text")
AI_BACKEND = [
    {
        "key": "claude_code_sdk",
        "backend_entry": "agent.backends.claude_sdk:ClaudeSdkBackend",
        "err": 'the claude backend needs `pip install -e "agent_sys[claude]"`',
    }
]


class PlanError(Exception):
    pass


def validator_name(kind: str) -> str:
    return "check_" + "".join(c if c.isalnum() or c == "_" else "_" for c in kind)


def _check_plan(plan: dict) -> list[str]:
    errs = []
    tasks = plan.get("tasks") or []
    kinds = {k["name"]: k for k in plan.get("kinds") or []}
    if not tasks:
        errs.append("plan has no tasks")
    names = [t.get("name") for t in tasks]
    if len(set(names)) != len(names):
        errs.append(f"task names repeat: {names}")
    if "main" in names or "publish_inputs" in names:
        errs.append("'main' and 'publish_inputs' are generated; do not name a task that")
    if len(str(plan.get("goal", ""))) > GOAL_MAX or not plan.get("goal"):
        errs.append(f"plan goal must be 1..{GOAL_MAX} chars")
    produced: set[str] = set()
    for t in tasks:
        if t.get("kind") not in ("ai", "program"):
            errs.append(f"task {t.get('name')}: kind must be ai or program")
        if not t.get("goal") or len(str(t["goal"])) > GOAL_MAX:
            errs.append(f"task {t.get('name')}: goal must be 1..{GOAL_MAX} chars")
        if t.get("kind") == "program" and not t.get("entry"):
            errs.append(f"task {t.get('name')}: a program task needs `entry`")
        if not t.get("readme"):
            errs.append(f"task {t.get('name')}: needs a readme (for an AI task it IS the prompt)")
        if not t.get("outputs"):
            errs.append(
                f"task {t.get('name')}: declares no outputs — a task with no handoff cannot be checked"
            )
        for k in list(t.get("inputs") or []) + list(t.get("outputs") or []):
            if k not in kinds:
                errs.append(f"task {t.get('name')}: kind {k!r} is not declared under kinds")
        for k in t.get("inputs") or []:
            if (
                k in kinds
                and k not in produced
                and any(k in (u.get("outputs") or []) for u in tasks)
            ):
                errs.append(
                    f"task {t.get('name')}: input {k!r} is produced only by a LATER task — reorder"
                )
        for f in t.get("froms") or []:
            if f not in names[: names.index(t.get("name"))]:
                errs.append(f"task {t.get('name')}: froms {f!r} must name an EARLIER task")
        produced |= set(t.get("outputs") or [])
    for name, k in kinds.items():
        for v in k.get("validators") or []:
            for key in ("name", "brief", "dimension", "strength", "check_py"):
                if not v.get(key):
                    errs.append(f"kind {name}: custom validator needs `{key}`")
            if v.get("check_py") and not Path(v["check_py"]).is_file():
                errs.append(f"kind {name}: check_py {v['check_py']} does not exist")
        if k.get("content_type") not in CONTENT_TYPES:
            errs.append(f"kind {name}: content_type must be one of {CONTENT_TYPES}")
        consumed = any(name in (t.get("inputs") or []) for t in tasks)
        if name not in produced and not consumed:
            errs.append(f"kind {name}: declared but neither produced nor consumed by any task")
        if name not in produced and consumed and not k.get("fixture"):
            errs.append(
                f"kind {name}: consumed but produced by no task — give it a `fixture` content dir"
            )
        if k.get("fixture") and not (Path(k["fixture"]) / "README.md").is_file():
            errs.append(
                f"kind {name}: fixture {k['fixture']} is not a handoff content dir (no README.md)"
            )
    return errs


def scaffold(plan: dict, dest: Path, *, force: bool = False) -> list[Path]:
    """Write the package at `dest`. Refuses a non-empty `dest` unless `force`
    (and even then only overwrites files it generates — it never deletes)."""
    errs = _check_plan(plan)
    if errs:
        raise PlanError("plan is not valid:\n  " + "\n  ".join(errs))
    if dest.exists() and any(dest.iterdir()) and not force:
        raise PlanError(
            f"{dest} exists and is not empty (pass --force to overwrite generated files)"
        )
    # Everything is computed first and written last, so a refusal leaves the
    # destination exactly as it was.
    files: dict[str, tuple[str, int | None]] = {}
    trees: dict[str, Path] = {}

    def put(rel: str, text: str, mode: int | None = None) -> None:
        files[rel] = (text, mode)

    tasks = list(plan["tasks"])
    kinds = {k["name"]: k for k in plan["kinds"]}
    produced = {k for t in tasks for k in t.get("outputs") or []}
    external = [k for k in kinds if k not in produced]

    # -- shared.yaml: the agents ---------------------------------------- #
    agents = [
        {
            "module": "agent",
            "name": "runner",
            "kind": "program",
            "version": "1",
            "description": "Runs a program task's entry.sh. No model call.",
        },
        {
            "module": "agent",
            "name": "worker",
            "kind": "ai",
            "version": "1",
            "description": "A Claude agent; the task readme is its brief.",
            "backends": AI_BACKEND,
        },
    ]
    for t in tasks:
        if t.get("env"):
            a = dict(agents[0] if t["kind"] == "program" else agents[1])
            a["name"] = f"{t['name']}_agent"
            a["env"] = {str(k): str(v) for k, v in t["env"].items()}
            agents.append(a)
    put(
        "shared.yaml",
        _yaml_doc(
            "Agents. Order inside a file matters (define before use); across files it does not.",
            agents,
        ),
    )

    # -- one file per task: its output kinds' validators, the kinds, the task #
    producer: dict[str, str] = {}
    entries = []
    if external:
        docs = _kind_docs(external, kinds)
        docs.append(
            _task_doc(
                "publish_inputs",
                "runner",
                "Publish the external input handoffs from fixtures.",
                "Publish fixture handoffs.",
                [],
                external,
            )
        )
        put(
            "steps/publish_inputs.yaml",
            _yaml_doc("External inputs, validated on publication.", docs),
        )
        body = [
            "#!/bin/sh",
            "# Publishes handoffs produced outside this package. Validated on publication,",
            "# exactly as if a task here had produced them.",
            "set -eu",
        ]
        for k in external:  # _check_plan guarantees each has a fixture
            trees[f"assets/fixture/{k}"] = Path(kinds[k]["fixture"])
            body.append(
                f'cp -a "$AGENT_SYS_TASK_PACKAGE/assets/fixture/{k}/." "$AGENT_SYS_OUTPUT_{pkg_mod.env_name(k)}/"'
            )
        put("assets/publish_inputs.task/entry.sh", "\n".join(body) + "\n", 0o755)
        put(
            "assets/publish_inputs.task/readme.md",
            "# publish_inputs\n\nCopies each external input's content into its output slot. "
            "The kinds' validators then judge it like any producer's output.\n",
        )
        for k in external:
            producer[k] = "publish_inputs"
        entries.append({"closure": "publish_inputs", "froms": []})

    for t in tasks:
        name = t["name"]
        agent = (
            f"{name}_agent" if t.get("env") else ("runner" if t["kind"] == "program" else "worker")
        )
        docs = _kind_docs(t.get("outputs") or [], kinds)
        docs.append(
            _task_doc(
                name,
                agent,
                t.get("description") or t["goal"],
                t["goal"],
                t.get("inputs") or [],
                t.get("outputs") or [],
                t.get("resources"),
            )
        )
        put(
            f"steps/{name}.yaml", _yaml_doc(f"Task {name}: validators, kinds, then the task.", docs)
        )
        put(f"assets/{name}.task/readme.md", str(t["readme"]).rstrip() + "\n")
        if t["kind"] == "program":
            put(
                f"assets/{name}.task/entry.sh",
                "#!/bin/sh\n# Runs as `/bin/sh entry.sh`, cwd = the task zone. POSIX sh only.\nset -eu\n"
                + str(t["entry"]).rstrip()
                + "\n",
                0o755,
            )
        froms = sorted(
            {producer[k] for k in t.get("inputs") or [] if k in producer}
            | set(t.get("froms") or []),
            key=lambda n: [e["closure"] for e in entries].index(n),
        )
        entries.append({"closure": name, "froms": froms})
        for k in t.get("outputs") or []:
            producer[k] = name
    entries[-1]["is_end"] = True

    # -- validators' bodies --------------------------------------------- #
    put("assets/lib/handoff_check.py", (TEMPLATES / "handoff_check.py").read_text(), 0o755)
    for k, spec in kinds.items():
        v = validator_name(k)
        args = _check_args(spec)
        put(
            f"assets/{v}.validator/entry.sh",
            "#!/bin/sh\n# Validator body: cwd is a fresh zone with inputs.json, args.json, materials.json.\nset -eu\n"
            'PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?neither package variable is set}}"\n'
            'exec "${AGENT_SYS_DEMO_PYTHON:-python3}" "$PKG/assets/lib/handoff_check.py"\n',
            0o755,
        )
        put(f"assets/{v}.validator/readme.md", _validator_readme(k, spec, args))
        for cv in spec.get("validators") or []:
            put(f"assets/{cv['name']}.validator/check.py", Path(cv["check_py"]).read_text(), 0o755)
            put(
                f"assets/{cv['name']}.validator/entry.sh",
                "#!/bin/sh\nset -eu\n"
                'PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?neither package variable is set}}"\n'
                f'exec "${{AGENT_SYS_DEMO_PYTHON:-python3}}" "$PKG/assets/{cv["name"]}.validator/check.py"\n',
                0o755,
            )
            put(
                f"assets/{cv['name']}.validator/readme.md",
                f"# {cv['name']}\n\n{cv['brief']}\n\nHand-written; see check.py. Judges `{k}`.\n"
                + _calibration(cv),
            )

    # -- main ------------------------------------------------------------ #
    last_outputs = list(tasks[-1].get("outputs") or [])
    main = {
        "module": "task",
        "name": "main",
        "description": str(plan.get("description") or plan["goal"]),
        "handoffs": list(kinds),
        "validators": [],
        "task": {
            "goal": str(plan["goal"]),
            "version": "1",
            "inputs": [],
            "outputs": last_outputs,
            "permissions": {
                "grants": [{"path": f"handoffs/{k}", "access": "write", "kind": k} for k in kinds]
            },
            "subgraph": entries,
        },
    }
    put("main.yaml", _yaml_single("The run's entry: a non-leaf whose work is its subgraph.", main))
    put("assets/main.task/readme.md", _main_readme(plan, tasks, external))
    put(
        "plan.task.yaml.txt",
        "# The plan this package was generated from (astk scaffold). Not loaded.\n"
        + yaml.safe_dump(plan, sort_keys=False, allow_unicode=True),
    )
    put(GENERATED, "".join(f"{r}\n" for r in sorted([*files, *trees, GENERATED])))
    return _write(dest, files, trees)


#: The list of what `scaffold` wrote, so `--force` can tell a stale file from a new one.
GENERATED = ".astk-generated"


def _write(
    dest: Path, files: dict[str, tuple[str, int | None]], trees: dict[str, Path]
) -> list[Path]:
    """Refuse before writing: stale generated files (a task or kind was renamed or
    removed — scaffold never deletes, so the user must), and a fixture whose
    destination already holds different content."""
    problems = []
    previous = dest / GENERATED
    if previous.is_file():
        stale = [
            r
            for r in previous.read_text().split()
            if r not in files and r not in trees and (dest / r).exists()
        ]
        problems += [
            f"stale generated file from the last scaffold (remove it yourself): {dest / r}"
            for r in stale
        ]
    for rel, src in trees.items():
        dst = dest / rel
        if dst.exists() and hstore.digest(dst) != hstore.digest(src):
            problems.append(f"{dst} exists with different content than fixture {src}")
    if problems:
        raise PlanError("refusing to write anything:\n  " + "\n  ".join(problems))
    written: list[Path] = []
    for rel, (text, mode) in files.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        if mode:
            os.chmod(p, mode)
        written.append(p)
    for rel, src in trees.items():
        dst = dest / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, symlinks=True)
        written.append(dst)
    return written


def _check_args(spec: dict) -> dict:
    args = {"forbid_placeholders": True, "forbid_secrets": True}
    args.update(spec.get("check") or {})
    return args


def _kind_docs(names: list[str], kinds: dict) -> list[dict]:
    docs: list[dict] = []
    for k in names:
        spec = kinds[k]
        docs.append(
            {
                "module": "validator",
                "name": validator_name(k),
                "version": "1",
                "brief": f"{k} is complete, structured and free of placeholders and secret values",
                "dimension": "completeness",
                "strength": "strong",
                "inputs": [k],
                "tags": {"logic_source": "external_static", "cost": "seconds"},
                "args": _check_args(spec),
            }
        )
        for cv in spec.get("validators") or []:
            vdoc = {
                "module": "validator",
                "name": cv["name"],
                "version": "1",
                "brief": cv["brief"],
                "dimension": cv["dimension"],
                "strength": cv["strength"],
                "inputs": [k],
                "tags": {
                    "logic_source": cv.get("logic_source", "external_dynamic"),
                    "cost": cv.get("cost", "minutes"),
                },
            }
            if cv.get("args"):
                vdoc["args"] = dict(cv["args"])
            docs.append(vdoc)
        kind_doc = {
            "module": "handoff",
            "name": k,
            "version": "1",
            "description": str(spec.get("description") or k),
            "content_type": spec["content_type"],
            "scope": "fixed.required",
            "validators": [validator_name(k)] + [cv["name"] for cv in spec.get("validators") or []],
        }
        extra = list(spec.get("items") or [])
        if spec["content_type"] == "reproducible":
            # agent_sys admits a reproducible kind only if `env` is required
            # (`handoff/kind.py`): a result without its environment is not one.
            extra = ["env", *[i for i in extra if i != "env"]]
        if extra:
            kind_doc["items_schema"] = {
                "type": "object",
                "properties": {i: {"type": "string"} for i in extra},
                "required": extra,
            }
        docs.append(kind_doc)
    return docs


def _task_doc(name, agent, description, goal, inputs, outputs, resources=None) -> dict:
    grants = [{"path": f"handoffs/{k}", "access": "read", "kind": k} for k in inputs]
    grants += [{"path": f"handoffs/{k}", "access": "write", "kind": k} for k in outputs]
    task = {
        "goal": str(goal),
        "version": "1",
        "inputs": list(inputs),
        "outputs": list(outputs),
        "permissions": {"grants": grants},
    }
    if resources:
        task["resources"] = dict(resources)
    return {
        "module": "task",
        "name": name,
        "description": str(description),
        "agent": agent,
        "handoffs": sorted(set(inputs) | set(outputs), key=(list(inputs) + list(outputs)).index),
        "validators": [],
        "task": task,
    }


def _yaml_doc(comment: str, docs: list[dict]) -> str:
    return f"# {comment}\n" + yaml.safe_dump(docs, sort_keys=False, allow_unicode=True, width=100)


def _yaml_single(comment: str, doc: dict) -> str:
    return f"# {comment}\n" + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)


def _validator_readme(kind: str, spec: dict, args: dict) -> str:
    return (
        f"# {validator_name(kind)}\n\n"
        f"Judges `{kind}` ({spec['content_type']}): {spec.get('description', '')}\n\n"
        "Generic declarative check (`assets/lib/handoff_check.py`), configured by this\n"
        "validator's `args` in the task YAML:\n\n```yaml\n"
        + yaml.safe_dump(args, sort_keys=False)
        + "```\n\n"
        "**Calibration.** It checks presence, structure, parseability and stated bounds —\n"
        "what any correct result satisfies — never wording or an exact number. A line\n"
        "floor is set well under a real result. If it refuses a result that is right,\n"
        "the bar is wrong: fix the `args` and record why, do not pad the handoff.\n\n"
        "Reasons are written to `validator_report.txt` in the validation zone.\n"
        + _calibration(spec)
    )


def _calibration(spec: dict) -> str:
    text = str(spec.get("calibration") or "").strip()
    return f"\n## Calibration\n\n{text}\n" if text else ""


def _main_readme(plan: dict, tasks: list[dict], external: list[str]) -> str:
    rows = [
        f"| `{t['name']}` | {t['kind']} | {', '.join(t.get('inputs') or []) or '—'} | "
        f"{', '.join(t.get('outputs') or [])} | {t['goal']} |"
        for t in tasks
    ]
    pre = ""
    if external:
        pre = (
            f"\nExternal inputs ({', '.join(external)}) are published first by `publish_inputs`, "
            "and validated on publication.\n"
        )
    return (
        f"# main — {plan.get('package', 'task')}\n\n{plan.get('description', '')}\n\n"
        "This task runs nothing itself; its work is its subgraph.\n" + pre + "\n"
        "| task | agent | inputs | outputs | goal |\n|---|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n"
    )


def load_plan(path: Path) -> dict:
    plan = yaml.safe_load(path.read_text())
    if not isinstance(plan, dict):
        raise PlanError(f"{path}: a plan is a YAML mapping")
    return plan
