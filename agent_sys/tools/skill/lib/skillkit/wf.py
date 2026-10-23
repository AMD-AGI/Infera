"""Emit a Claude Code Workflow script that executes a package's leaf graph.

One agent per leaf, started the moment every leaf it waits for has finished
(a promise per leaf — no stage barriers), so independent branches run side by
side exactly as the scheduler would dispatch them. Each agent drives the leaf
through `astk prepare / run-program / finish` and returns a typed status; a
leaf whose dependency did not succeed is skipped, never run on stale input.

The script is a starting point the skill hands to Claude: refining one large
AI leaf into several stages (its readme's STEPS) is an edit to this text.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from skillkit.pkg import Package
from skillkit.runner import ASTK

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "leaf": {"type": "string"},
        "status": {"type": "string", "enum": ["succeeded", "failed", "skipped"]},
        "finish_attempts": {"type": "integer"},
        "outputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"kind": {"type": "string"}, "verdict": {"type": "string"}},
                "required": ["kind", "verdict"],
            },
        },
        "finish_exit": {"type": "integer", "description": "exit code of the LAST astk finish call"},
        "summary": {"type": "string"},
    },
    "required": ["leaf", "status", "finish_exit", "summary"],
}


def _prompt(leaf_name: str, kind: str, run: Path, max_finish: int) -> str:
    astk = f"python3 {shlex.quote(str(ASTK))}"
    run = shlex.quote(str(run))  # type: ignore[assignment]
    common = (
        f"You are executing leaf `{leaf_name}` of an agent_sys task package inside Claude Code, "
        f"without the agent_sys runtime. Run dir: {run}.\n"
        f"1. `{astk} prepare --run {run} --leaf {leaf_name}` — prints the zone, env.sh and brief.md paths.\n"
        f"2. Read brief.md completely before doing anything else.\n"
    )
    if kind == "program":
        work = (
            f"3. This is a program leaf: `{astk} run-program --run {run} --leaf {leaf_name}`. "
            "Do NOT do its work by hand. If it exits non-zero, read logs/program.log in the zone, "
            "report the cause, and stop (a program body is not yours to rewrite).\n"
        )
    else:
        work = (
            "3. This is an AI leaf: do the work brief.md describes, following its STEPS and traps "
            'as written. Prefix every shell command with `. <zone>/env.sh && cd "$AGENT_SYS_MY_ZONE" && `. '
            "Write each output handoff (README.md with the exact sections + items/) where brief.md says.\n"
        )
    if kind == "program":
        gate = (
            f"4. `{astk} finish --run {run} --leaf {leaf_name}` is the quality gate. A program leaf's "
            "output is the program's: if finish reports NOT PASSED, do not hand-edit the handoff — "
            "report the verdict lines and the program log's relevant lines, status `failed`.\n"
        )
    else:
        gate = (
            f"4. `{astk} finish --run {run} --leaf {leaf_name}` is the quality gate. If it reports "
            "NOT PASSED, read every reason it prints, fix the HANDOFF (never a validator, never its "
            f"args), and run finish again — at most {max_finish} finish calls in total. A result that "
            "only passes by weakening evidence is a failure; report it as one.\n"
        )
    gate += (
        "Return status `succeeded` only if the last finish exited 0 (PASSED for every output), and that exit code as `finish_exit`; quote the "
        "verdict lines in `summary`."
    )
    return common + work + gate


def emit(
    pkg: Package,
    run: Path,
    *,
    max_finish: int = 3,
    root: str = "main",
    done: frozenset[str] = frozenset(),
) -> str:
    """`done`: leaves the run already has (succeeded or seeded) — not re-run."""
    leaves = pkg.leaves(root)
    plan = [
        {
            "name": leaf.name,
            "kind": leaf.agent_kind,
            "deps": leaf.deps,
            "done": leaf.name in done,
            "prompt": _prompt(leaf.name, leaf.agent_kind, run, max_finish),
        }
        for leaf in leaves
    ]
    phases = sorted(
        {leaf.ancestors[-1] if leaf.ancestors else root for leaf in leaves},
        key=lambda p: [x.ancestors[-1] if x.ancestors else root for x in leaves].index(p),
    )
    meta = {
        "name": f"agent-sys-run-{pkg.root.name}"[:60],
        "description": f"Execute {pkg.root.name} ({len(leaves)} leaves) in Claude Code with agent_sys validators as the gate",
        "phases": [{"title": p} for p in phases] + [{"title": "closure"}],
    }
    group = {leaf.name: (leaf.ancestors[-1] if leaf.ancestors else root) for leaf in leaves}
    for p in plan:
        p["phase"] = group[p["name"]]
    closure_prompt = (
        f"Run `python3 {shlex.quote(str(ASTK))} closure --run {shlex.quote(str(run))}` (exit 1 just means "
        "incomplete) and return its JSON output verbatim, followed by one line per leaf that did not "
        "succeed naming the first refusal reason from its finish output."
    )
    return f"""export const meta = {json.dumps(meta, indent=2)}

// Generated by `astk workflow` from {pkg.root}
// Leaves in listing order (already topological); deps are derived from handoff kinds + froms.
const PLAN = {json.dumps(plan, indent=2)}
const RESULT = {json.dumps(RESULT_SCHEMA)}
const CLOSURE_PROMPT = {json.dumps(closure_prompt)}

const done = {{}}
for (const leaf of PLAN) {{
  done[leaf.name] = Promise.all(leaf.deps.map(d => done[d])).then(async deps => {{
    if (leaf.done) return {{ leaf: leaf.name, status: 'succeeded', finish_exit: 0, summary: 'already done in the run state' }}
    const blocked = deps.filter(r => !r || r.status !== 'succeeded' || r.finish_exit !== 0)
    if (blocked.length) {{
      log(`skip ${{leaf.name}}: upstream not succeeded`)
      return {{ leaf: leaf.name, status: 'skipped', finish_exit: -1, summary: 'upstream: ' + blocked.map(r => r ? r.leaf + '=' + r.status : 'null').join(', ') }}
    }}
    const opts = {{ label: leaf.name, phase: leaf.phase, schema: RESULT }}
    if (leaf.kind === 'program') opts.effort = 'low'
    const r = await agent(leaf.prompt, opts)
    return r || {{ leaf: leaf.name, status: 'failed', finish_exit: -1, summary: 'agent returned nothing' }}
  }})
}}
const results = await Promise.all(PLAN.map(l => done[l.name]))

phase('closure')
const closure = await agent(CLOSURE_PROMPT, {{ label: 'closure', phase: 'closure', effort: 'low' }})
return {{ results, closure }}
"""
