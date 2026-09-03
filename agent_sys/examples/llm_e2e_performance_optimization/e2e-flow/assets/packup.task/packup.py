#!/usr/bin/env python3
"""Assemble the integration stage's deliverable in the experiment-result-packup layout.

Every input arrives as `AGENT_SYS_INPUT_<KIND>`, so this body reads the
environment rather than taking nine command-line arguments — the kinds it
consumes are declared in `steps/verdict.yaml` and duplicating that list here
would be a second place to keep it in step.

**`content_type: code`, not `reproducible`.** Laying a packup into `reproducible`
renames `results/` to `items/result` and leaves `REPRODUCE.md` with no item to
be, which destroys the thing `check_packup_shape` exists to check. `code`'s
`codes` item is unconstrained inside, so the layout survives.

REPRODUCE.md is written to be executed, not read: `check_packup_shape` counts its
command lines rather than its prose, because it is the one file in a packup that
somebody runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

#: **Eight, and they span all five stages.** This is the flow's one export, not
#: one stage's: `check_packup_shape` grades it on being a kit somebody who was
#: not here can follow, and a kit that carries only the integration numbers
#: cannot explain where the kernel under test came from.
#:
#: Six of `integration-demo`'s nine collapsed into two (CONTRACT.md 7) and three
#: upstream stages joined.
INPUTS = (
    "deploy_kit",
    "profiling_evidence",
    "operator_workset",
    "kernel_optimization",
    "patch_overlay",
    "stock.measurement",
    "patched.measurement",
    "integration_report",
)


def env_name(kind: str) -> str:
    """`env_mgr/grants.py:450 _env_name`, duplicated: uppercase, non-alphanumerics to `_`.

    `kind.upper()` was enough while every kind was an identifier. It is not now:
    `stock.measurement` reaches a body as `AGENT_SYS_INPUT_STOCK_MEASUREMENT`,
    and `STOCK.MEASUREMENT` is not a variable name — the lookup would simply
    miss and the input would be reported as absent.
    """
    return "".join(c if c.isalnum() else "_" for c in kind).upper()


def staged(kind: str) -> Path | None:
    value = os.environ.get(f"AGENT_SYS_INPUT_{env_name(kind)}")
    return Path(value) if value else None


def read_json(path: Path | None, default=None):
    if path is None:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--package", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    codes = out / "items" / "codes"
    for sub in ("results", "logs", "scripts", "handoffs"):
        (codes / sub).mkdir(parents=True, exist_ok=True)

    sources = {kind: staged(kind) for kind in INPUTS}
    missing = [k for k, v in sources.items() if v is None or not v.is_dir()]
    if missing:
        raise SystemExit(f"packup: these inputs did not arrive: {', '.join(missing)}")

    report = read_json(sources["integration_report"] / "items" / "text.json", {}) or {}
    verdict = report.get("verdict", {})
    accepted = bool(verdict.get("accepted"))
    manifest = {}
    for found in sorted((sources["kernel_optimization"]).glob("items/codes/*/apply/manifest.json")):
        manifest = read_json(found, {}) or {}
        break
    plan = read_json(sources["patch_overlay"] / "items" / "result" / "mounts.json", {}) or {}

    # ---- results -------------------------------------------------------------
    # The report, both arms' summaries, and the eval index. Flat files rather
    # than a directory tree: `results/` is what a reader opens first.
    shutil.copyfile(
        sources["integration_report"] / "items" / "report.md", codes / "results" / "report.md"
    )
    shutil.copyfile(
        sources["integration_report"] / "items" / "text.json",
        codes / "results" / "integration_report.json",
    )
    for arm in ("stock", "patched"):
        accept = sources[f"{arm}.measurement"] / "items" / "result"
        for name in ("smoke.json", "needle.json", "probe.json"):
            if (accept / name).is_file():
                shutil.copyfile(accept / name, codes / "results" / f"{arm}.{name}")
        index = accept / "lm_eval" / ".index"
        if index.is_file():
            shutil.copyfile(index, codes / "results" / f"{arm}.lm_eval.index")
        bench = sources[f"{arm}.measurement"] / "items" / "result"
        for round_dir in sorted(p for p in bench.glob("r*") if p.is_dir()):
            summary = round_dir / "summary.json"
            if summary.is_file():
                shutil.copyfile(summary, codes / "results" / f"{arm}.{round_dir.name}.summary.json")

    # ---- the upstream stages -------------------------------------------------
    # **This is the flow's export, not the integration stage's.** A kit that
    # carries only the two arms' numbers cannot explain where the kernel under
    # test came from, which m1 brought the service up, or what m2 measured that
    # made m3 pick this operator — and a reproducer who was not here needs all
    # four to get back to the same place.
    #
    # One representative artefact per stage rather than the whole handoff: the
    # handoffs themselves are the record, and a packup is the path through them.
    upstream = {
        "m1.deploy_kit": sorted(sources["deploy_kit"].glob("items/codes/**/README.md"))[:1],
        "m2.bench": sorted(sources["profiling_evidence"].glob("items/result/**/summary.json"))[:1],
        "m2.kernel_table": sorted(sources["profiling_evidence"].glob("items/result/**/*.csv"))[:1],
        "m3.workset": sorted(sources["operator_workset"].glob("items/codes/*/*.yaml"))[:1],
        "m4.verification": sorted(sources["kernel_optimization"].glob("items/codes/**/verification.json"))[:1],
        "m4.forge_result": sorted(sources["kernel_optimization"].glob("items/codes/**/forge_result.json"))[:1],
    }
    for label, found in upstream.items():
        if not found:
            # Recorded and not fatal. A stage that legitimately produced no such
            # artefact should leave a trace saying so, because "absent" and
            # "never looked for" are the same silence otherwise.
            (codes / "results" / f"{label}.MISSING").write_text(
                f"no artefact matching this stage's pattern was found in its handoff\n",
                encoding="utf-8",
            )
            continue
        shutil.copyfile(found[0], codes / "results" / f"{label}{found[0].suffix}")

    # ---- logs ----------------------------------------------------------------
    # Already gzipped upstream, and they stay that way: the seal scans every
    # UTF-8 file and a raw engine log is thousands of container-internal paths it
    # would refuse, none of which is a fact about this machine.
    for kind in ("deploy_kit", "profiling_evidence", "operator_workset",
                 "kernel_optimization", "patch_overlay",
                 "stock.measurement", "patched.measurement"):
        src = sources[kind] / "items" / "logs"
        if not src.is_dir():
            continue
        dest = codes / "logs" / kind
        dest.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copyfile(f, dest / f.name)

    # ---- scripts -------------------------------------------------------------
    # The `command` item of every handoff that has one. These are the runnable
    # record: each takes its site paths as shell variables, which is both what
    # makes them portable and what let them past the locality seal.
    for kind in ("patch_overlay", "stock.measurement", "patched.measurement"):
        src = sources[kind] / "items" / "command"
        if src.is_file():
            dest = codes / "scripts" / f"{kind}.command.sh"
            shutil.copyfile(src, dest)
            dest.chmod(0o755)

    # ---- the patch itself ----------------------------------------------------
    shutil.copytree(
        sources["kernel_optimization"] / "items" / "codes",
        codes / "handoffs" / "kernel_optimization",
    )
    shutil.copyfile(
        sources["patch_overlay"] / "items" / "result" / "mounts.json",
        codes / "handoffs" / "mounts.json",
    )

    # ---- environment.md ------------------------------------------------------
    env_stock = read_json(sources["stock.measurement"] / "items" / "env" / "deployment.json", {}) or {}
    context = read_json(sources["stock.measurement"] / "items" / "env" / "context.json", {}) or {}
    engine_argv = ""
    argv_file = sources["stock.measurement"] / "items" / "env" / "engine_argv.txt"
    if argv_file.is_file():
        engine_argv = argv_file.read_text(encoding="utf-8", errors="replace").strip()

    (codes / "environment.md").write_text(
        f"""# Environment

Both arms ran on one node inside one Slurm allocation, from one image, against
one checkpoint. That is the point: the only thing that differed between them was
the set of bind mounts, so anything the report shows as a difference has one
candidate cause.

| what | value |
|---|---|
| node | `{env_stock.get('node')}` |
| Slurm job | `{env_stock.get('slurm_jobid')}` |
| GPUs | 8 x AMD Instinct MI355X (gfx950) |
| engine image | `{env_stock.get('image')}` |
| image id | `{env_stock.get('image_id')}` |
| model | `{env_stock.get('served_model_name')}`, FP8 |
| tensor parallel | {env_stock.get('tp_size')} |
| topology | MIX (aggregated); prefill and decode are the same process |
| router | infera, kv-aware policy, python backend |
| discovery | etcd on port {(env_stock.get('ports') or {}).get('etcd')} — not 2379, which the node's Kubernetes control plane holds over TLS |
| AIPerf | `{(context.get('bench') or {}).get('rounds')}` round(s), own container (the engine image ships Python 3.10, AIPerf needs 3.11+) |
| evaluator | `sglang.test.run_eval`, inside the engine image, `--thinking-mode glm-45` |

## The engine, as it actually ran

Recorded from the running process rather than from what the bring-up asked for,
because a self-declared flag cannot catch a silent fallback:

```
{engine_argv}
```

## What the patch changed

Operator `{manifest.get('operator_id')}` (`{manifest.get('logical_operator')}`),
{len(manifest.get('files') or [])} file(s), applied as `{manifest.get('apply_mode')}`.

Each file is bind-mounted read-only over its path inside the container. This
works because sglang is installed into the image in editable mode, so the
interpreter reads the image's source tree directly. `handoffs/mounts.json` lists
each mount with the hash of what it replaced and the hash of what replaced it,
and `patched.measurement`'s `env/container_hashes.tsv` records what those paths
hashed **inside the running container** — which is the only static proof that a
mount took, because a bind mount leaves the path unchanged and `__file__`
therefore reads identically on both arms.
""",
        encoding="utf-8",
    )

    # ---- notes.md ------------------------------------------------------------
    reasons = verdict.get("reasons") or []
    is_mock = (manifest.get("expect") or {}).get("source") == "mock"
    (codes / "notes.md").write_text(
        f"""# Notes

## What this run can and cannot tell you

It compares two deployments that differed in one variable. It does not measure
GLM-5.3-Flash, and it does not measure the optimisation in isolation — the
operator's standalone correctness and timing belong to the analyze stage's
`workset_evidence`, upstream of here.

{"**The patch under test is a mock.** It adds two log lines and changes no arithmetic, so the correct answer was 'no difference'. An accepted verdict is evidence that the pipeline runs and that the comparison does not invent regressions; it is not evidence about any real optimisation." if is_mock else "The patch under test is a real change."}

## Reading the numbers

The eval scores carry Wilson intervals and their difference carries a Newcombe
interval, because at {(context.get('eval') or {}).get('num_examples')} questions two runs of the same deployment differ
by several points routinely. "The score went down" is not evidence; an interval
that excludes zero is.

The two replay rounds are not comparable to each other. Round 1 is cold for this
trace and round 2 is warm, and on this trace the two differ by roughly an order
of magnitude — a Mooncake trace carries `hash_ids`, so prefix hit rate decides
how much prefill there is to do.

The needle has two lengths and only the shorter is a pass/fail. The longer one is
recorded because a stock deployment was measured to fail its head depth, and a
gate the baseline cannot pass is a gate that gets switched off. Needle results
are regression evidence, not a long-context capability claim.

## Verdict

{"Accepted." if accepted else "Rejected:"}
{chr(10).join('- ' + r for r in reasons) if reasons else ''}

## Known limits of this stage

- A patch that is mounted but never entered would give two identical arms and a
  green report. `check_patch_live` guards it with two layers — the hash inside
  the running container, and the markers the patch declared — and the second
  layer only exists when the patch declares markers.
- Patches that need compiling (HIP, CK, assembly) cannot be delivered as an
  overlay and are out of scope; `apply_mode: rebuild` fails fast rather than
  being silently ignored.
- There is no repeat within a replay round, so a single round has no interval of
  its own. The performance bars are thresholds, not significance tests.
""",
        encoding="utf-8",
    )

    # ---- REPRODUCE.md --------------------------------------------------------
    (codes / "REPRODUCE.md").write_text(
        """# Reproduce

Reproducing *this* result means reproducing the comparison, which needs two cold
deployments and about an hour. Reproducing enough to believe the record is
cheaper, and that is what this file asks for.

## Success is

1. the stock deployment comes up and serves,
2. `smoke.py` passes all four checks,
3. the gated needle length retrieves at all three depths,
4. one 200-question `gsm8k` run produces a score.

That is about twenty-five minutes, most of it the cold start.

## What you need

- an allocation holding one 8-GPU MI355X node, and the engine image built on
  **that** node — docker images are node-local, and the build is about ten
  minutes from the Dockerfile in `examples/glm53flash-demo/patches/`
- the GLM-5.3-Flash FP8 checkpoint readable from the node
- the GSM8K test split, 1319 rows
- the `integration-demo` package, reachable from the node by absolute path

## Steps

Export the site paths once. `PKG` is the package directory and the three script
directories are derived from it rather than written out in full. That is not
style: the publication seal reads any two-segment slash-separated run as a
filesystem path unless the character before it says otherwise, and a
placeholder in angle brackets does not — so spelling those directories out
would make this document unpublishable. A shell variable in front of the slash
does say otherwise, which is why they are written this way:

```bash
export PKG=<the integration-demo package directory>
export MODEL_MOUNT=<directory holding the checkpoint>
export WORK_ROOT=<node-local work area, on local disk>
export GSM8K_SRC=<the GSM8K test split>
export NODE_IP=<the node's data-plane IP>
export SCRIPTS="$PKG/assets/serve"
export ACCEPT="$PKG/assets/accept"
export BENCH="$PKG/assets/bench"
```

Bring the stock arm up. `MOUNT_SPEC` empty is what makes it stock:

```bash
MOUNT_SPEC= bash scripts/deployment_stock.command.sh
```

Run the correctness suite:

```bash
URL=http://$NODE_IP:8100 OUT=$WORK_ROOT/accept bash scripts/acceptance_stock.command.sh
```

Read the three result files it writes: `smoke.json` must have `ok: true`,
`needle.json` must have `ok: true`, and `lm_eval/.index` must have one row with a
scored count at or near 200.

To go further and reproduce the comparison, rebuild the overlay and bring the
patched arm up with it:

```bash
export IMAGE=<the engine image>
export OVERLAY_ROOT="$WORK_ROOT/overlay"
export PATCHES=handoffs/kernel_patch/patches
export MANIFEST=handoffs/kernel_patch/manifest.json
export ROOTS="$PKG/assets/lib/container_roots.yaml"
bash scripts/patch_overlay.command.sh
```

That prints one `-v` argument per file. Write them as a `host<TAB>container`
file, point `MOUNT_SPEC` at it, and repeat the two steps above against
`scripts/deployment_patched.command.sh` and
`scripts/acceptance_patched.command.sh`.

## What will not match

Absolute throughput. It depends on how warm the radix cache is, and the same
configuration replayed twice on the same trace was measured at 631 and 1004
output tokens per second. What should match is the *comparison*: whichever
numbers you get, the two arms should sit within the bars recorded in
`results/integration_report.json` under `bars`.
""",
        encoding="utf-8",
    )

    # ---- README.md -----------------------------------------------------------
    perf = [
        row
        for row in report.get("performance", [])
        if row.get("metric") == "output_token_throughput_tps" and row.get("round") == "r1"
    ]
    headline = perf[0] if perf else {}
    (codes / "README.md").write_text(
        f"""# Integration acceptance — {manifest.get('operator_id')} on GLM-5.3-Flash

**Verdict: {"ACCEPTED" if accepted else "REJECTED"}.**

The last stage of the end-to-end performance optimisation pipeline: take the
kernel-optimization stage's deliverable, put it in front of a real deployment,
and decide whether it broke or slowed anything.

## What was done

Two deployments of GLM-5.3-Flash on one MI355X node, in one session, differing in
exactly one thing — {len(plan.get('mounts', []))} read-only bind mount(s) carrying
the patched file(s). Each arm was measured by the same script in the same order:

1. **smoke** — one `mixed` worker, the served model name, an arithmetic answer
   that must be 391, and 512 tokens of prose with no 8-gram repeated more than
   four times.
2. **needle** — a lexical passphrase buried at three depths of a multi-chunk
   prompt, at two lengths. Prompt length is read back from
   `usage.prompt_tokens` rather than estimated.
3. **probe** — the gate on the eval: reachability, answerability, stability
   under repeats and concurrency, and that a long shared prefix does not change
   the answer.
4. **llm-eval** — `sglang.test.run_eval` over `gsm8k` and `mixed_prefix_gsm8k`.
5. **trace replay** — AIPerf against a Mooncake production trace, {(context.get('bench') or {}).get('rounds')} round(s).

## What came out

`results/report.md` is the argument. `results/integration_report.json` is the
same content for a machine, and it carries the thresholds the verdict used, so
the verdict can be re-read later against the bar it was actually decided against.

{f"Headline: output token throughput {headline.get('stock')} -> {headline.get('patched')} on the first replay round, {headline.get('rel_delta', 0):+.1%}, verdict `{headline.get('verdict')}`." if headline.get('rel_delta') is not None else "See results/report.md for the per-metric table."}

{"Rejected for: " + "; ".join(reasons) if reasons else "No smoke check, needle depth, eval interval or performance metric moved past its bar."}

## Layout

```
README.md          this file
REPRODUCE.md       what to run to reproduce it, and what will not match
environment.md     node, image, engine argv as it actually ran, what the patch changed
notes.md           what this run can and cannot tell you
results/           the report, both arms' summaries, the eval index
logs/              every handoff's logs, gzipped as they were published
scripts/           the runnable `command` item of every handoff that has one
handoffs/          the patch set itself and the mount plan
```

## Read this before quoting a number

Neither arm's numbers mean anything alone, and the two replay rounds do not mean
anything against each other — round 1 is cold for the trace and round 2 is warm.
`notes.md` says what else. Generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}.
""",
        encoding="utf-8",
    )

    (out / "items" / "watchout").write_text(
        "Everything in results/ is a comparison between two arms measured in one session.\n"
        "Lifting a single number out of it and quoting it as GLM-5.3-Flash's throughput or\n"
        "GSM8K score would be wrong twice: the arms were run back to back on one node with\n"
        "one trace, and the replay rounds differ in cache warmth by roughly an order of\n"
        "magnitude by construction.\n"
        "\n"
        "logs/ is gzipped and stays that way. The raw engine logs carry thousands of\n"
        "container-internal absolute paths, none of them a fact about the machine that ran\n"
        "this, and the publication seal refuses text files that name them.\n",
        encoding="utf-8",
    )

    (out / "README.md").write_text(
        f"""# integration_packup

## Purpose

The integration stage's deliverable, in the `experiment-result-packup` layout: a
directory somebody can be handed that says what was tested, what happened, how to
reproduce it, and what not to conclude.

Verdict: **{"ACCEPTED" if accepted else "REJECTED"}**.

## Interface

`items/codes/` is the packup. Start at `README.md`, then `results/report.md` for
the argument and `REPRODUCE.md` to run it again. `scripts/` holds the `command`
item of every handoff in the run, each of which is executable and takes its site
paths as shell variables. `handoffs/kernel_patch/` is the patch set itself, so the
packup describes a change it also carries.

## Boundary

This is a record, not a runnable pipeline. Reproducing the whole comparison needs
an allocation, an engine image built on the node it will run on, and about an
hour; `REPRODUCE.md` defines a smaller success condition than that on purpose and
says which numbers will not match.

It carries no model weights, no container image and no source tree — only the
patch, the hashes that pin it to one image, and the evidence.
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(Path(args.package) / "assets" / "lib" / "redact.py"),
            str(out),
            f"TASK_PACKAGE={args.package}",
            f"ZONE={Path.cwd()}",
            "TMPDIR=/tmp",
            f"HOME={Path.home()}",
        ],
        check=True,
    )

    files = sum(1 for _ in codes.rglob("*") if _.is_file())
    print(f"packup: {files} file(s), verdict {'ACCEPTED' if accepted else 'REJECTED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
