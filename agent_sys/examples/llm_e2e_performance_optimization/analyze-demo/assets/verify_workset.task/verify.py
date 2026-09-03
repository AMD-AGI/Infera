#!/usr/bin/env python3
"""What `verify_workset` runs: execute every driver on the target GPU.

forge-loop treats `--driver` as a protected file — its optimizing agent may not
modify it — so every conclusion it reaches about correctness and speed rests on
a driver this package wrote. A driver that is subtly wrong sends the
optimization somewhere plausible and useless, and nothing downstream notices. So
each one is executed here, on the hardware it will run on, before it is handed
over.

Per operator, in the serving container:

    python scripts/forge_driver.py                       -> SNR / allclose
    5 x  python scripts/forge_driver.py --warmup 10 --iters 20 --bench-mode

which is mission 3.2.7's "5次加权平均，每次运行loop 10次以上取平均" in the shape
`assets/lib/bench_stats.py` computes. That module is imported by this producer
and by `check_workset_runs`, so the two cannot disagree about what the weighted
average means.

**One operator's failure does not end the task.** It is recorded against that
operator and the rest continue; `check_workset_runs` then decides whether enough
of them measured. A body that exited on the first failure would throw away the
evidence for the others, and a program task's stdout is discarded on success —
`agent/backends/program.py` keeps a tail only on a non-zero exit — so anything
worth keeping has to be written into the handoff explicitly.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PACKAGE = Path(
    os.environ.get("AGENT_SYS_TASK_PACKAGE") or os.environ["AGENT_SYS_DEMO_PACKAGE"]
)
sys.path.insert(0, str(_PACKAGE / "assets" / "lib"))

import bench_stats  # noqa: E402
import store  # noqa: E402

_SNR = re.compile(r"SNR:\s*(-?[\d.]+)\s*dB", re.IGNORECASE)
_ALLCLOSE = re.compile(r"allclose:\s*(True|False)", re.IGNORECASE)

#: The canonical bench line the workset contract asks for: `time_ms: 0.1861`.
#: A single machine-readable key, so parsing does not depend on prose.
_TIME_MS = re.compile(r"^\s*time_ms:\s*(-?[\d.]+)\s*$", re.IGNORECASE | re.MULTILINE)

#: Fallback for a driver that prints the number in prose. Deliberately loose —
#: `0.1861 ms/iter over 20 iters` is what a real driver produced, and an earlier
#: keyword-anchored pattern rejected it while the measurement was perfectly
#: good. The last match wins, because a driver that prints per-case lines
#: followed by a total means the total.
_MS_PROSE = re.compile(r"(-?[\d.]+)\s*(?:ms\b|milliseconds)", re.IGNORECASE)

README = """# workset_evidence

## Purpose

What each workset's driver actually did on the target GPU: whether it ran,
whether it met the correctness gate, and how long the operator took.

This is the reading, not the apparatus. `operator_workset` holds the drivers;
this holds what they measured. Keeping them apart means a workset can be
re-measured without being rebuilt.

{headline}

## Schema

`items/text.json`:

```json
{{"generated_at": "...",
  "environment": {{"node": "<hostname>", "image": "...", "gpu_target": "gfx950"}},
  "protocol": {{"groups": 5, "warmup": 10, "iters": 20,
               "note": "mission 3.2.7: five weighted groups, >=10 iterations each"}},
  "summary": {{"operators": 0, "ran": 0, "passed": 0, "pass_ratio": 0.0}},
  "operators": [
    {{"operator_id": "...", "ran": true, "correct": true,
      "snr_db": 62.1, "allclose": true,
      "bench": {{"groups": 5, "iters_total": 100, "weighted_mean_ms": 0.42,
                "min_group_ms": 0.41, "max_group_ms": 0.43, "rsd": 0.012,
                "per_group_ms": []}},
      "failure": ""}}
  ]}}
```

`ran` is false when the driver could not be executed at all — a missing entry
point, an import error, a compilation failure. `correct` is false when it ran
and the numbers did not meet the gate. The two are different problems and are
recorded separately.

`items/logs/<operator_id>.txt` holds the driver's own stdout and stderr.

## Watch out

`bench.weighted_mean_ms` is a **standalone** measurement of one operator. The
`avg_us` in the upstream profile is a serving-time average over mixed batch
sizes with other kernels competing for the machine; the two are not comparable
and the second is not a baseline for the first. `forge_task.yaml`'s
`targets.baseline_wall_ms` takes the number from here.
"""

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "workset_evidence",
    "type": "object",
    "required": ["generated_at", "environment", "protocol", "summary", "operators"],
    "properties": {
        "generated_at": {"type": "string"},
        "environment": {"type": "object"},
        "protocol": {"type": "object"},
        "summary": {"type": "object"},
        "operators": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["operator_id", "ran", "correct"],
                "properties": {
                    "operator_id": {"type": "string"},
                    "ran": {"type": "boolean"},
                    "correct": {"type": "boolean"},
                    "snr_db": {"type": ["number", "null"]},
                    "allclose": {"type": ["boolean", "null"]},
                    "bench": {"type": ["object", "null"]},
                    "failure": {"type": "string"},
                },
            },
        },
    },
}


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set; this body has nowhere to write.")
    return value


#: How to get a shell onto the node holding the GPUs. **A site fact, so it is a
#: package variable** (`--var node_transport=`), not a constant.
#:
#: - `srun` — Slurm. `srun --jobid=<id> --overlap` joins the existing allocation
#:   rather than queueing for a new one, which is what lets this share a node
#:   with whatever else the job is holding. This is the default because it is
#:   what the cluster this package was written on has.
#: - `spur` — the spur scheduler. `spur exec <jobid>` routes by job id and takes
#:   no node name. Measured on `crsuse2-m2m-*`: Slurm's client binaries are not
#:   in the namespace at all (`command -v srun` finds nothing there), and the
#:   `/usr/local/bin/srun` visible from the login node is a re-implementation
#:   that wants a TTY and exits 128 without one.
#: - `local` — `agent-sys` is already running on the node holding the GPUs, so
#:   there is no gap to cross and the docker call is made directly. This is the
#:   cheapest and the least to go wrong; prefer it whenever the orchestrator and
#:   the GPU are the same machine, which `hostname` settles.
TRANSPORTS = ("srun", "spur", "local")


def _launcher(timeout_hint: str) -> list[str]:
    """The argv prefix that puts `bash -lc <...>` on the GPU node."""
    transport = (os.environ.get("AD_NODE_TRANSPORT") or "srun").strip()
    if transport not in TRANSPORTS:
        raise SystemExit(
            f"AD_NODE_TRANSPORT={transport!r} is not one of {TRANSPORTS}. "
            f"Supply it as --var node_transport=<one of those>."
        )
    if transport == "local":
        return ["bash", "-lc"]

    jobid = os.environ.get("AD_JOBID", "")
    node = os.environ.get("AD_GPU_NODE", "")
    if not jobid or (transport == "srun" and not node):
        raise SystemExit(
            f"the {transport!r} transport needs AD_JOBID"
            + (" and AD_GPU_NODE" if transport == "srun" else "")
            + f", and neither carries a default ({timeout_hint}). "
            f"Supply them: --var jobid=<job> --var gpu_node=<hostname>"
        )
    if transport == "spur":
        # `spur exec` routes by job id, so it takes no node name, and it does
        # not carry the caller's environment across — everything this body
        # needs is already inside the `docker run` string, so that is fine.
        return ["spur", "exec", jobid, "bash", "-lc"]
    return [
        "srun", f"--jobid={jobid}", "--overlap", "-N1", "-n1", "-w", node,
        "--chdir=/tmp", "bash", "-lc",
    ]


def run_remote(command: str, timeout: int) -> subprocess.CompletedProcess:
    """Run `command` inside the serving container on the allocated node.

    The container flags are the ones `glm53flash-demo/scripts/mix_worker.sh`
    uses; without `--device=/dev/kfd` a ROCm process fails with a device-open
    error that reads like a driver problem.
    """
    image = os.environ.get("AD_IMAGE", "")
    work = os.environ.get("AD_WORK_ROOT", "/data/agent_sys_analyze")

    # **The container runs as root, and `--user` is not an option.** aiter
    # compiles FlyDSL kernels on first use into
    # `/sgl-workspace/aiter/aiter/jit/flydsl_cache/`, a directory inside the
    # image; as a non-root user that write fails and every driver exits 1 with
    # `Permission denied` on a cache path. Measured — running as the invoking
    # user looks tidier and breaks the JIT outright.
    #
    # `PYTHONDONTWRITEBYTECODE` is what keeps root out of the staging tree
    # instead: without it the container leaves `__pycache__/*.pyc` owned by root
    # under a user-owned directory, and the next run's cleanup fails on a file
    # it did not create. `clear_staging` handles anything that still slips
    # through.
    inner = (
        f"docker run --rm "
        f"-e PYTHONDONTWRITEBYTECODE=1 "
        f"{_visible_devices()}"
        f"--device=/dev/kfd --device=/dev/dri --group-add video "
        f"--ipc=host --network=host --security-opt seccomp=unconfined "
        f"-v {work}:{work} -w {work}/verify "
        f"{image} bash -lc {json.dumps(command)}"
    )
    return subprocess.run(
        _launcher("per-operator timeout") + [inner],
        capture_output=True, text=True, timeout=timeout,
    )


def _visible_devices() -> str:
    """`-e HIP_VISIBLE_DEVICES=...`, when this run does not own the whole node.

    A card index is an identifier bound on a shared host, so it is a package
    variable (`--var visible_devices=2,3`) and not a constant. Empty means "use
    whatever the node gives us", which is right when the allocation is the whole
    node and wrong the moment it is not: without it a driver opens card 0 and
    competes with whoever else is holding it.
    """
    devices = (os.environ.get("AD_VISIBLE_DEVICES") or "").strip()
    return f"-e HIP_VISIBLE_DEVICES={devices} " if devices else ""


def parse_correctness(text: str) -> tuple[float | None, bool | None]:
    snr = _SNR.search(text)
    allclose = _ALLCLOSE.search(text)
    return (
        float(snr.group(1)) if snr else None,
        (allclose.group(1).lower() == "true") if allclose else None,
    )


def parse_ms(text: str) -> float | None:
    """The measured per-iteration time, in milliseconds.

    `time_ms:` first, because it is the line the workset contract asks a driver
    to print and it cannot be confused with anything else. Prose second, for a
    driver written against KernelForge's examples rather than against this
    contract.
    """
    explicit = _TIME_MS.findall(text)
    if explicit:
        return float(explicit[-1])
    hits = _MS_PROSE.findall(text)
    return float(hits[-1]) if hits else None


def measure(operator_id: str, timeout: int, groups: int, warmup: int, iters: int) -> dict:
    """One operator: correctness once, then `groups` timed groups."""
    record: dict = {
        "operator_id": operator_id,
        "ran": False,
        "correct": False,
        "snr_db": None,
        "allclose": None,
        "bench": None,
        "failure": "",
        "log": "",
    }
    base = f"cd code/{operator_id} && python scripts/forge_driver.py"

    try:
        done = run_remote(base, timeout)
    except subprocess.TimeoutExpired:
        record["failure"] = f"correctness run exceeded {timeout}s"
        return record
    log = [f"$ {base}", done.stdout, done.stderr]
    if done.returncode != 0:
        record["failure"] = f"correctness run exited {done.returncode}"
        record["log"] = "\n".join(log)
        return record

    record["ran"] = True
    snr, allclose = parse_correctness(done.stdout)
    record["snr_db"], record["allclose"] = snr, allclose
    threshold = float(os.environ.get("AD_SNR_THRESHOLD") or 30.0)
    if snr is not None:
        record["correct"] = snr >= threshold
        if not record["correct"]:
            record["failure"] = f"SNR {snr} dB is below the {threshold} dB gate"
    elif allclose is not None:
        record["correct"] = allclose
        if not allclose:
            record["failure"] = "allclose reported False"
    else:
        record["failure"] = (
            "the driver printed neither an 'SNR: <x> dB' nor an 'allclose: <bool>' "
            "line; forge-loop reads correctness only from those two"
        )
        record["log"] = "\n".join(log)
        return record

    measured = []
    for index in range(groups):
        command = f"{base} --warmup {warmup} --iters {iters} --bench-mode"
        try:
            done = run_remote(command, timeout)
        except subprocess.TimeoutExpired:
            record["failure"] = f"bench group {index} exceeded {timeout}s"
            break
        log += [f"$ {command}", done.stdout, done.stderr]
        if done.returncode != 0:
            record["failure"] = f"bench group {index} exited {done.returncode}"
            break
        milliseconds = parse_ms(done.stdout)
        if milliseconds is None:
            record["failure"] = f"bench group {index} printed no '<n> ms' figure"
            break
        measured.append({"iters": iters, "mean_ms": milliseconds})

    if len(measured) == groups:
        try:
            record["bench"] = bench_stats.summarize(measured)
        except bench_stats.BenchShapeError as error:
            record["failure"] = str(error)
    record["log"] = "\n".join(log)
    return record


def stage(content: Path) -> None:
    """Copy the worksets onto the node the drivers will run on.

    `AD_WORK_ROOT` is node-local. The handoff lives in the run store on the
    orchestrating host, and a container on another node cannot see it.
    """
    work = os.environ.get("AD_WORK_ROOT", "/data/agent_sys_analyze")
    staging = Path(work) / "verify"
    # Cleared from inside a container so that anything a previous root-run left
    # behind can actually be removed. `rm -rf` as the invoking user fails with
    # `Permission denied` on a root-owned `.pyc` and takes the whole task with
    # it — a second-run-only failure that is easy to miss.
    image = os.environ.get("AD_IMAGE", "")
    launcher = _launcher("staging")
    subprocess.run(
        launcher
        + [
            f"docker run --rm -v {work}:{work} {image} "
            f"bash -lc 'rm -rf {staging} && mkdir -p {staging} && chmod 777 {staging}'"
        ],
        check=True, capture_output=True, text=True, timeout=600,
    )
    # The transport shares the allocation but not a filesystem: the work root is
    # node-local. scp is avoided in favour of tar over the launcher's stdin,
    # which needs no credentials and no reachable sshd — and which degrades to a
    # plain local `tar` when the transport is `local`.
    archive = shutil.make_archive(str(Path("/tmp") / "workset"), "gztar", root_dir=content / "items", base_dir="code")
    with open(archive, "rb") as handle:
        subprocess.run(
            launcher + [f"tar xzf - -C {staging}"],
            stdin=handle, check=True, capture_output=True, text=True, timeout=600,
        )


def main() -> int:
    staged = store.declared_dir("operator_workset", direction="INPUT")
    if staged is None:
        raise SystemExit("AGENT_SYS_INPUT_OPERATOR_WORKSET does not name a readable directory.")

    directories = sorted(d.name for d in (staged / "items" / "code").iterdir() if d.is_dir())
    if not directories:
        raise SystemExit("the workset handoff holds no operator directory")

    stage(staged)

    timeout = int(os.environ.get("AD_PER_OP_TIMEOUT_S") or 900)
    groups, warmup, iters = 5, 10, 20
    operators = [measure(name, timeout, groups, warmup, iters) for name in directories]

    ran = sum(1 for o in operators if o["ran"])
    passed = sum(1 for o in operators if o["correct"] and o["bench"])

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "node": os.environ.get("AD_GPU_NODE", ""),
            "image": os.environ.get("AD_IMAGE", ""),
            "gpu_target": os.environ.get("AD_GPU_TARGET", ""),
            "gpu_type": os.environ.get("AD_GPU_TYPE", ""),
        },
        "protocol": {
            "groups": groups,
            "warmup": warmup,
            "iters": iters,
            "note": "mission 3.2.7: five weighted groups, at least ten iterations each",
        },
        "summary": {
            "operators": len(operators),
            "ran": ran,
            "passed": passed,
            "pass_ratio": round(passed / len(operators), 3),
        },
        "operators": [{k: v for k, v in o.items() if k != "log"} for o in operators],
    }

    dst = Path(_required("AGENT_SYS_OUTPUT_WORKSET_EVIDENCE"))
    items = dst / "items"
    (items / "logs").mkdir(parents=True, exist_ok=True)
    (items / "text.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (items / "schema").write_text(json.dumps(SCHEMA, indent=2), encoding="utf-8")
    for operator in operators:
        (items / "logs" / f"{operator['operator_id']}.txt").write_text(
            operator["log"] or "(no output captured)", encoding="utf-8"
        )

    headline = (
        f"{passed} of {len(operators)} workset(s) ran and measured cleanly "
        f"({ran} executed at all)."
    )
    failures = [o for o in operators if not (o["correct"] and o["bench"])]
    if failures:
        headline += "\n\nNot yet usable by forge-loop:\n\n" + "\n".join(
            f"- `{o['operator_id']}`: {o['failure'] or 'no measurement produced'}"
            for o in failures
        )
    (dst / "README.md").write_text(README.format(headline=headline), encoding="utf-8")

    print(f"verify_workset: {passed}/{len(operators)} passed, {ran} ran")
    for operator in operators:
        mark = "PASS" if operator["correct"] and operator["bench"] else "FAIL"
        detail = (
            f"{operator['bench']['weighted_mean_ms']:.4f} ms, rsd {operator['bench']['rsd']:.4f}"
            if operator["bench"]
            else operator["failure"]
        )
        print(f"  {mark} {operator['operator_id']}: {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
