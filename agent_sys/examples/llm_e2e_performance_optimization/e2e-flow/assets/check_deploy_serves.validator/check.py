#!/usr/bin/env python3
"""`check_deploy_serves` — usability, **strong**. Program, no model call.

Mission M1.2.3, the *real-run* validator: bring the service up **from the kit's
own scripts**, curl `/health`, send a researched set of diagnostic probes, then a
1k-in / 1k-out concurrency-16 three-minute load.

The four phases, and what each is worth on its own:

  1. **bring-up** — `scripts/deploy.sh` from the kit under test, with the three
     `runtime_contract` parameters re-pointed so this deployment cannot collide
     with the one that produced the kit (which may still be up). This phase alone
     is the check M2.3 leans on when it deletes the `serve_baseline` step: it is
     the proof that a later stage can deploy from this handoff and nothing else.
  2. **probes** — `probes.yaml`, executed by `probe_runner.py` on the node.
     Nothing here invents a probe (M1.2.3.3).
  3. **load** — `assets/bench/aiperf_synthetic.sh`, which is the integration
     package's `aiperf_replay.sh` with the trace swapped for the synthetic
     generator. **Not a third load generator.**
  4. **teardown** — always, including on every failure path, because the thing
     this body leaves behind on a shared node is a GPU nobody else can use.

**Where this runs and why it looks the way it does.** The validator body runs on
the login node, which has no GPU and no docker; the deployment runs on the held
compute node. So every phase is dispatched through `assets/lib/remote.sh`, the
same seam every other body in this package uses, and the probe plan is resolved
here and executed there by a standard-library-only program.

**Everything site-specific arrives through `args`, not through the environment.**
A validator declares no agent, so the package's `env` block — `E2E_TRANSPORT`,
`E2E_JOBID`, all of it — never reaches this body; only the policy-derived
environment does. That is measured, and the previous stage records one run lost
to it. What is *not* in `args` is everything the kit itself recorded: the node,
the model, the image, the ports. Those are read out of the kit's own
`codes/environment.yaml`, because the kit is the thing under test and a probe
pointed at anything but what the kit wrote is testing something else.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import schema as schema_lib  # noqa: E402
import zone  # noqa: E402
from workset_io import arg_num  # noqa: E402 — CONTRACT §4.2, the one numeric-arg reader

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"
REMOTE_SH = LIB / "remote.sh"


class NodeError(RuntimeError):
    def __init__(self, command: str, returncode: int, output: str):
        self.command, self.returncode, self.output = command, returncode, output
        super().__init__(f"node command failed (rc={returncode}): {command}\n{output}")


def on(command: str, transport: dict, *, check: bool = True, timeout: int | None = None) -> str:
    """Run `command` on the compute node, through the one seam this package has.

    `remote.sh` is sourced rather than reimplemented: a second spelling of
    `srun --overlap …` here would be a second thing to keep in step with it. The
    command is passed as a positional parameter, so it needs no quoting of its
    own.
    """
    env = dict(os.environ)
    # `_`-prefixed keys are this function's own parameters, not variables to
    # export. **Read, never popped**: `transport` is one dict reused across every
    # call, so a `pop` here fixed the first call and left every later one
    # unpatched — measured as bring-up succeeding and the very next `cat` dying
    # `rc=127`.
    env.update({k: str(v) for k, v in transport.items() if v and not k.startswith("_")})

    # **The transport binary is not on a validation zone's `PATH`.** Measured:
    # `spur` lives in `/usr/local/bin`, and a validator body is started with a
    # *closed* environment in which `PATH` is deliberately absent
    # (`validator/environment.py`), so POSIX `sh` substitutes its built-in
    # `/usr/bin:/bin`. `remote.sh`'s probe then finds neither `spur` nor `srun`,
    # falls through to its `spur` default, and the call dies `rc=127,
    # spur: command not found` — reported as the *kit's* `deploy.sh` failing,
    # which is three layers from the cause.
    #
    # A site fact, so it is a parameter (CONTRACT.md §6) rather than a literal,
    # and it is appended rather than replacing: nothing here should be able to
    # take `python3` away from a body that found one.
    extra = transport.get("_path_extra") or ""
    if extra:
        parts = [p for p in env.get("PATH", "").split(":") if p]
        env["PATH"] = ":".join(parts + [p for p in extra.split(":") if p and p not in parts])

    # **And the transport binary needs its own environment, which the closed
    # environment also strips.** Measured: `spur` reads `SPUR_CONTROLLER_ADDR`
    # and without it exits 1 with `failed to connect to controller … Connection
    # refused` — a message that names the controller and not the missing
    # variable, and which this body would otherwise report as the *kit's*
    # `deploy.sh` failing. Same class as the `PATH` above and the same remedy: a
    # site fact, passed as a parameter, injected here.
    for pair in (transport.get("_env_extra") or "").split():
        name, _, value = pair.partition("=")
        if name and value:
            env[name] = value
    proc = subprocess.run(
        ["bash", "-c", f'. "{REMOTE_SH}"; on "$1"', "_", command],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if check and proc.returncode != 0:
        raise NodeError(command, proc.returncode, output)
    return output


def seconds(parameters: dict, name: str, default: int) -> int:
    """A numeric `args` value, as a number. Delegates to the shared reader.

    **Every value in `args.json` arrives as a string**, whatever it looked like
    in the step yaml — `'${deploy_bringup_timeout_seconds:-3600}'` reaches a body
    as `"3600"`. Passing that to `subprocess.run(timeout=…)` raises
    `TypeError: unsupported operand type(s) for +: 'float' and 'str'` from inside
    the timeout arithmetic, naming neither the parameter nor the caller. Measured
    in a real run; a hand-written `args.json` with JSON numbers hides it, which
    is why it survived several standalone passes.

    **And in a validator the consequence is worse than a wrong answer.** m2
    measured it: the `TypeError` escapes, the body exits non-zero **before
    writing `verdict.json`**, and the phase reads a *broken validator* rather
    than a refused handoff — so the failure points at the checker rather than at
    the kit. That is exactly how this one presented.

    `workset_io.arg_num` and not a bare `int()`, per CONTRACT §4.2: bare `int()`
    fixes this arithmetic half and leaves the truthiness half, where
    `args.get(n) or default` reads the default when the operator said `0` —
    and, worse, *works* on the `${...}` string form while failing on a genuine
    yaml integer, so half an `args` block behaves differently from the other
    half.
    """
    return int(arg_num(parameters, name, default, cast=int))


def envvars(mapping: dict) -> str:
    """`A=1 B=2 ` — an explicit prefix, because no transport carries an environment.

    Measured on this cluster: `MARK=x spur exec <id> bash -lc 'echo $MARK'`
    prints empty. A body that relied on inheritance would work for an operator
    whose login shell happens to export the names and fail for everyone else.
    """
    return "".join(f"{k}={shlex.quote(str(v))} " for k, v in mapping.items())


# --------------------------------------------------------------------------- #
# resolving the plan


def resolve(text: str, bindings: dict[str, str]) -> str:
    for name, value in bindings.items():
        text = text.replace("${" + name + "}", str(value))
    return text


def resolve_deep(node, bindings: dict[str, str]):
    if isinstance(node, str):
        return resolve(node, bindings)
    if isinstance(node, list):
        return [resolve_deep(item, bindings) for item in node]
    if isinstance(node, dict):
        return {key: resolve_deep(value, bindings) for key, value in node.items()}
    return node


def build_plan(probes: dict, bindings: dict[str, str], available: set[str]) -> tuple[dict, list[str]]:
    """`probes.yaml` plus this deployment's facts, as a plan `probe_runner` can run.

    A probe whose `when:` condition is not met is **dropped and named**, not
    silently skipped and not failed: `engine_endpoint_known` is false for a
    deployment shape that does not publish the engine's own port, and that is a
    legitimate kit rather than a broken one.
    """
    plan, dropped = [], []
    for probe in probes["probes"]:
        condition = probe.get("when")
        if condition and condition not in available:
            dropped.append(f"{probe['name']} (needs {condition})")
            continue
        resolved = resolve_deep(probe, bindings)

        # `oversize_prompt_tokens` stays in the plan as a **number** and is
        # expanded by `probe_runner.py` on the node.
        #
        # **Measured 2026-09-03, first run against a live node.** Expanding it
        # here put a ~200 KB prompt into the plan, and the plan travels to the
        # node inside the command string — so the call died with
        # `OSError: [Errno 7] Argument list too long: 'bash'`, after a successful
        # bring-up, in a way no static fixture could have produced. A plan
        # carries intent; a filler string is not intent.
        size = resolved["request"].get("oversize_prompt_tokens")
        if size is not None:
            resolved["request"]["oversize_prompt_tokens"] = int(size)
        plan.append(resolved)
    return {"probes": plan}, dropped


# --------------------------------------------------------------------------- #
# the load's own acceptance


def judge_load(summary: dict, accept: dict, load_shape: dict) -> list[str]:
    """The load's floors, from `probes.yaml`. Deliberately low — see the yaml.

    A criterion whose input is absent is reported as **unevaluated**, never as
    passed. The whole package exists because a previous stage reported ten
    validators PASS over a run in which every result was zero.
    """
    faults = []

    # `summarise.py` writes `{metrics: {...}, missing: [...], source: ...}`. This
    # read the top level for one revision and both criteria came back
    # "unevaluated" against a load that had in fact succeeded — the honest
    # refusal working, and pointing at the wrong thing. `or summary` keeps a flat
    # document working, since the shape is that module's to change.
    metrics = summary.get("metrics") or summary
    # What the summariser itself could not find. Surfaced rather than inferred
    # from an absent key: "AIPerf did not report it" and "I looked in the wrong
    # place" are different faults and only one of them is the deployment's.
    for name in summary.get("missing") or []:
        print(f"check_deploy_serves: AIPerf reported no {name!r}")

    count = (metrics.get("request_count") or {}).get("avg")
    floor = accept.get("min_completed_requests")
    if floor is None and accept.get("min_completed_requests_per_slot"):
        # Derived from the window rather than fixed: see `probes.yaml`'s note.
        # `max(1, …)` because a window shorter than one request still has to
        # demand that every slot completed once.
        slots = int(load_shape["concurrency"])
        per_slot = int(accept["min_completed_requests_per_slot"])
        # **Measure the per-request time; do not estimate it.** `probes.yaml`
        # already said this — *"derived from the window and the **measured**
        # per-request time"* — and the code used a constant instead. That
        # divergence refused rung 1 on 2026-09-04: the engine measured 42.51 ms
        # ITL against the 32.5 ms the constant was calibrated on, so 1024 tokens
        # took 43.5 s not 33 s, four rounds fit in 180 s not five, and the run
        # completed **exactly** the 64 requests its own speed allows against a
        # floor of 80. It failed for a reason about the window rather than about
        # the deployment, which is verbatim what the scaling was introduced to
        # prevent — fixed for a shorter window, reproduced at a slower engine.
        #
        # **And the constant had no provenance.** `32.5` occurs exactly once in
        # this package: in the comment justifying it. No artefact carries it.
        #
        # The estimate stays as the fallback for a summary that reports neither
        # value, and only for that.
        # **`request_latency_ms` is the quantity, not a proxy for it.** The first
        # version of this multiplied `output_sequence_length × inter_token_latency`
        # — a product of two means, which omits prefill and needs two keys to be
        # spelled right. It needed one of them spelled right and got it wrong:
        # the summary's key is `inter_token_latency_ms`, the lookup asked for
        # `inter_token_latency`, and the miss fell through to the constant. **The
        # resulting failure was byte-identical to the one being fixed**, under a
        # comment explaining at length why it could not be the estimate.
        #
        # Measured on the refused run: direct 44.5 s, product 43.5 s, both giving
        # floor 64 against 64 completed. The direct one is right for the reason
        # rather than by agreement — it is the per-request wall time this floor
        # is about, prefill included.
        latency_ms = (metrics.get("request_latency_ms") or {}).get("avg")
        osl = (metrics.get("output_sequence_length") or {}).get("avg")
        itl = (metrics.get("inter_token_latency_ms") or {}).get("avg")
        if latency_ms:
            est = max(1.0, float(latency_ms) / 1000.0)
            basis = f"measured request_latency_ms.avg {float(latency_ms):.0f} ms"
        elif osl and itl:
            est = max(1.0, float(osl) * float(itl) / 1000.0)
            basis = f"measured {float(osl):.0f} tok x {float(itl):.2f} ms (no request_latency_ms)"
        else:
            est = float(accept.get("seconds_per_request_estimate", 35))
            # **The fallback announces itself.** A missing key and a summary that
            # legitimately reports neither value took the same silent path, and
            # that is what let a misspelling look like the fix not working rather
            # than the fix not running. Now the report says which happened.
            basis = (
                f"ESTIMATED {est:.0f}s/request — the summary reported neither "
                f"request_latency_ms nor output_sequence_length x inter_token_latency_ms. "
                f"If those metrics exist under other names this floor is not measured."
            )
            print(f"check_deploy_serves: WARNING: {basis}")
        rounds = max(1, int(int(load_shape["duration_seconds"]) // est))
        floor = slots * per_slot * rounds
        # Said out loud because the number is now run-dependent: a reader
        # comparing two runs' floors must be able to see why they differ.
        print(
            f"check_deploy_serves: completion floor {floor} = {slots} slot(s) x "
            f"{per_slot} x {rounds} round(s) over {load_shape['duration_seconds']}s ({basis})"
        )
    if floor is not None:
        if count is None:
            faults.append("request count is absent from the AIPerf summary — unevaluated, not passed")
        elif count < floor:
            faults.append(f"{count:.0f} completed request(s) under load, needs {floor}")

    mean_osl = (metrics.get("output_sequence_length") or {}).get("avg")
    floor = accept.get("min_mean_output_tokens")
    if floor is not None:
        if mean_osl is None:
            faults.append("output sequence length is absent from the AIPerf summary — unevaluated, not passed")
        elif mean_osl < floor:
            faults.append(
                f"mean output length {mean_osl:.0f} tokens, needs {floor}: the engine "
                f"stopped early, so every per-token number describes a shorter "
                f"workload than the one that was asked for"
            )
    return faults


# --------------------------------------------------------------------------- #
# one kit


def check_one(content: Path, parameters: dict, transport: dict, probes: dict) -> list[str]:
    layout_name = parameters.get("layout", "deploy_kit.layout")
    import yaml

    layout = yaml.safe_load(
        (schema_lib.package_root() / "assets" / "schemas" / f"{layout_name}.yaml").read_text()
    )
    package = schema_lib.package_root()

    # ---- the kit, and what it says about itself ----------------------------
    codes = content / layout["anchors"]["codes"]["path"]
    pattern = re.compile(layout["anchors"]["packup"]["name_pattern"])
    found = [e for e in sorted(codes.iterdir()) if e.is_dir() and pattern.match(e.name)] if codes.is_dir() else []
    if len(found) != 1:
        return [f"expected exactly one packup directory under items/codes, found {len(found)}"]
    kit = found[0]

    record = codes / "environment.yaml"
    if not record.is_file():
        return ["no codes/environment.yaml — nothing says what this kit deploys or where"]
    environment = schema_lib._read_doc(record)
    fixed = environment.get("fixed") or {}
    served = fixed.get("served_model_name") or fixed.get("model_name")

    # ---- re-point every identifier this deployment binds -------------------
    # A fresh tag, a port base well clear of the recorded one, and a workdir of
    # our own. The run that produced this kit may still be up, and a validator
    # that takes its container name or its port is a validator that breaks the
    # thing it is checking.
    tag = f"serves-{uuid.uuid4().hex[:8]}"
    work_root = f"{parameters['work_root'].rstrip('/')}/{tag}"
    overrides = {
        # Root-owned `__pycache__` inside the handoff is the first half of the
        # problem `reclaim.sh` cleans up after; not creating it is cheaper than
        # chowning it. CONTRACT.md §5.0.
        "PYTHONDONTWRITEBYTECODE": "1",
        "E2E_KIT_RUN_TAG": tag,
        "E2E_KIT_PORT_BASE": str(parameters["port_base"]),
        "E2E_KIT_WORK_ROOT": work_root,
    }
    handshake = f"{work_root}/deployment.json"

    faults: list[str] = []
    deploy = parameters.get("deploy_entrypoint", "scripts/deploy.sh")
    teardown = parameters.get("teardown_entrypoint", "scripts/teardown.sh")

    print(f"check_deploy_serves: kit {kit.name}, tag {tag}, port base {overrides['E2E_KIT_PORT_BASE']}")

    try:
        # ---- 1. bring-up, from the kit's own scripts ------------------------
        print("check_deploy_serves: 1/4 bring-up")
        try:
            output = on(
                f"mkdir -p {shlex.quote(work_root)} && cd {shlex.quote(str(kit))} && "
                f"{envvars(overrides)}bash {shlex.quote(deploy)}",
                transport,
                timeout=seconds(parameters, "bringup_timeout_seconds", 3600),
            )
        except NodeError as exc:
            return [f"{deploy} failed (rc={exc.returncode}); last output:\n{exc.output[-4000:]}"]
        except subprocess.TimeoutExpired:
            return [
                f"{deploy} did not finish within "
                f"{seconds(parameters, 'bringup_timeout_seconds', 3600)}s. A cold start is "
                f"minutes of JIT and weight load, so this budget is slack rather than a "
                f"target — a timeout here means it hung, not that it was slow"
            ]
        print(output[-2000:])

        # The handshake `runtime_contract` mandates. Its absence is the fault,
        # not a reason to go guessing at ports.
        try:
            raw = on(f"cat {shlex.quote(handshake)}", transport)
            deployment = json.loads(raw)
        except (NodeError, ValueError) as exc:
            return [
                f"{deploy} exited 0 but wrote no readable handshake at {handshake}: {exc}. "
                f"`deploy_kit.layout.yaml` runtime_contract.handshake makes this file the "
                f"one way a consumer learns the endpoint, and m2 deploys from this kit too"
            ]
        missing = [
            key
            for key in layout["runtime_contract"]["handshake"]["required_keys"]
            if not deployment.get(key)
        ]
        if missing:
            return [f"the handshake at {handshake} is missing {missing}"]

        router = deployment["endpoint"]
        engine = deployment.get("engine_endpoint")

        # `runtime_contract.writable_work_root`, answered rather than inferred.
        #
        # The failure this catches **reports success**: SGLang writes a profiler
        # trace to a path the *container* sees, so with no mount docker creates
        # the directory in the container layer, the capture succeeds, and the
        # host sees nothing. Measured by m2, and it is why m2's old bring-up
        # carried a `docker inspect` check.
        #
        # Checked from the host side rather than by exec'ing into the container:
        # the property that matters is that a file written **inside** appears
        # **outside**, and `docker exec` proves that in one direction only if the
        # reader is also inside. So: write from inside, read from outside.
        # **`containerized: false` is declared, not inferred.** A deployment that
        # runs no container has no boundary to cross, so the property is
        # satisfied trivially — but "there is no container" and "the container
        # died during bring-up" look identical from `docker exec`, and one of
        # those is a pass and the other is the fault this check exists to find.
        # So the kit says which it is, and a kit that says nothing is treated as
        # containerised, which is the safe default.
        inside = deployment["work_root_in_container"]
        # **The host side comes from the handshake too.** Comparing against this
        # body's own `work_root` assumed the kit mounts exactly that directory;
        # the real kit mounts a child of it (`pick_params.sh` writes `DK_RUN_DIR`
        # into `run.env`, which `env.sh` sources after any export). Reading the
        # kit's own declaration tests what the property actually is — a file
        # written inside appears outside — without dictating a layout.
        on_host = deployment["work_root_on_host"]
        container = deployment["container"]
        token = f"e2e-writable-{tag}"
        probe_file = f"{on_host}/.writable_probe"
        if deployment.get("containerized") is False:
            wrote = on(
                f"printf %s {token} > {shlex.quote(inside)}/.writable_probe"
                f" && cat {shlex.quote(probe_file)}",
                transport,
                check=False,
            )
        else:
            wrote = on(
                f"docker exec {shlex.quote(container)} sh -c "
                + shlex.quote(f"printf %s {token} > {shlex.quote(inside)}/.writable_probe")
                + f" && cat {shlex.quote(probe_file)}",
                transport,
                check=False,
            )
        if token not in wrote:
            faults.append(
                f"the work root is not writable from inside {container} at "
                f"{inside!r}: a file written there does not appear on the host at "
                f"{probe_file}. Anything a later stage asks the engine to write — "
                f"a profiler trace above all — will land in the container layer, "
                f"and the write will report success. Output: {wrote.strip()[-400:]}"
            )

        # ---- 2. the probes ---------------------------------------------------
        print("check_deploy_serves: 2/4 diagnostic probes")
        available = set()
        if engine:
            available.add("engine_endpoint_known")
        if fixed.get("context_length"):
            available.add("context_length_known")

        plan, dropped = build_plan(
            probes,
            {
                "router": router,
                "engine": engine or "",
                "model": served or "",
                "ctx": str(fixed.get("context_length") or ""),
            },
            available,
        )
        for name in dropped:
            print(f"check_deploy_serves:   not applicable: {name}")

        # The plan travels inside the command string, so its size is bounded by
        # `getconf ARG_MAX`. Guarded rather than assumed: the failure mode is
        # `OSError: [Errno 7] Argument list too long: 'bash'` raised from
        # `subprocess`, which names neither the plan nor the probe that grew it.
        encoded = json.dumps(plan)
        if len(encoded) > 128 * 1024:
            return [
                f"the resolved probe plan is {len(encoded)} bytes, which will not fit "
                f"in a command line. A probe is carrying data rather than intent — "
                f"move the expansion into probe_runner.py, as `oversize_prompt_tokens` is"
            ]

        plan_path = f"{work_root}/probe_plan.json"
        results_path = f"{work_root}/probe_results.json"
        Path("probe_plan.json").write_text(json.dumps(plan, indent=2))
        # The zone and the node share `/shared_nfs`, which is what makes a plain
        # copy possible; `require_visible_on_node` in `remote.sh` is the check
        # that says so when it stops being true.
        on(
            f"cat > {shlex.quote(plan_path)} <<'E2E_PLAN_EOF'\n" + encoded + "\nE2E_PLAN_EOF",
            transport,
        )
        probe_output = on(
            f"python3 {shlex.quote(str(package / 'assets/check_deploy_serves.validator/probe_runner.py'))} "
            f"--plan {shlex.quote(plan_path)} --out {shlex.quote(results_path)}",
            transport,
            check=False,
        )
        print(probe_output)
        try:
            probe_results = json.loads(on(f"cat {shlex.quote(results_path)}", transport))
        except (NodeError, ValueError) as exc:
            return [f"the probe runner wrote no results: {exc}\n{probe_output[-2000:]}"]

        Path("probe_results.json").write_text(json.dumps(probe_results, indent=2))
        for row in probe_results["probes"]:
            if row["passed"] is False and row["severity"] == "fail":
                faults.append(
                    f"probe {row['name']}: " + "; ".join(row["faults"])
                )
        for name in probe_results["warned"]:
            print(f"check_deploy_serves: WARN probe {name} did not pass (severity warn, not fatal)")

        if faults:
            # The load is minutes of GPU on a shared node and it cannot tell us
            # anything a failed probe has not already: skip it, and say so.
            faults.append("the load was not sent — a deployment that fails a fatal probe has nothing to measure")
            return faults

        # ---- 3. the load -----------------------------------------------------
        load = probes["load"]
        print(
            f"check_deploy_serves: 3/4 load — {load['input_tokens']}/{load['output_tokens']}, "
            f"concurrency {load['concurrency']}, {load['duration_seconds']}s"
        )
        port = router.rsplit(":", 1)[-1].rstrip("/")
        load_out = f"{work_root}/aiperf"
        load_env = {
            # **From the handshake, not from `fixed.node_ip`.** The load must hit
            # the endpoint the probes just passed against; those are the same
            # deployment and there is one right answer. Measured why it matters:
            # the real kit binds `127.0.0.1` by default, so a load aimed at the
            # node's routable address reaches nothing while every probe passed.
            "NODE_IP": router.split("//", 1)[-1].rsplit(":", 1)[0],
            "ROUTER_PORT": port,
            "SERVED": served,
            "MODEL": fixed["model_path"],
            "MODEL_MOUNT": fixed["model_path"],
            "AIPERF_IMAGE": parameters["aiperf_image"],
            "AIPERF_OUT": load_out,
            "SCRIPTS": str(package / "assets/bench"),
            "ISL": load["input_tokens"],
            "OSL": load["output_tokens"],
            "CONCURRENCY": load["concurrency"],
            "DURATION_S": load["duration_seconds"],
            "TAG": tag,
        }
        try:
            load_output = on(
                f"{envvars(load_env)}bash {shlex.quote(str(package / 'assets/bench/aiperf_synthetic.sh'))}",
                transport,
                timeout=int(load["duration_seconds"]) + seconds(parameters, "load_slack_seconds", 900),
            )
        except NodeError as exc:
            return faults + [f"the load failed (rc={exc.returncode}):\n{exc.output[-4000:]}"]
        except subprocess.TimeoutExpired:
            return faults + [
                f"the load did not finish within {load['duration_seconds']}s plus slack; "
                f"a {load['duration_seconds']}s benchmark that overruns is a deployment "
                f"that stopped answering partway through"
            ]
        print(load_output[-3000:])

        csv = f"{load_out}/{tag}/profile_export_aiperf.csv"
        summary_path = f"{work_root}/load_summary.json"
        try:
            on(
                f"python3 {shlex.quote(str(package / 'assets/bench/summarise.py'))} "
                f"{shlex.quote(csv)} {shlex.quote(summary_path)}",
                transport,
            )
            summary = json.loads(on(f"cat {shlex.quote(summary_path)}", transport))
        except (NodeError, ValueError) as exc:
            return faults + [
                f"the load ran but produced no readable summary at {csv}: {exc}. "
                f"A load with no numbers has not shown the deployment serves under load"
            ]
        Path("load_summary.json").write_text(json.dumps(summary, indent=2))
        faults += judge_load(summary, load["accept"], load)

    finally:
        # ---- 4. teardown, on every path -------------------------------------
        # `check=False`: a teardown that fails must be reported, and it must not
        # replace the verdict on the deployment. What it may never do is be
        # skipped — this leaves a GPU behind on a node four other owners share.
        # **A bug in this block leaks on every invocation, not on a bad one.**
        # Worth stating because the causality is not obvious in either
        # direction: the string-arg `TypeError` lived here as well as in
        # bring-up, so `check_deploy_serves` left its processes behind *every
        # time it ran* — the leak was a property of the bug rather than an
        # accident of one unlucky run, and "it failed but it cleaned up" was
        # never true while that bug was live. Two stub routers were found alive
        # on a shared node from a run that had already finished. Anything added
        # to this block gets the same scrutiny as the thing it tears down.
        print("check_deploy_serves: 4/4 teardown")
        # Hand back anything the deployment's container wrote as root, from
        # inside that container — the only context with the privilege. It runs
        # **before** teardown, because a container that is gone cannot chown
        # (`../lib/reclaim.sh` is a no-op then, by design, so the order is a
        # correctness point rather than a safety one).
        #
        # Reading root-owned files works, which is what makes this easy to miss:
        # `copy_out`, the seal and every validator succeed, and the failure lands
        # on the *next* run when the zone's own user cannot clean up. Found by m3
        # on the first real GPU run; CONTRACT.md §5.0.
        try:
            container = locals().get("deployment", {}).get("container")
            if container:
                print(
                    on(
                        f"sh {shlex.quote(str(package / 'assets/lib/reclaim.sh'))} "
                        f"{shlex.quote(container)} {shlex.quote(work_root)}",
                        transport,
                        check=False,
                        timeout=seconds(parameters, "teardown_timeout_seconds", 600),
                    )[-1000:]
                )
        except Exception as exc:
            print(f"check_deploy_serves: reclaim failed: {exc}", file=sys.stderr)
        try:
            print(
                on(
                    f"cd {shlex.quote(str(kit))} && {envvars(overrides)}bash {shlex.quote(teardown)}",
                    transport,
                    check=False,
                    timeout=seconds(parameters, "teardown_timeout_seconds", 600),
                )[-2000:]
            )
        except Exception as exc:  # a teardown that itself hangs must not mask the verdict
            print(f"check_deploy_serves: TEARDOWN FAILED: {exc}", file=sys.stderr)
            print(
                f"check_deploy_serves: containers and ports tagged {tag} may still be "
                f"held on {fixed.get('node')} — check before the next run",
                file=sys.stderr,
            )

    return faults


def main() -> int:
    parameters = zone.args()
    import yaml

    probes = yaml.safe_load((HERE / "probes.yaml").read_text())

    # `args` overrides for the load shape, so `--var deploy_load_seconds=20`
    # makes a wiring run cheap without editing the document that states the
    # standard.
    for key, arg in (
        ("input_tokens", "load_input_tokens"),
        ("output_tokens", "load_output_tokens"),
        ("concurrency", "load_concurrency"),
        ("duration_seconds", "load_seconds"),
    ):
        # `arg_num`, not `if parameters.get(arg)`: an operator who says `0`
        # means 0, and the truthy `"0"` string form would silently work while a
        # genuine yaml `0` was skipped.
        probes["load"][key] = seconds(parameters, arg, int(probes["load"][key]))

    # `assets/lib/remote.sh` reads these three and forwards the whole `E2E_*`
    # block to the far side of an `spur exec` (`remote.sh:84` — the transport
    # carries no environment of its own, measured).
    # `auto` is `remote.sh`'s to resolve and it does (`_transport`, fixed by m2
    # after this body worked around it). The workaround that lived here is gone
    # rather than left in place: it was a second copy of a rule, and a
    # workaround whose removal criterion nobody triggers is how two copies drift.
    transport = {
        "E2E_TRANSPORT": parameters.get("transport", "auto"),
        "E2E_JOBID": parameters.get("jobid", ""),
        "E2E_NODE": parameters.get("node", ""),
        # Where the transport binary lives. Consumed by `on` and never exported.
        "_path_extra": parameters.get("transport_path", "/usr/local/bin:/usr/local/sbin"),
        "_env_extra": parameters.get("transport_env", ""),
    }

    results: dict[str, bool] = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        if content is None:
            results[hid] = False
            print(f"check_deploy_serves: {hid}: no staged content")
            continue
        try:
            faults = check_one(content, parameters, transport, probes)
        except Exception as exc:  # a body that dies must refuse, not vanish
            faults = [f"the check itself failed: {type(exc).__name__}: {exc}"]
        results[hid] = not faults
        for fault in faults:
            print(f"check_deploy_serves: {hid}: FAIL: {fault}")
    zone.write_verdict(results)
    print(f"check_deploy_serves: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
