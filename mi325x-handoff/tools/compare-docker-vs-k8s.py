#!/usr/bin/env python3
"""Static diff: the docker recipe's effective argv vs the k8s recipe's.

The k8s recipe README claims "every engine and kvd flag is identical" between the
two, with `--router-backend` the only difference. That claim is worth checking
rather than trusting -- the same section also carried a fabricated reason for the
router flag itself, which only came apart once someone read the code.

Eyeballing a shell script against a YAML `command:` array does not scale and is
exactly how a claim like that survives. So the docker side is captured by actually
running the launch scripts with a `python3` shim on PATH that records its argv and
exits, which means what gets compared is the argv the engine would really have
received -- including everything env.sh computed and every conditional block the
script took. The k8s side is read straight out of the manifest.

Usage:
  compare-docker-vs-k8s.py --combo disaggregated
  compare-docker-vs-k8s.py --combo disaggregated --docker-env MTP=0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO / "examples/glm5.2_gfx942"

# Values that must differ because the substrate differs, not because the two
# deployments disagree. Flags listed here are reported separately instead of as
# a mismatch, each with the reason it is expected to move.
SUBSTRATE = {
    "--host": "both bind 0.0.0.0; port/addressing is substrate-local",
    "--port": "docker uses host ports, k8s uses fixed container ports",
    "--advertise-host": "docker resolves an env IP, k8s uses the downward API $(POD_IP)",
    "--discovery-backend": "docker runs its own etcd, the operator's backend is kubernetes",
    "--etcd-endpoint": "etcd only exists on the docker side",
    "--model-path": "path of the read-only mount, not a setting",
    "--router-tokenizer-path": "same mount-path difference",
    "--disaggregation-ib-device": "rail name is per-cluster hardware",
    "--infera-kvd-socket": "socket path is substrate-local",
    "--disaggregation-bootstrap-port": "port allocation is substrate-local",
}


def base_env() -> dict[str, str]:
    """A minimal environment for running the launch scripts.

    Deliberately not `os.environ`. env.sh honours anything already exported, so
    inheriting the caller's shell makes the answer depend on whether that shell had
    sourced temp/docker-pd-env.sh -- which reported 15 phantom environment
    differences the first time it happened, all of them the caller's own exports
    showing up as "only docker".
    """
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C",
    }


def capture_docker_argv(script: str, extra_env: dict[str, str]) -> list[str]:
    """Run one launch script with a shim python3 and return the argv it would exec."""
    shim = Path(tempfile.mkdtemp(prefix="argvshim-"))
    dump = shim / "argv.json"
    # The shebang must be the real interpreter's absolute path. `/usr/bin/env
    # python3` would resolve through the PATH this shim is prepended to and find
    # the shim itself.
    (shim / "python3").write_text(
        f"#!{sys.executable}\n"
        "import json,sys\n"
        f"open({str(dump)!r},'w').write(json.dumps(sys.argv[1:]))\n"
    )
    (shim / "python3").chmod(0o755)
    # The scripts kill a previous instance and sleep before launching; neither is
    # wanted here and pkill would match unrelated processes on a shared box.
    # infera-router is stubbed because launch_router.sh refuses to start without it
    # on PATH -- a good check when launching, in the way when only reading argv.
    for stub in ("pkill", "sleep", "infera-router"):
        (shim / stub).write_text("#!/bin/sh\nexit 0\n")
        (shim / stub).chmod(0o755)

    env = base_env()
    env["PATH"] = f"{shim}:{env['PATH']}"
    env.update({
        # env.sh resolves these from hostnames; pin them so the capture does not
        # depend on DNS, and so both legs are named the same way every run.
        "PREFILL_IP": "10.0.0.1", "DECODE_IP": "10.0.0.2",
        "MODEL": "/models/GLM-5.2-FP8",
        "KVD_L3_DIR": "/kvd-l3",
        "LOG_DIR": str(shim), "RESULT_DIR": str(shim),
    })
    env.update(extra_env)

    r = subprocess.run(["bash", str(DOCKER_DIR / script)], env=env,
                       capture_output=True, text=True)
    # The launch scripts background the engine with `nohup ... &`, so the script
    # returns before the shim has written anything. Poll rather than sleep.
    for _ in range(100):
        if dump.exists() and dump.stat().st_size:
            break
        time.sleep(0.05)
    if not dump.exists():
        sys.exit(f"{script}: shim never invoked.\nstdout:{r.stdout}\nstderr:{r.stderr}")
    argv = json.loads(dump.read_text())
    shutil.rmtree(shim, ignore_errors=True)
    return argv


# Prefixes of variables read out of the environment by the engine, its runtime, or
# the KV transport -- as opposed to the recipe's own configuration knobs.
RUNTIME_ENV_PREFIXES = (
    "SGLANG_", "HSA_", "HIP_", "CUDA_", "SAFETENSORS_", "MC_", "MOONCAKE_",
    "NCCL_", "INFERA_", "TORCH_", "PYTORCH_", "ROCM_", "GPU_", "AMD_",
)


def capture_docker_env(script: str, extra_env: dict[str, str]) -> dict[str, str]:
    """The engine-affecting environment the launch script exports.

    Comparing only `command:` would miss this entirely, and several of these change
    performance rather than plumbing (the DSA prefill path, aiter, scratch reclaim).
    Captured by dumping the shim's own os.environ, so anything env.sh or the script
    exported is included, then filtered to the variables an engine reads.
    """
    shim = Path(tempfile.mkdtemp(prefix="envshim-"))
    dump = shim / "env.json"
    (shim / "python3").write_text(
        f"#!{sys.executable}\n"
        "import json,os\n"
        f"open({str(dump)!r},'w').write(json.dumps(dict(os.environ)))\n"
    )
    (shim / "python3").chmod(0o755)
    for stub in ("pkill", "sleep", "infera-router"):
        (shim / stub).write_text("#!/bin/sh\nexit 0\n")
        (shim / stub).chmod(0o755)

    env = base_env()
    env["PATH"] = f"{shim}:{env['PATH']}"
    env.update({
        "PREFILL_IP": "10.0.0.1", "DECODE_IP": "10.0.0.2",
        "MODEL": "/models/GLM-5.2-FP8", "KVD_L3_DIR": "/kvd-l3",
        "LOG_DIR": str(shim), "RESULT_DIR": str(shim),
    })
    env.update(extra_env)

    subprocess.run(["bash", str(DOCKER_DIR / script)], env=env,
                   capture_output=True, text=True)
    for _ in range(100):
        if dump.exists() and dump.stat().st_size:
            break
        time.sleep(0.05)
    if not dump.exists():
        shutil.rmtree(shim, ignore_errors=True)
        return {}
    after = json.loads(dump.read_text())
    shutil.rmtree(shim, ignore_errors=True)
    # Filter to variables the engine/runtime actually reads. env.sh also exports its
    # own knobs (TP, CHUNK, MTP_STEPS, KVD_*, ISL...), but those are inputs to the
    # launch script rather than things any process reads from the environment, and
    # whatever they selected already shows up in the argv comparison. Including them
    # would report ~40 differences that are all the same fact twice.
    return {k: v for k, v in after.items() if k.startswith(RUNTIME_ENV_PREFIXES)}


def k8s_env(manifest: Path) -> dict[str, dict[str, str]]:
    """Per-service `env:` entries from the manifest's flow-style maps."""
    text = manifest.read_text()
    out: dict[str, dict[str, str]] = {}
    svc = None
    for line in text.splitlines():
        m = re.match(r"^    (\w+):\s*$", line)
        if m:
            svc = m.group(1)
            continue
        m = re.match(r"^\s*- \{name: (\w+), value: \"?([^\"}]*)\"?\}", line)
        if m and svc:
            out.setdefault(svc, {})[m.group(1)] = m.group(2)
    return out


def k8s_commands(manifest: Path) -> dict[str, list[str]]:
    """Extract each service's container command from the recipe manifest.

    Hand-parsed rather than via a YAML lib: the recipe manifests carry the
    placeholders (`<NODE>`) and flow-style maps that make them intentionally
    un-loadable until rendered, and the `command:` arrays are the only part needed.
    """
    text = manifest.read_text()
    out: dict[str, list[str]] = {}
    # Service blocks are 4-space keys under `services:`; commands are the JSON-ish
    # arrays that follow `command:`.
    svc = None
    for m in re.finditer(
        r"^    (\w+):\s*$|^\s*command: (\[.*?\])\s*$",
        text, flags=re.MULTILINE | re.DOTALL,
    ):
        if m.group(1):
            svc = m.group(1)
        elif svc:
            raw = m.group(2)
            # The arrays span lines and use single quotes for JSON payloads.
            items = re.findall(r'"((?:[^"\\]|\\.)*)"|\'([^\']*)\'', raw)
            argv = [a if a else b for a, b in items]
            out.setdefault(svc, argv)
    return out


def normalize(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """argv -> ({flag: value}, [bare flags]). Values that are themselves JSON are
    canonicalised so key order cannot show up as a difference."""
    flags: dict[str, str] = {}
    bare: list[str] = []
    i = 0
    # Drop the module invocation itself (-m infera.engine.sglang / python3 -m ...).
    while i < len(argv) and not argv[i].startswith("--"):
        i += 1
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if nxt is None or nxt.startswith("--"):
            bare.append(tok)
            i += 1
        else:
            v = nxt
            if v.lstrip().startswith("{"):
                try:
                    v = json.dumps(json.loads(v), sort_keys=True)
                except Exception:  # noqa: BLE001 - not JSON after all; compare raw
                    pass
            flags[tok] = v
            i += 2
    return flags, bare


def report(name: str, d_argv: list[str], k_argv: list[str]) -> int:
    df, db = normalize(d_argv)
    kf, kb = normalize(k_argv)
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")

    real: list[str] = []
    for flag in sorted(set(df) | set(kf)):
        dv, kv = df.get(flag), kf.get(flag)
        if dv == kv:
            continue
        if flag in SUBSTRATE:
            print(f"  [substrate] {flag}\n              docker={dv!r} k8s={kv!r}\n"
                  f"              {SUBSTRATE[flag]}")
        else:
            real.append(flag)
            tag = "MISSING in k8s" if kv is None else (
                "extra in k8s" if dv is None else "VALUE DIFFERS")
            print(f"  [DIFF] {flag}  ({tag})\n         docker={dv!r}\n         k8s   ={kv!r}")

    only_d, only_k = sorted(set(db) - set(kb)), sorted(set(kb) - set(db))
    if only_d:
        print(f"  [DIFF] bare flags only in docker: {only_d}")
    if only_k:
        print(f"  [DIFF] bare flags only in k8s:   {only_k}")

    n = len(real) + len(only_d) + len(only_k)
    same = len(set(df) & set(kf)) - len(real) - len([f for f in SUBSTRATE if f in df and f in kf and df.get(f) != kf.get(f)])
    print(f"\n  {len(set(db) & set(kb))} bare flags identical, ~{same} valued flags identical, "
          f"{n} real difference(s)")
    return n


# Environment variables that differ for substrate reasons rather than tuning ones.
ENV_SUBSTRATE = {
    "SGLANG_HOST_IP", "HOST_IP", "POD_IP", "POD_NAME", "POD_NAMESPACE",
    "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",  # device plugin owns these in k8s
    "MC_GID_INDEX",                                 # per-cluster RoCE hardware
}


def report_env(name: str, denv: dict[str, str], kenv: dict[str, str]) -> int:
    print(f"\n  --- environment ---")
    real = 0
    for k in sorted(set(denv) | set(kenv)):
        dv, kv = denv.get(k), kenv.get(k)
        if dv == kv or k in ENV_SUBSTRATE:
            continue
        real += 1
        tag = "only docker" if kv is None else ("only k8s" if dv is None else "differs")
        print(f"  [ENV {tag}] {k}: docker={dv!r} k8s={kv!r}")
    if real == 0:
        print("  (no non-substrate environment differences)")
    return real


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--combo", default="disaggregated",
                   choices=("disaggregated", "disaggregated-kvd"))
    p.add_argument("--docker-env", action="append", default=[],
                   help="VAR=VAL overrides for the docker side (repeatable)")
    p.add_argument("--k8s-manifest", default="",
                   help="compare against this manifest instead of the shipped recipe. "
                        "Point it at a render-deploy.py output to include what "
                        "rendering injects for a specific cluster -- on tw041/tw044 "
                        "that is the single-rail Mooncake pin, which the shipped "
                        "manifest does not carry and the docker example has no "
                        "equivalent of.")
    args = p.parse_args()

    extra = dict(kv.split("=", 1) for kv in args.docker_env)
    if args.combo.endswith("-kvd"):
        extra.setdefault("KVD", "1")

    manifest = (Path(args.k8s_manifest) if args.k8s_manifest
                else REPO / f"examples/recipes/glm5.2-fp8-gfx942/{args.combo}/deploy.yaml")
    kcmd = k8s_commands(manifest)
    kenv = k8s_env(manifest)
    print(f"docker: {DOCKER_DIR}   (env overrides: {extra or 'none'})")
    print(f"k8s   : {manifest}   (services: {sorted(kcmd)})")

    total = 0
    for name, script, svc in (
        ("PREFILL leg", "launch/launch_prefill.sh", "prefill"),
        ("DECODE leg", "launch/launch_decode.sh", "decode"),
        ("ROUTER", "launch/launch_router.sh", "server"),
    ):
        if svc not in kcmd:
            print(f"\n(skipping {name}: no `{svc}` service in the manifest)")
            continue
        total += report(name, capture_docker_argv(script, extra), kcmd[svc])
        total += report_env(name, capture_docker_env(script, extra), kenv.get(svc, {}))

    print(f"\n{'=' * 78}\nTOTAL real differences: {total}\n{'=' * 78}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
