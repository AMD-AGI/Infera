#!/usr/bin/env python3
"""(v3x, task g4tune xnode 2026-10-21: --decode-node = host of the decode container.)
Audit the live same-node 1P1D against the config: /get_server_info + docker inspect env.
v3 (v0521cap2): + per-role {ROLE}_EXPERT8 / {ROLE}_MAX_TOTAL_TOKENS (synthetic 8-expert leg).
v2 (v0521dbg): + hicache_ratio, enable_dsa_fused_indexer, and the E1/E2 env knobs
(SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER, SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB).

Usage: audit_live.yihou.py --config CFG --node NODE --prefill URL --decode URL --prefix P --out DIR
Prints one line per check (PASS / FAIL / ABSENT) and exits 1 on any FAIL.
ABSENT means the key is not in this sglang's server_info -- read it, do not ignore it.
"""
import argparse
import json
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path


def config_env(path: str, role: str) -> dict:
    out = subprocess.run(
        ["bash", "-c", f"role={role}; set -a; source {shlex.quote(path)} >/dev/null; env -0"],
        check=True, capture_output=True,
    ).stdout.decode()
    return dict(item.split("=", 1) for item in out.split("\0") if "=" in item)


def expected(env: dict, role: str) -> tuple[dict, dict]:
    p = role.upper()
    info = {
        "tp_size": int(env[f"{p}_TP"]), "dp_size": int(env[f"{p}_DP"]),
        "ep_size": int(env[f"{p}_EP"]), "enable_dp_attention": env[f"{p}_DPA"] == "1",
        "mem_fraction_static": float(env[f"{p}_MEM_FRACTION"]),
        "dsa_prefill_backend": env["DSA_PREFILL_BACKEND"],
        "dsa_decode_backend": env["DSA_DECODE_BACKEND"],
        "kv_cache_dtype": env["KV_CACHE_DTYPE"],
        "json_model_override_args": env["JSON_MODEL_OVERRIDE_ARGS"],
        "enable_hierarchical_cache": env[f"{p}_HICACHE"] == "1",
        "disable_shared_experts_fusion": env.get(f"{p}_EXPERT8") == "1",
        "max_total_tokens": int(env[f"{p}_MAX_TOTAL_TOKENS"]) if env.get(f"{p}_MAX_TOTAL_TOKENS") else None,
        "enable_aiter_allreduce_fusion": env.get("AITER_ALLREDUCE_FUSION", "1") == "1",
        "disaggregation_mode": role,
        "disaggregation_transfer_backend": env["KV_P2P_TRANSFER"],
        "served_model_name": env["SERVED_MODEL"],
        "model_path": env["MODEL"],
    }
    # None = flag absent (v0521 auto: ON where supported); False = --no-enable-dsa-fused-indexer.
    info["enable_dsa_fused_indexer"] = False if env.get("DSA_FUSED_INDEXER") == "off" else None
    if role == "prefill":
        info.update(hicache_size=int(env["PREFILL_HICACHE_SIZE"]),
                    hicache_ratio=float(env["PREFILL_HICACHE_RATIO"]),
                    hicache_write_policy=env["PREFILL_HICACHE_WRITE_POLICY"],
                    hicache_io_backend=env["PREFILL_HICACHE_IO_BACKEND"],
                    hicache_mem_layout=env["PREFILL_HICACHE_MEM_LAYOUT"],
                    disable_custom_all_reduce=False, speculative_algorithm=None)
    else:
        # g4tune 2026-10-20: expected value follows the config (config.yihou.g4new.sh drops the flag).
        info.update(disable_custom_all_reduce="--disable-custom-all-reduce" in env.get("DECODE_EXTRA_ARGS", ""),
                    speculative_algorithm="EAGLE",
                    hicache_ratio=None, hicache_size=0,
                    speculative_num_steps=int(env["DECODE_SPEC_STEPS"]),
                    speculative_eagle_topk=int(env["DECODE_SPEC_TOPK"]),
                    speculative_num_draft_tokens=int(env["DECODE_SPEC_DRAFT_TOKENS"]))
    # Both-leg agent (DEBUG_AGENT=1, engine.sh) or decode-only agent (config.yihou.seedguard.sh puts
    # it in DECODE_EXTRA_ENV when DECODE_DEBUG_AGENT=1).
    agent = env.get("DEBUG_AGENT") == "1" or (
        role == "decode" and "HSA_TOOLS_LIB=" in env.get("DECODE_EXTRA_ENV", ""))
    cenv = {
        "MC_TE_FILTERS": env["MC_TE_FILTERS"],
        "MOONCAKE_DISABLE_HIP_DMABUF": env["MOONCAKE_DISABLE_HIP_DMABUF"],
        "SGLANG_OPT_USE_TOPK_V2": env["SGLANG_OPT_USE_TOPK_V2"],
        "HSA_NO_SCRATCH_RECLAIM": env[f"{p}_HSA_NO_SCRATCH_RECLAIM"],
        "MC_DISABLE_HIP": None,  # must be ABSENT: it would demote hip below rdma
        "SGLANG_SIMULATE_ACC_LEN": (env.get("DECODE_SIMULATE_ACC_LEN") or None)
        if role == "decode" else None,
        "SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER":
            "1" if env.get("AITER_FUSED_FP8_INDEXER_WRITER") == "off" else None,
        # ROCm debug agent, lean: exact values engine.sh emits (no --precise-memory / --all).
        "HSA_TOOLS_LIB": (env.get("DEBUG_AGENT_LIB") or "/opt/rocm/lib/librocm-debug-agent.so.2")
            if agent else None,
        "HSA_ENABLE_DEBUG": "1" if agent else None,
        "HSA_ENABLE_IPC_MODE_LEGACY": "1" if env.get("IPC_MODE_LEGACY") == "1" else None,
        # lean: stderr (no --output), no --precise-memory / --all
        "ROCM_DEBUG_AGENT_OPTIONS": (env.get("DEBUG_AGENT_OPTIONS") or "--disable-linux-signals") if agent else None,
        # config.yihou.seedguard.sh (R029): guard mode, decode only
        "YIHOU_DSA_SEED_GUARD": env.get("YIHOU_DSA_SEED_GUARD") if role == "decode" and env.get("YIHOU_DSA_SEED_GUARD") else None,
        "SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB":
            (env.get("HICACHE_HOST_REGISTER_CHUNK_GB") or None) if role == "prefill" else None,
    }
    return info, cenv


def same(want, got) -> bool:
    if isinstance(want, float):
        return got is not None and abs(float(got) - want) < 1e-9
    if isinstance(want, str) and want.startswith("{"):
        try:
            got_obj = json.loads(got) if isinstance(got, str) else got
            return got_obj == json.loads(want)
        except (TypeError, ValueError):
            return False
    if isinstance(got, list) and len(got) == 1:
        got = got[0]
    return want == got


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--config", "--node", "--decode-node", "--prefill", "--decode", "--prefix", "--out"):
        ap.add_argument(name, required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for role, url in (("prefill", a.prefill), ("decode", a.decode)):
        env = config_env(a.config, role)
        want_info, want_env = expected(env, role)
        info = json.load(urllib.request.urlopen(url + "/get_server_info", timeout=30))
        (out / f"{role}-server-info.json").write_text(json.dumps(info, indent=2, sort_keys=True))
        inspect = json.loads(subprocess.run(
            ["ssh", a.node if role == "prefill" else a.decode_node, f"docker inspect {a.prefix}-{role}-0"],
            check=True, capture_output=True, text=True).stdout)[0]
        (out / f"{role}-container.json").write_text(json.dumps(inspect, indent=2))
        cenv = dict(e.split("=", 1) for e in inspect["Config"]["Env"] if "=" in e)
        print(f"== {role} {url}  image={inspect.get('Image')}")
        for key, want in want_info.items():
            # sglang >= 0.5.21 resolves `--dp-size N --enable-dp-attention` into attn_dp_size=N,
            # dp_size=1 (same handling as agentx_env.py); compare attn_dp_size then.
            if key == "dp_size" and want_info.get("enable_dp_attention") and "attn_dp_size" in info:
                ok = same(want, info["attn_dp_size"])
                failed += not ok
                print(f"  {'PASS' if ok else 'FAIL'}   attn_dp_size={info['attn_dp_size']!r} (want {want!r}; dp_size={info.get('dp_size')!r})")
                continue
            if key not in info:
                print(f"  ABSENT {key} (want {want!r})")
                continue
            ok = same(want, info[key])
            failed += not ok
            print(f"  {'PASS' if ok else 'FAIL'}   {key}={info[key]!r} (want {want!r})")
        pid_mode = (inspect.get("HostConfig") or {}).get("PidMode") or ""
        want_pm = "host" if env.get("ENGINE_PID_HOST") == "1" else ""
        ok = pid_mode == want_pm
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}   HostConfig.PidMode={pid_mode!r} (want {want_pm!r})")
        # R045 NUMA placement knobs (engine.sh): empty knob = no flag = docker default "".
        p = role.upper()
        for hc_key, knob in (("CpusetCpus", f"{p}_CPUSET_CPUS"), ("CpusetMems", f"{p}_CPUSET_MEMS")):
            got = (inspect.get("HostConfig") or {}).get(hc_key) or ""
            want = env.get(knob, "")
            ok = got == want
            failed += not ok
            print(f"  {'PASS' if ok else 'FAIL'}   HostConfig.{hc_key}={got!r} (want {want!r})")
        if env.get("ENGINE_SET_CPU_AFFINITY"):   # unset knob = image default, not checked
            got, want = cenv.get("SGLANG_SET_CPU_AFFINITY"), env["ENGINE_SET_CPU_AFFINITY"]
            ok = got == want
            failed += not ok
            print(f"  {'PASS' if ok else 'FAIL'}   env SGLANG_SET_CPU_AFFINITY={got!r} (want {want!r})")
        for key, want in want_env.items():
            got = cenv.get(key)
            ok = got == want
            failed += not ok
            print(f"  {'PASS' if ok else 'FAIL'}   env {key}={got!r} (want {want!r})")
        # Generic: every KEY=VALUE the config puts in <ROLE>_EXTRA_ENV must be in the container
        # exactly (overlays such as seedguard/xferpar add role env this way; R029/R033).
        for item in env.get(f"{role.upper()}_EXTRA_ENV", "").split():
            if "=" not in item or item.split("=", 1)[0] in want_env:
                continue
            key, want = item.split("=", 1)
            got = cenv.get(key)
            ok = got == want
            failed += not ok
            print(f"  {'PASS' if ok else 'FAIL'}   extra-env {key}={got!r} (want {want!r})")
        # Source-patch mounts: every YIHOU_MOUNTS row must be bind-mounted read-only at its target.
        live = {m.get("Destination"): m for m in inspect.get("Mounts", [])}
        for row in Path(env["YIHOU_MOUNTS"]).read_text().splitlines():
            if not row.strip():
                continue
            src, dst = row.split("\t")[:2]
            m = live.get(dst)
            ok = m is not None and m.get("Source") == src and m.get("RW") is False
            failed += not ok
            got = None if m is None else f"{m.get('Source')} rw={m.get('RW')}"
            print(f"  {'PASS' if ok else 'FAIL'}   mount {dst} <- {got} (want {src} ro)")
        # R050 role-only rows ({ROLE}_MOUNTS_ADD): present ro on this leg, ABSENT on the other leg.
        other = "DECODE" if role == "prefill" else "PREFILL"
        for knob, want_present in ((f"{p}_MOUNTS_ADD", True), (f"{other}_MOUNTS_ADD", False)):
            if not env.get(knob):
                continue
            for row in Path(env[knob]).read_text().splitlines():
                if not row.strip():
                    continue
                src, dst = row.split("\t")[:2]
                m = live.get(dst)
                got = None if m is None else f"{m.get('Source')} rw={m.get('RW')}"
                if want_present:
                    ok = m is not None and m.get("Source") == src and m.get("RW") is False
                    print(f"  {'PASS' if ok else 'FAIL'}   {knob} mount {dst} <- {got} (want {src} ro)")
                else:
                    ok = m is None or m.get("Source") != src
                    print(f"  {'PASS' if ok else 'FAIL'}   {knob} mount {dst} <- {got} (want NOT {src} on {role})")
                failed += not ok
    print("AUDIT", "PASS" if not failed else f"FAIL ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
