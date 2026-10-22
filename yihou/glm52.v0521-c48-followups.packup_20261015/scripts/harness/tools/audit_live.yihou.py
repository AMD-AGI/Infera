#!/usr/bin/env python3
"""Audit the live same-node 1P1D against the config: /get_server_info + docker inspect env.

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
        "disable_shared_experts_fusion": False,
        "max_total_tokens": None,
        "enable_aiter_allreduce_fusion": env.get("AITER_ALLREDUCE_FUSION", "1") == "1",
        "disaggregation_mode": role,
        "disaggregation_transfer_backend": env["KV_P2P_TRANSFER"],
        "served_model_name": env["SERVED_MODEL"],
        "model_path": env["MODEL"],
    }
    if role == "prefill":
        info.update(hicache_size=int(env["PREFILL_HICACHE_SIZE"]),
                    hicache_write_policy=env["PREFILL_HICACHE_WRITE_POLICY"],
                    hicache_io_backend=env["PREFILL_HICACHE_IO_BACKEND"],
                    hicache_mem_layout=env["PREFILL_HICACHE_MEM_LAYOUT"],
                    disable_custom_all_reduce=False, speculative_algorithm=None)
    else:
        info.update(disable_custom_all_reduce=True, speculative_algorithm="EAGLE",
                    speculative_num_steps=int(env["DECODE_SPEC_STEPS"]),
                    speculative_eagle_topk=int(env["DECODE_SPEC_TOPK"]),
                    speculative_num_draft_tokens=int(env["DECODE_SPEC_DRAFT_TOKENS"]))
    cenv = {
        "MC_TE_FILTERS": env["MC_TE_FILTERS"],
        "MOONCAKE_DISABLE_HIP_DMABUF": env["MOONCAKE_DISABLE_HIP_DMABUF"],
        "SGLANG_OPT_USE_TOPK_V2": env["SGLANG_OPT_USE_TOPK_V2"],
        "HSA_NO_SCRATCH_RECLAIM": env[f"{p}_HSA_NO_SCRATCH_RECLAIM"],
        "MC_DISABLE_HIP": None,  # must be ABSENT: it would demote hip below rdma
        "SGLANG_SIMULATE_ACC_LEN": (env.get("DECODE_SIMULATE_ACC_LEN") or None)
        if role == "decode" else None,
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
    for name in ("--config", "--node", "--prefill", "--decode", "--prefix", "--out"):
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
            ["ssh", a.node, f"docker inspect {a.prefix}-{role}-0"],
            check=True, capture_output=True, text=True).stdout)[0]
        (out / f"{role}-container.json").write_text(json.dumps(inspect, indent=2))
        cenv = dict(e.split("=", 1) for e in inspect["Config"]["Env"] if "=" in e)
        print(f"== {role} {url}  image={inspect.get('Image')}")
        for key, want in want_info.items():
            if key not in info:
                print(f"  ABSENT {key} (want {want!r})")
                continue
            ok = same(want, info[key])
            failed += not ok
            print(f"  {'PASS' if ok else 'FAIL'}   {key}={info[key]!r} (want {want!r})")
        for key, want in want_env.items():
            got = cenv.get(key)
            ok = got == want
            failed += not ok
            print(f"  {'PASS' if ok else 'FAIL'}   env {key}={got!r} (want {want!r})")
    print("AUDIT", "PASS" if not failed else f"FAIL ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
