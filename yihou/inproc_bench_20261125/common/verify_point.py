"""Gate one bench point. Exit 0 = every gate passed; 1 = a gate failed; 2 = unreadable point.

Gates (all tools): launch_status.exit_code == 0, result.json present and complete == true,
provenance present (command.json env.json versions.json kit_hashes.sha256), no unredacted secret in the
provenance files. Tool gates:
  sglang step_bench   : decode_step_s.n == output_len - 1; every rank replayed the decode CUDA graph on every
                        timed step; tpot identity decode_tok_s * tpot_ms/1000 == C
  vllm engine_step    : steady_steps >= --min-steady (default 8); output_tokens == C * OSL
  vllm forward_bench  : every case sample_spread_pct <= --max-spread (default 0.5) [warning-only unless --strict]
  infer throughput    : complete == true
Usage: verify_point.py <iteration dir> [--min-steady N] [--max-spread P] [--strict]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact import _TOKEN_VALUE_RES  # noqa: E402

PROV = ("command.json", "env.json", "versions.json", "kit_hashes.sha256")


def check(d: Path, min_steady=8, max_spread=0.5, strict=False):
    fails, warns = [], []
    try:
        res = json.loads((d / "result.json").read_text())
    except Exception as e:
        return None, [f"result.json unreadable: {e}"], []
    try:
        ls = json.loads((d / "launch_status.json").read_text())
        if ls.get("exit_code") != 0:
            fails.append(f"exit_code={ls.get('exit_code')}")
    except Exception:
        fails.append("launch_status.json missing (not launched via run_point.sh?)")
    if res.get("complete") is not True:
        fails.append("complete != true")
    for f in PROV:
        if not (d / f).exists():
            fails.append(f"missing {f}")
        else:
            txt = (d / f).read_text(errors="replace")
            if any(r.search(txt) for r in _TOKEN_VALUE_RES):
                fails.append(f"token-shaped secret in {f}")
    tool = res.get("tool") or res.get("mode")
    if tool == "step_bench":
        n = res.get("decode_step_s", {}).get("n")
        if n != res.get("output_len", 0) - 1:
            fails.append(f"decode steps {n} != OSL-1")
        reps = res.get("decode_graph_replays_per_rank") or []
        if not reps or all(r is None for r in reps):
            (fails if strict else warns).append("decode CUDA-graph use not verified (no replay counter in result)")
        if any(r is not None and r < n for r in reps):
            fails.append(f"decode graph replays {reps} < decode steps {n} (eager fallback)")
        if res.get("tpot_ms") and res.get("decode_tok_s"):
            c = res["decode_tok_s"] * res["tpot_ms"] / 1000.0
            if abs(c - res["batch_size_global"]) > 1e-6 * res["batch_size_global"]:
                fails.append("identity decode_tok_s * tpot != C")
    elif tool == "engine_step_bench":
        if (res.get("steady_steps") or 0) < min_steady:
            fails.append(f"steady_steps {res.get('steady_steps')} < {min_steady}")
        if res.get("output_tokens") != res.get("batch_size", 0) * res.get("output_len", 0):
            fails.append("output_tokens != C*OSL")
    elif tool == "forward_bench":
        for c in res.get("cases", []):
            sp = c.get("sample_spread_pct")
            if sp is not None and sp > max_spread:
                (fails if strict else warns).append(f"{c['case']}: sample spread {sp:.2f}% > {max_spread}%")
    return res, fails, warns


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("point")
    ap.add_argument("--min-steady", type=int, default=8)
    ap.add_argument("--max-spread", type=float, default=0.5)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    res, fails, warns = check(Path(a.point), a.min_steady, a.max_spread, a.strict)
    for w in warns:
        print("WARN", w)
    for f in fails:
        print("FAIL", f)
    if res is None:
        return 2
    print("PASS" if not fails else "FAILED", a.point)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
