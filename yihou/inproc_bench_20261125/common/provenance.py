"""Write the provenance files of one bench point (stdlib only; called inside the container).

Files: command.json (argv, redacted), env.json (process env, redacted), versions.json (python, torch,
engine version + source commit when found, ROCm), kit_hashes.sha256 (sha256 of every kit file used).
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact import redact_argv, redact_env  # noqa: E402


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # provenance must never fail a run
        return f"<unavailable: {type(e).__name__}>"


def engine_versions(engine: str) -> dict:
    v = {"python": platform.python_version(), "hostname": platform.node()}
    try:
        import torch
        v["torch"] = torch.__version__
        v["hip"] = getattr(torch.version, "hip", None)
        v["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            v["device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        v["torch"] = f"<unavailable: {e}>"
    if engine == "sglang":
        v["sglang"] = _run([sys.executable, "-m", "pip", "show", "sglang"]).split("\n")[1:2]
        v["sglang_git"] = _run(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"])
    elif engine == "vllm":
        try:
            import vllm
            v["vllm"] = vllm.__version__
        except Exception as e:
            v["vllm"] = f"<unavailable: {e}>"
    try:
        import aiter  # noqa: F401
        v["aiter_file"] = aiter.__file__
    except Exception:
        pass
    v["rocm_version_file"] = _run(["cat", "/opt/rocm/.info/version"])
    return v


def kit_hashes(kit_root: Path) -> list:
    rows = []
    for p in sorted(kit_root.rglob("*")):
        if p.is_file() and p.suffix in {".py", ".sh", ".args", ".env", ".json", ".yaml"} and "__pycache__" not in p.parts:
            rows.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(kit_root)}")
    return rows


def write(out_dir: str, engine: str, argv=None, kit_root=None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "command.json").write_text(json.dumps({"argv": redact_argv(argv or sys.argv)}, indent=2))
    (out / "env.json").write_text(json.dumps(redact_env(dict(os.environ)), indent=2))
    (out / "versions.json").write_text(json.dumps(engine_versions(engine), indent=2, default=str))
    root = Path(kit_root) if kit_root else Path(__file__).resolve().parents[1]
    (out / "kit_hashes.sha256").write_text("\n".join(kit_hashes(root)) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--engine", choices=["sglang", "vllm"], required=True)
    a, rest = ap.parse_known_args()
    write(a.out_dir, a.engine, argv=rest)
