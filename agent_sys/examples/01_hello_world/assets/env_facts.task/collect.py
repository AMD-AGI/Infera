"""Measure this machine and write the `env_facts` handoff.

The output directory arrives as `AGENT_SYS_OUTPUT_ENV_FACTS`: it already exists
and is the only place this task may write. A `structured_text` handoff is
`items/text.json` plus a README with `## Purpose` and `## Schema`.
"""

import glob
import json
import os
import platform
import shutil
import socket
from pathlib import Path

README = """# env_facts

## Purpose

Facts about the machine this run executed on, measured by a script, so that an
AI agent can check them and a validator can check the agent.

## Schema

`items/text.json` is one JSON object with these keys:

| key | type | how it is measured |
|---|---|---|
| `hostname` | string | `socket.gethostname()` |
| `os` | string | `platform.system()` and `platform.release()` |
| `cpu_count` | integer | `os.cpu_count()` |
| `gpu_render_nodes` | integer | number of `/dev/dri/renderD*` |
| `in_container` | boolean | `/.dockerenv` exists |
| `claude_cli` | boolean | a `claude` executable is on `PATH` |
"""


def measure() -> dict:
    return {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
        "gpu_render_nodes": len(glob.glob("/dev/dri/renderD*")),
        "in_container": os.path.exists("/.dockerenv"),
        "claude_cli": shutil.which("claude") is not None,
    }


def main() -> None:
    out = Path(os.environ["AGENT_SYS_OUTPUT_ENV_FACTS"])
    (out / "items").mkdir(parents=True, exist_ok=True)
    facts = measure()
    (out / "items" / "text.json").write_text(json.dumps(facts, indent=2) + "\n")
    (out / "README.md").write_text(README)
    print(json.dumps(facts))


if __name__ == "__main__":
    main()
