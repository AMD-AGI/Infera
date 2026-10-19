"""What a validator body needs: its inputs, its args, the content to judge, and
where to put the verdict.

A validator runs with its own fresh zone as the working directory. The system
puts three files there: `inputs.json` (the handoff ids to judge), `args.json`
(the validator's declared `args`) and `materials.json` (each id's staged
content directory). The body answers with `verdict.json`: one boolean per id.
"""

import json
from pathlib import Path


def inputs() -> list[str]:
    return list(json.loads(Path("inputs.json").read_text()))


def args() -> dict:
    path = Path("args.json")
    return json.loads(path.read_text()) if path.is_file() else {}


def content(hid: str) -> Path | None:
    path = Path("materials.json")
    staged = json.loads(path.read_text()).get(hid) if path.is_file() else None
    return Path(staged) if staged and Path(staged).is_dir() else None


def document(hid: str) -> dict | None:
    """`items/text.json` of a structured_text handoff, or None if unreadable."""
    root = content(hid)
    if root is None or not (root / "items" / "text.json").is_file():
        return None
    try:
        data = json.loads((root / "items" / "text.json").read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def write_verdict(results: dict[str, bool]) -> None:
    Path("verdict.json").write_text(json.dumps(results, indent=2))
