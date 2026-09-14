#!/usr/bin/env python3
"""Render the summary into a report. Never runs: summary is sealed INVALID."""

import json
import os
import sys
from pathlib import Path


def main() -> int:
    src = Path(os.environ["AGENT_SYS_INPUT_SUMMARY"])
    content = (src / "items" / "text.json").read_text(encoding="utf-8")
    Path("report.json").write_text(
        json.dumps({"summary": json.loads(content), "rendered": True}, indent=2),
        encoding="utf-8",
    )
    print("consume: rendered report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
