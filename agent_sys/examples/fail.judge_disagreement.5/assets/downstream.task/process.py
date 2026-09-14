#!/usr/bin/env python3
"""Process the merged review. Never runs: review is sealed INVALID."""

import json
import os
import sys
from pathlib import Path


def main() -> int:
    src = Path(os.environ["AGENT_SYS_INPUT_REVIEW"])
    data = json.loads((src / "items" / "text.json").read_text(encoding="utf-8"))
    Path("result.json").write_text(
        json.dumps({"processed": True, "totals": data["totals"]}, indent=2),
        encoding="utf-8",
    )
    print("downstream: processed review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
