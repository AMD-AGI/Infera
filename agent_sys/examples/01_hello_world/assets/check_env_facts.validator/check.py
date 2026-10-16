"""check_env_facts: every fact key is present with a value of the right type."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import vio  # noqa: E402 - the path insert above makes it importable

TYPES = {
    "hostname": str,
    "os": str,
    "cpu_count": int,
    "gpu_render_nodes": int,
    "in_container": bool,
    "claude_cli": bool,
}


def ok(facts: dict | None) -> bool:
    if facts is None:
        return False
    for key, kind in TYPES.items():
        value = facts.get(key)
        # bool is a subclass of int, so an integer fact must not be a bool.
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            print(f"  {key}: {value!r} is not {kind.__name__}")
            return False
    return True


def main() -> int:
    results = {hid: ok(vio.document(hid)) for hid in vio.inputs()}
    vio.write_verdict(results)
    print(f"check_env_facts: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
