#!/usr/bin/env python3
"""Launch the chain with the repo on `sys.path` but NOT in the environment.

WHY THIS EXISTS — two requirements that a `PYTHONPATH=` prefix cannot satisfy
at once.

1. **The run process must use one tree.** `agent_sys/cli/main.py:31,48` imports
   its siblings by BARE name (`from cli import …`, `from env_mgr import …`).
   Measured on this host, with no PYTHONPATH:

       agent_sys.cli.main -> /home/yihou/dev/git.16-19/infera/agent_sys/cli/main.py
       env_mgr            -> /home/yihou/dev/git.16-19/infera.aiopt.all/agent_sys/env_mgr/…
       cli                -> /home/yihou/dev/git.16-19/infera.aiopt.all/agent_sys/cli/…

   So `-m agent_sys.cli.main` alone loads OUR main.py and the OTHER tree's
   env_mgr — mixed, silently. `run_with_long_stall.py:111` inserts the *cwd*
   (repo root), which fixes `agent_sys.*` and not the bare names.
   `<repo>/agent_sys` has to be on `sys.path`.

2. **`PYTHONPATH` must NOT be in the live environment**, because
   `env_mgr/harness.py:107` builds a validator's environment as

       str(key): str(live.get(str(key), value))

   over the settings file's `env` block, with `_RESERVED = (CLAUDE_CONFIG_DIR,
   CLAUDE_CODE_TMPDIR, TMPDIR, PATH)` — and **PYTHONPATH is not reserved.** So a
   live `PYTHONPATH` OVERRIDES the settings file's, which is the entire
   mechanism m1's jsonschema fix depends on. A `PYTHONPATH=… python3 …` prefix
   would silently reinstate
   `ImportError: cannot import name 'Draft202012Validator'` on 15 of 15 kinds.

   Nor can the two be merged into one value: the settings path is a cp310 build
   and this process is miniconda 3.13; m1 measured
   `ModuleNotFoundError: No module named 'rpds.rpds'` when it reached the run
   process.

**So: sys.path in-process, environment untouched.** Children inherit no
PYTHONPATH, `live.get("PYTHONPATH", …)` misses, and the settings file's value
reaches the 3.10 validators.

`CLAUDE_CONFIG_DIR` is passed through the environment on purpose — it is
`_RESERVED`, so the harness handles it rather than merging it.

    usage: python3 launch_chain.py --stall-after 900 run --package … --var …
"""

import os
import runpy
import sys
from pathlib import Path

REPO = Path("/home/yihou/dev/git.16-19/infera")
WRAPPER = (REPO / "agent_sys/examples/llm_e2e_performance_optimization"
                  / "e2e-flow/assets/lib/run_with_long_stall.py")


def main() -> int:
    if "PYTHONPATH" in os.environ:
        print(
            "launch_chain: PYTHONPATH is set in the environment "
            f"({os.environ['PYTHONPATH']!r}). Refusing.\n"
            "  It would reach the validators through env_mgr/harness.py:107's\n"
            "  live.get() and override the settings file's jsonschema path,\n"
            "  reinstating the Draft202012Validator ImportError on every kind.\n"
            "  Run this with `env -u PYTHONPATH`, or unset it.",
            file=sys.stderr,
        )
        return 2

    for p in (REPO / "agent_sys", REPO):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # Prove the tree is consistent BEFORE dispatching anything, because a mixed
    # tree fails much later and reads like a package defect.
    #
    # **`find_spec` has THREE failure modes and the first version handled none.**
    # Measured here:
    #     find_spec("no_such_module")        -> None
    #     find_spec("no_such_pkg.sub")       -> raises ModuleNotFoundError
    #     find_spec("json.no_such_sub")      -> None
    # and for a namespace package `spec.origin` is None while
    # `submodule_search_locations` is a list; for a plain module it is the
    # reverse, so `list(...)` on it raises TypeError.
    #
    # A guard that raises AttributeError on "the module is missing" reports a
    # traceback where it owes a sentence — and "missing" is precisely the case
    # it exists to catch. Same shape as `check_deploy_kit` crashing instead of
    # refusing, which cost this round its first launch.
    import importlib.util as u

    def _where(name: str) -> tuple[str | None, str | None]:
        """(origin, error). Never raises, never returns both None."""
        try:
            spec = u.find_spec(name)
        except (ImportError, ValueError) as error:      # missing parent package
            return None, f"{type(error).__name__}: {error}"
        if spec is None:
            return None, "not found on sys.path"
        if spec.origin:
            return spec.origin, None
        locations = list(spec.submodule_search_locations or [])
        if locations:
            return locations[0], None
        return None, "resolved to a spec with neither an origin nor a search path"

    bad = []
    for name in ("agent_sys.cli.main", "cli.main", "env_mgr.harness", "agent"):
        origin, error = _where(name)
        if error is not None:
            bad.append(f"    {name:20} -> {error}")
        elif not str(origin).startswith(str(REPO) + "/"):
            bad.append(f"    {name:20} -> {origin}   (outside {REPO})")
        else:
            print(f"launch_chain: {name:20} -> {origin}", file=sys.stderr)
    if bad:
        print("launch_chain: REFUSING — the import tree is not consistent:\n"
              + "\n".join(bad), file=sys.stderr)
        return 2

    os.chdir(REPO)
    sys.argv = [str(WRAPPER), *sys.argv[1:]]
    runpy.run_path(str(WRAPPER), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
