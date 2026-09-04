#!/bin/sh
# Print the path of a run's `deploy_kit` environment record.
#
#     kit_env.sh <run-directory>            -> prints the path, or exits 1
#     cat "$(kit_env.sh "$RUN")"
#     grep -E '^  (tp_size|node|image):' "$(kit_env.sh "$RUN")"
#
# **One authority for a read that five RUN-PLAN sections do** (CONTRACT §4.3).
# Every rung's section begins by taking deployment facts out of the rung below's
# run, and until 2026-09-04 all five did it with the same one-liner:
#
#     find "$RUN" -path '*items/codes/environment.yaml' | head -1
#
# That line is wrong twice and returned the right answer every time, which is
# why it survived five copies.
#
# **It does not scope to `deploy_kit`, and it is not even deterministic.** On a
# completed tree the pattern matches **17 paths**: three real handoffs —
# `deploy_kit`, `kernel_optimization` and `operator_workset` are all `code`-typed
# and carry the record at the same relative path — plus fourteen staged copies
# under `zones/…/handoffs/` and validation `materials/`.
#
# Measured on `20260904T112414-cf3e82`, ten invocations of the one-liner against
# an unchanging tree: **six returned `kernel_optimization`, four returned
# `deploy_kit`.** `find` here is `bfs`, which does not promise directory order,
# so `head -1` is a coin flip between three handoffs rather than a fixed wrong
# pick. The values are right whichever way it lands, because CONTRACT §2 puts
# one identical environment document in all fifteen kinds — so the read changes
# its source between two runs of the same command and nobody can tell. That is
# the purest form of a check that is correct by position rather than by
# construction: there is not even a stable position.
#
# **It does not order versions.** `find` walks directory order, so with `v0`,
# `v1`, `v2` present it reads the **oldest**. Confirmed on a fixture: the
# superseded `v0` yielded `tp_size: 2` and a stale image while `v2` held the
# right ones. This half is masked today only because rung 1's failed `v0` holds
# **0 files** and so has no record to match — and rungs 3, 4 and 5 read from
# rungs that will have retried, so they are the copies most likely to meet a
# populated one. A stale `tp_size` here becomes a wrong `--var expect_ranks`,
# which `check_trace_coverage` is `strong` about, on a capture nobody re-reads.
#
# Reads the store for the kind rather than pattern-matching the path, and takes
# the highest version that actually holds the file — a failed attempt leaves a
# version directory behind with nothing in it.
#
# **Works before the handoff seals.** Verified against rung 1 while `deploy_kit`
# was still `generating`: m1 writes the record at bring-up, so every value a
# later rung needs is readable about an hour before the seal. What waits on the
# seal is the graph edge, not a number.
set -eu

RUN="${1:?usage: kit_env.sh <run-directory>}"
[ -d "$RUN" ] || { echo "kit_env: not a directory: $RUN" >&2; exit 1; }

KIND="${KIT_ENV_KIND:-deploy_kit}"

FOUND="$(python3 - "$RUN" "$KIND" <<'PY'
import json, pathlib, sys

run, kind = pathlib.Path(sys.argv[1]), sys.argv[2]
store = run / "store" / "handoff"
if not store.is_dir():
    sys.exit(0)
for f in sorted(store.glob("*.json")):
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    if d.get("type") != kind:
        continue
    versions = sorted((run / "handoffs" / d["id"]).glob("v*"),
                      key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1,
                      reverse=True)
    for v in versions:
        env = v / "content" / "items" / "codes" / "environment.yaml"
        if env.is_file():
            print(env)
            sys.exit(0)
PY
)"

if [ -z "$FOUND" ]; then
  echo "kit_env: no $KIND environment.yaml under $RUN" >&2
  echo "  Looked in <run>/store/handoff/*.json for type=$KIND, then that" >&2
  echo "  handoff's highest v<N>/content/items/codes/environment.yaml." >&2
  echo "  If the run never reached m1 there is nothing to read and the rung" >&2
  echo "  below has to run first — that is a real answer, not a broken query." >&2
  exit 1
fi
echo "$FOUND"
