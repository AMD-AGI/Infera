#!/bin/sh
# Print the path of a run's `deploy_kit` environment record.
#
#     kit_env.sh <run-directory>            -> prints the path, or exits 1
#     cat "$(kit_env.sh "$RUN")"
#     grep -E '^  (tp_size|node|image):' "$(kit_env.sh "$RUN")"
#
# **Only stdout is the path.** The version taken and the store's opinion of it
# go to stderr, so `$(…)` stays clean and a person watching still sees them.
#
# `KIT_ENV_KIND` overrides the kind, and it can only name a **`code`-typed** one
# — `deploy_kit`, `kernel_optimization`, `operator_workset`, `e2e_packup` —
# because this looks under `items/codes/`. A `reproducible` or `structured_text`
# kind keeps the same document at `items/env/environment.yaml` (CONTRACT §2), so
# `KIT_ENV_KIND=stock.measurement` refuses with *"no … environment.yaml"*, which
# reads like "no such handoff" and means "wrong content type". Found by aiming
# the invalid-status test at the wrong kind.
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

# **Two filters, not one** — m3's refinement, and it is the difference between
# fixing this and moving it. Scoping by kind alone does NOT reach one path,
# because the fourteen staged copies under `zones/…/handoffs/` and validation
# `materials/` are *also* `deploy_kit`. Restricting to `<run>/handoffs/` removes
# those 14 of 17; the kind then picks 1 of the 3 remaining. A helper that did
# only the second would still need `head -1` to break the tie, which is the
# mechanism of the original defect surviving its own fix.
#
# The `<run>/handoffs/<id>/v*` glob below IS the first filter, structurally —
# it can only ever look inside the store's own tree.
candidates = []
for f in sorted(store.glob("*.json")):
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    if d.get("type") != kind:
        continue
    # **The store's version number and the on-disk `vN` are two namespaces**, and
    # m3 found the disagreement: on `20260904T114914-0a0cdd`, five handoffs —
    # `deploy_kit` among them — carry a `v1` directory while the store records
    # only version 0. Measured before acting on it: in **every** one of those
    # five, `v0` holds **zero files** and `v1` holds the content. So the two
    # rules do not disagree about which artefact — there is exactly one
    # populated directory and both select it.
    #
    # Swept to be sure the ordering rule is not quietly wrong somewhere else:
    # **283 handoff directories across every run, and none has more than one
    # populated version.** 92 have their content somewhere other than `v0`,
    # which is why skipping empty directories is load-bearing and why ordering
    # is not. The `if env.is_file()` below is doing the real work.
    status = {v.get("version"): v.get("status") for v in (d.get("versions") or [])}
    for v in sorted((run / "handoffs" / d["id"]).glob("v*"),
                    key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1,
                    reverse=True):
        env = v / "content" / "items" / "codes" / "environment.yaml"
        if env.is_file():
            # Highest version WITHIN one handoff is an ordering, not a tie-break:
            # a later version supersedes an earlier one by definition. m3 asked
            # to narrow that to *when it succeeded*, and the sweep says the case
            # has never arisen — so the rule stays and the reader is told what it
            # took instead of being asked to trust it.
            candidates.append((d["id"], env, v.name, status))
            break

# **Refuse rather than pick**, m3's, and the argument is this file's own history:
# a silent tie-break is how the defect existed in the first place. Two handoffs
# of one kind in one run is a fact nobody predicted, and picking costs whatever
# the wrong copy says on the day the copies stop agreeing.
if len(candidates) > 1:
    print("AMBIGUOUS", file=sys.stderr)
    for hid, env, _vn, _st in candidates:
        print(f"  {hid}  {env}", file=sys.stderr)
    sys.exit(3)

if candidates:
    hid, env, vname, status = candidates[0]
    # **Say which version was taken, and what the store thinks of it** — m3's
    # cheapest and best suggestion. One line converts a silent choice into a
    # readable one, so the next person meets a fact instead of a mystery.
    #
    # **Reported, never judged**, the same call `check_environment` makes about
    # `runtime.container`. Refusing on a status that is not `valid` would break
    # the read this script exists for: rung 1's kit is `generating` for the hour
    # before it seals, and reading it then is the point. But `invalid` is real —
    # three handoffs carry it on `114914` — so a caller reading a kit the store
    # has rejected should be told, not stopped.
    said = status.get(0, "not recorded by the store")
    note = "" if said == "valid" else "   <-- NOT valid; read it knowing that"
    print(f"kit_env: {kind} {hid[:8]} {vname}, store says {said}{note}", file=sys.stderr)
    print(env)
PY
)" || { rc=$?; [ "$rc" -eq 3 ] && {
  echo "kit_env: more than one $KIND in $RUN holds an environment record (listed above)." >&2
  echo "  Not picking one. Two handoffs of a kind in one run is unpredicted, and" >&2
  echo "  a silent choice here is exactly the defect this script replaced." >&2
  echo "  Set KIT_ENV_KIND, or name the path directly once you know which." >&2
  exit 1
}; exit "$rc"; }

if [ -z "$FOUND" ]; then
  echo "kit_env: no $KIND environment.yaml under $RUN" >&2
  echo "  Looked in <run>/store/handoff/*.json for type=$KIND, then that" >&2
  echo "  handoff's highest v<N>/content/items/codes/environment.yaml." >&2
  echo "  If the run never reached m1 there is nothing to read and the rung" >&2
  echo "  below has to run first — that is a real answer, not a broken query." >&2
  exit 1
fi
echo "$FOUND"
