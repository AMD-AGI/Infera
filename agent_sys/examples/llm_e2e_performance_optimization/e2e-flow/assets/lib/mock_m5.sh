#!/usr/bin/env bash
# The three m5 mock adaptations that `mock.sh` cannot do, because they are not
# one-to-one copies. `MOCK-MAP.md` calls them (D), (E) and (F).
#
#   mock_m5.sh arms      <environment.yaml>   -- (D) six sealed kinds -> two
#   mock_m5.sh report    <environment.yaml>   -- (E) the report, plus M5.4's blocks
#   mock_m5.sh packup    <environment.yaml>   -- (F) the unsealed out-of-band packup
#
# **Nothing here invents a measurement.** Every number comes out of a sealed
# handoff that a real run on this cluster produced. Three things are written that
# no sealed artefact carries, and each is labelled in the artefact itself:
#
#   * `environment.yaml`, which post-dates the sealed handoffs — adaptation (A),
#     rendered by `env_render.py` from m1's record.
#   * the `serve` step in `env/steps.json` — no sealed artefact records the
#     bring-up window. The mock derives one that sits before the arm's first
#     measured step and after the other arm's last, which is what the real
#     ordering was, and marks it `"note": "mock ..."`.
#   * `stock_vs_m2` and `kernel_reconciliation` in the report — these are the two
#     blocks the mission added (M5.1.3.1, M5.1.3.2) and the 2026-09-02 run
#     predates both, so `stock_vs_m2` is filled in as *not measured, and here is
#     why*, which is the case the schema requires a producer to state rather
#     than omit.
set -euo pipefail

: "${E2E_MOCK_ROOT:=/shared_nfs/yihou/agent_sys/cheat_for_mock}"
: "${E2E_MOCK_REPORT:=refused}"
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
S="${E2E_MOCK_ROOT}/stage5-integration"

what="${1:?usage: mock_m5.sh arms|report|packup <environment.yaml>}"
# **No apostrophe in a `${var:?word}` message.** bash parses `word` with quoting
# rules still active, so a lone `'` inside it opens a string that never closes
# and the whole file fails to parse — at a line number in a *later* heredoc,
# because that is where the parser finally gives up. Measured here: "from m1's
# deploy_kit" reported a syntax error 80 lines further down, in unrelated code.
envyaml="${2:?the environment.yaml to carry forward, out of the deploy_kit m1 produced}"

out_for() {
  # `env_mgr/grants.py:450 _env_name`: uppercase, every non-alphanumeric to `_`.
  local var
  var="AGENT_SYS_OUTPUT_$(printf '%s' "$1" | tr -c '[:alnum:]' '_' | tr '[:lower:]' '[:upper:]')"
  printf '%s' "${!var:?$var is unset — the task does not declare $1 as an output}"
}

case "$what" in
arms)
  # (D). The bring-up windows below are derived from the sealed step timestamps:
  # stock measured 12:58:12–13:41:32 and patched 13:45:47–14:35:51, so a stock
  # bring-up ending just before 12:58 and a patched one beginning after 13:41:32
  # is what the real ordering was. They are the only invented numbers in this
  # script and they are labelled in the artefact.
  for arm in stock patched; do
    out="$(out_for "${arm}.measurement")"
    stage="$(mktemp -d)"
    for kind in deployment acceptance bench; do
      mkdir -p "$stage/$kind"
      cp -a "$S/${kind}_${arm}/content/." "$stage/$kind/"
    done
    if [ "$arm" = stock ]; then
      started=2026-09-02T12:50:12+00:00; seconds=480
    else
      started=2026-09-02T13:41:47+00:00; seconds=240
    fi
    python3 "$PKG/assets/lib/merge_arm.py" \
      --arm "$arm" \
      --from "$stage/deployment" --from "$stage/acceptance" --from "$stage/bench" \
      --out "$out" \
      --serve-started "$started" --serve-seconds "$seconds" \
      --package "$PKG" --environment "$envyaml"
    python3 - "$out/items/env/steps.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["steps"][0]["note"] = (
    "mock: no sealed artefact records the bring-up window. This one is derived from the "
    "sealed step timestamps so that it sits before this arm's first measured step and "
    "after the other arm's last, which is the order the real run had."
)
json.dump(d, open(p, "w"), indent=2)
PY
    rm -rf "$stage"
  done
  ;;

report)
  # (E). The sealed report's verdict is `false`, and mocking it verbatim
  # correctly stops the graph — which is the only cheap end-to-end test of a
  # strong refusal this package has. `E2E_MOCK_REPORT=accepted` swaps the
  # performance rows for the STOCK CONTROL measured under matched load on the
  # same node (patched 475.7 ms against stock 470.3 ms mean ITL, 1.1% apart),
  # which is what a non-regressing run on this cluster actually looks like.
  out="$(out_for integration_report)"
  mkdir -p "$out/items"
  cp -a "$S/integration_report/content/." "$out/"
  python3 - "$out/items/text.json" "$E2E_MOCK_REPORT" "$PKG" <<'PY'
import json, sys
path, mode, pkg = sys.argv[1:4]
d = json.load(open(path))

# The two blocks the mission added. The 2026-09-02 run predates both.
d["stock_vs_m2"] = {
    "source": None,
    "unavailable_because": (
        "mock: the 2026-09-02 integration run was a standalone package and had no m2 "
        "profiling_evidence as an input, so there is no profiling_mode_off bench for the "
        "stock arm to be compared against. A real run of this flow has one."
    ),
    "tolerance": None, "metrics": [], "ok": None,
}
d["kernel_reconciliation"] = {
    "kernel_speedup": 1.0,
    "kernel_share_of_profile": None,
    "predicted_e2e_speedup": None,
    "observed_e2e_speedup": None,
    "agrees": None,
    "unavailable_because": (
        "mock: the patch under test declares itself a no-op (expect.source == 'mock', "
        "speedup 1.0) and no m2 kernel_table share was carried into this report."
    ),
    "note": None,
}

if mode == "accepted":
    # The stock control, measured on the same node under the same co-tenant load
    # fifteen minutes after the patched arm. Not a synthesised pass: these are
    # measured numbers, and the reason they differ from the sealed report's is
    # that the sealed report compared arms an hour and one neighbour apart.
    control = {
        ("output_token_throughput_tps", "avg"): (46.96, 47.38),
        ("request_throughput_rps", "avg"): (0.39, 0.39),
        ("ttft_ms", "avg"): (2044.10, 2060.77),
        ("ttft_ms", "p90"): (2401.55, 2426.46),
        ("inter_token_latency_ms", "avg"): (470.33, 475.70),
        ("request_latency_ms", "avg"): (58120.0, 58707.22),
        ("request_latency_ms", "p90"): (79210.0, 80079.85),
    }
    for row in d["performance"]:
        pair = control.get((row.get("metric"), row.get("column")))
        if pair is None:
            continue
        stock, patched = pair
        row["stock"], row["patched"] = stock, patched
        row["rel_delta"] = round((patched - stock) / stock, 6) if stock else None
        row["bar"] = 0.05 if row["metric"] in {"output_token_throughput_tps", "request_throughput_rps"} else 0.10
        row["verdict"] = "same"
    d["bars"] = {"max_throughput_regression": 0.05, "max_latency_regression": 0.10,
                 "eval_confidence": 0.95}
    d["verdict"] = {"accepted": True, "reasons": [],
                    "warnings": ["mock: the two arms' numbers are the measured stock control, "
                                 "not the sealed run's arms; see MOCK-MAP.md (E)."]}
json.dump(d, open(path, "w"), indent=2)

# CONTRACT.md 3.4: a structured_text kind copies its schema into items/schema and
# its validator checks the copy is byte-identical to the package's.
import shutil
shutil.copyfile(f"{pkg}/assets/schemas/integration_report.schema.json", path.rsplit("/", 1)[0] + "/schema")
PY
  python3 "$PKG/assets/lib/env_render.py" --inherit "$envyaml" \
    --content-type structured_text --out "$out"
  ;;

packup)
  # (F). No `integration_packup` was ever sealed: the graph stopped at (E)
  # before it was dispatched. The source is the 47-file kit produced out of band
  # by `integration`'s own unmodified packup.py over the nine sealed handoffs and
  # graded PASS against the real check_packup_shape body.
  #
  # **It is not sealed**, and its provenance is PRODUCED-BY-DEPLOY.md in that
  # directory. That is recorded here rather than in a comment somewhere else,
  # because a reader meeting this artefact meets it through this script.
  out="$(out_for e2e_packup)"
  src="${E2E_PACKUP_MOCK:-/shared_nfs/yihou/agent_sys/debugging/integration/packup-out-of-band}"
  # `content/` and not the directory above it: the out-of-band kit is already
  # handoff-shaped — `content/items/codes/{README.md,REPRODUCE.md,results,...}` —
  # because it was produced by a real `packup.py` writing into a real output
  # slot. Only `PRODUCED-BY-DEPLOY.md` sits outside, and that is the provenance
  # record rather than part of the kit.
  kit="$src/content"
  [ -d "$kit/items/codes" ] || { echo "mock_m5: no items/codes under $kit" >&2; exit 1; }
  mkdir -p "$out"
  cp -a "$kit/." "$out/"
  python3 "$PKG/assets/lib/env_render.py" --inherit "$envyaml" \
    --content-type code --out "$out"
  cat > "$out/items/watchout" <<'EOF'
This packup was NOT produced by this run and was never sealed as a handoff.

It is the 47-file kit produced out of band on 2026-09-02 by integration's own
unmodified packup.py over that run's nine sealed handoffs, after the graph
stopped at a correctly refused integration_report. Its provenance record is
PRODUCED-BY-DEPLOY.md in the directory it was copied from.

It is a faithful sample of the SHAPE this kind takes and it describes a
different run from the one that copied it here.
EOF
  echo "mock_m5: packup <- $kit (not sealed; see items/watchout)" >&2
  ;;

*)
  echo "mock_m5: unknown selector: $what" >&2
  exit 2
  ;;
esac
