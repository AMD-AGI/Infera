#!/usr/bin/env python3
"""Validate a probe_torch_graph.py run: python3 validate_probe.py <out_dir> <mode>.

Checks (each PASS/FAIL/INFO) and exit code 1 if any gating check FAILs:
  G1 graph replays visible: dispatches launched by hipGraphLaunch inside the window == replays x K, K >= 3,
     every replay has the same K.
  G2 phase attribution works: every window dispatch maps (correlation id -> HIP API record) to a launch call
     whose CPU start lies inside a host bracket (replay i or the eager block). This is the rule the Muse
     analysis uses to label VERIFY/DRAFT/EXTEND.
  G3 stop works: no dispatch launched after window_t1 (the 5 post-stop replays are absent).
  G4 names: no '<missing symbol>' among window dispatches.
  G5 timing sanity per replay: kernel span (first start .. last end) <= event_ms + 0.05 ms and
     sum(kernel durations) <= event_ms + 0.05 ms.
  I1 causality: count of dispatches whose GPU start precedes the CPU start of their launch call (should be 0;
     a non-zero count means host/GPU clock offset or shifted timestamps).
  I2 graph attribution: fraction of replay dispatches with graph_exec_id != 0 and with a resolvable
     capture_correlation_id (expected >0 only in mode capture).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import calib_trace  # noqa: E402


def main():
    out, mode = Path(sys.argv[1]), sys.argv[2]
    side = json.loads((out / f"sidecar-{mode}.json").read_text())
    tr = calib_trace.load(out / f"trace-{mode}.json")
    w0, w1 = side["window_t0"], side["window_t1"]
    brackets = [(r["host_t0"], r["host_t1"], f"replay{i}") for i, r in enumerate(side["replays"])]
    brackets.append((side["eager"]["host_t0"], side["eager"]["host_t1"], "eager"))
    res = {}
    rows = []
    for d in tr["dispatch"]:
        a = tr["api"].get(d["corr"])
        d["launch_op"] = a["op"] if a else None
        d["launch_t"] = a["start"] if a else None
        rows.append(d)
    in_win = [d for d in rows if d["launch_t"] is not None and w0 <= d["launch_t"] <= w1]
    unmapped = [d for d in rows if d["launch_t"] is None and w0 <= d["start"] <= w1 + 5_000_000_000]
    after = [d for d in rows if d["launch_t"] is not None and d["launch_t"] > w1]

    def bracket(t):
        for b0, b1, lab in brackets:
            if b0 <= t <= b1:
                return lab
        return None

    per = defaultdict(list)
    for d in in_win:
        per[bracket(d["launch_t"])].append(d)
    graph_ops = Counter(d["launch_op"] for k, v in per.items() if k and k.startswith("replay") for d in v)
    ks = [len(per.get(f"replay{i}", [])) for i in range(len(side["replays"]))]
    res["G1_graph_replays_visible"] = {
        "pass": len(set(ks)) == 1 and ks[0] >= 3, "kernels_per_replay": ks, "launch_ops": dict(graph_ops)}
    res["G2_phase_attribution"] = {
        "pass": not unmapped and not per.get(None), "unmapped_no_api_record": len(unmapped),
        "launch_outside_brackets": len(per.get(None, [])), "eager_dispatches": len(per.get("eager", []))}
    res["G3_stop_honoured"] = {"pass": not after, "dispatches_launched_after_stop": len(after)}
    res["G4_kernel_names"] = {"pass": all(d["name"] != "<missing symbol>" for d in in_win),
                              "names": dict(Counter(d["name"][:80] for d in in_win).most_common(12))}
    bad = []
    ratios = []
    for i, r in enumerate(side["replays"]):
        v = per.get(f"replay{i}", [])
        if not v:
            continue
        span = (max(d["end"] for d in v) - min(d["start"] for d in v)) / 1e6
        busy = sum(d["end"] - d["start"] for d in v) / 1e6
        ratios.append(round(span / r["event_ms"], 3) if r["event_ms"] else None)
        if span > r["event_ms"] + 0.05 or busy > r["event_ms"] + 0.05:
            bad.append({"replay": i, "span_ms": span, "busy_ms": busy, "event_ms": r["event_ms"]})
    res["G5_timing_vs_hip_events"] = {"pass": not bad, "violations": bad[:5], "span_over_event": ratios}
    caus = [d for d in in_win if d["start"] < d["launch_t"]]
    res["I1_causality_violations"] = {"count": len(caus), "of": len(in_win),
                                      "worst_ns": min((d["start"] - d["launch_t"] for d in caus), default=0)}
    rep = [d for k, v in per.items() if k and k.startswith("replay") for d in v]
    res["I2_graph_attribution"] = {
        "graph_exec_nonzero": sum(1 for d in rep if d["graph_exec"]) / max(1, len(rep)),
        "capture_corr_resolved": sum(1 for d in rep if d["capture_corr"] and d["capture_corr"] in tr["api"])
        / max(1, len(rep)),
        "capture_ops": dict(Counter(tr["api"][d["capture_corr"]]["op"] for d in rep
                                    if d["capture_corr"] in tr["api"]))}
    res["sdk_maps"] = side.get("sdk_maps")
    res["verdict"] = "PASS" if all(v["pass"] for k, v in res.items() if k.startswith("G")) else "FAIL"
    (out / f"validate-{mode}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
