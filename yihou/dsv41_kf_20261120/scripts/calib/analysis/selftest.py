#!/usr/bin/env python3
"""CPU-only self-test of calib_table.py and smoke/validate_probe.py on synthetic json-tool documents.
Usage: python3 selftest.py <scratch dir inside a yihou path>
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPS = ["hipLaunchKernel", "hipGraphLaunch", "hipModuleLaunchKernel"]


def doc(dispatches, apis, symbols):
    return {"rocprofiler-sdk-json-tool": {
        "callback_records": {"kernel_symbols": [{"payload": {"kernel_id": k, "kernel_name": n}} for k, n in symbols.items()]},
        "buffer_records": {"names": [{"kind": "HIP_RUNTIME_API", "operations": OPS}],
                           "hip_api_traces": apis, "kernel_dispatch": dispatches}}}


def disp(corr, kid, s, e, grid=(1, 1, 1), gexec=0, cap=0):
    return {"correlation_id": {"internal": corr, "external": 0}, "start_timestamp": s, "end_timestamp": e,
            "thread_id": 1, "dispatch_info": {"kernel_id": kid, "dispatch_id": corr * 100 + kid, "queue_id": {"handle": 1},
                                              "grid_size": {"x": grid[0], "y": grid[1], "z": grid[2]},
                                              "workgroup_size": {"x": 256, "y": 1, "z": 1},
                                              "graph_exec_id": {"handle": gexec}, "graph_node_id": {"handle": 0},
                                              "capture_correlation_id": cap}}


def api(corr, op, s, e):
    return {"correlation_id": {"internal": corr}, "operation": op, "start_timestamp": s, "end_timestamp": e, "thread_id": 1}


def main():
    out = Path(sys.argv[1]).resolve()
    assert "yihou" in out.parts
    out.mkdir(parents=True, exist_ok=True)
    symbols = {1: "_expert_a4", 2: "muse_verify_fly_full", 3: "muse_verify_fly_swa", 4: "Cijk_Alik_Bljk_MT192x224x64",
               5: "cross_device_reduce_2stage", 6: "_fused_norm_residual", 7: "kernel_unified_attention_2d"}
    # server-shaped: 3 VERIFY (bs 26), 3 DRAFT, 1 EXTEND; times in ns
    D, A, steps = [], [], []
    t = 1_000_000_000
    corr = 10
    w0 = t
    for i in range(3):
        steps.append({"t0": t, "t1": t + 1_000_000, "stage": "DRAFT", "mode": "DECODE", "draft": True, "bs": 26, "extend_tokens": 0})
        A.append(api(corr, 1, t + 100, t + 200)); D.append(disp(corr, 4, t + 300, t + 500_300, gexec=7)); corr += 1
        t += 2_000_000
        steps.append({"t0": t, "t1": t + 1_000_000, "stage": "VERIFY", "mode": "TARGET_VERIFY", "draft": False, "bs": 26, "extend_tokens": 0})
        A.append(api(corr, 1, t + 100, t + 200))
        g = t + 1000
        for kid, dur, grid in [(1, 12_000_000, (64, 1, 1)), (2, 600_000, (8, 4, 1)), (3, 21_000, (8, 1, 1)),
                               (4, 8_000_000, (1, 1, 1)), (5, 6_000_000, (1, 1, 1)), (6, 1_800_000, (1, 1, 1))]:
            D.append(disp(corr, kid, g, g + dur, grid=grid, gexec=9)); g += dur
        corr += 1
        t += 40_000_000
    steps.append({"t0": t, "t1": t + 1_000_000, "stage": "EXTEND", "mode": "EXTEND", "draft": False, "bs": 1, "extend_tokens": 16384})
    A.append(api(corr, 0, t + 10, t + 20)); D.append(disp(corr, 7, t + 50, t + 300_000_050)); corr += 1
    A.append(api(corr, 0, t + 2_000_000, t + 2_000_010)); D.append(disp(corr, 6, t + 400_000_000, t + 400_010_000)); corr += 1  # OTHER
    D.append(disp(999, 6, t + 5_000, t + 6_000))  # UNATTRIBUTED
    w1 = t + 3_000_000
    sd = out / "server"
    sd.mkdir(exist_ok=True)
    (sd / "trace-111.json").write_text(json.dumps(doc(D, A, symbols)))
    (sd / "steps-111.json").write_text(json.dumps({"pid": 111, "tp_rank": 0, "window_start_ns": w0, "window_end_ns": w1,
                                                    "steps": steps, "close_reason": "steps"}))
    subprocess.check_call([sys.executable, str(HERE / "calib_table.py"), str(sd), "--out", str(out / "table")])
    res = json.loads((out / "table.json").read_text())[0]
    v = res["stages"]["VERIFY"]["buckets_ms_per_step"]
    assert abs(v["moe_mxfp4"] - 12.0) < 1e-6 and abs(v["attn_full"] - 0.6) < 1e-6 and abs(v["attn_swa"] - 0.021) < 1e-6, v
    assert abs(v["dense_gemm"] - 8.0) < 1e-6 and abs(v["allreduce"] - 6.0) < 1e-6, v
    assert res["stages"]["EXTEND"]["steps"] == 1 and res["unattributed_no_api"] == 1 and res["other_outside_forward"] == 1, res
    assert res["stages"]["DRAFT"]["steps"] == 3
    # probe-shaped
    pd = out / "probe"
    pd.mkdir(exist_ok=True)
    D, A, reps = [], [], []
    t = 5_000_000_000
    wt0 = t
    corr = 1
    for i in range(20):
        A.append(api(corr, 1, t + 10, t + 50))
        g = t + 100
        for kid in (4, 6, 6, 6, 6):
            D.append(disp(corr, kid, g, g + 10_000, gexec=3, cap=500 + kid)); g += 10_000
        reps.append({"host_t0": t, "host_t1": t + 60, "event_ms": 0.06})
        corr += 1
        t += 1_000_000
    et0 = t
    for i in range(10):
        A.append(api(corr, 0, t + 10, t + 20)); D.append(disp(corr, 4, t + 30, t + 1000)); corr += 1; t += 2000
    et1 = t
    wt1 = t + 10
    A.append(api(501 + 3, 2, 1, 2))
    (pd / "trace-capture.json").write_text(json.dumps(doc(D, A, symbols)))
    (pd / "sidecar-capture.json").write_text(json.dumps({"mode": "capture", "window_t0": wt0, "window_t1": wt1, "replays": reps,
                                                         "eager": {"host_t0": et0, "host_t1": et1, "count": 10}}))
    rc = subprocess.call([sys.executable, str(HERE.parent / "smoke/validate_probe.py"), str(pd), "capture"],
                         stdout=subprocess.DEVNULL)
    assert rc == 0, "validator should PASS on the synthetic probe"
    print("selftest OK")


if __name__ == "__main__":
    main()
