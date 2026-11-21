"""Shared loader for rocprofiler-sdk json-tool traces (fork rocm-systems, json-tool.cpp layout).

Layout used [F: json-tool.cpp buffer_records/callback_records; save.hpp field names; inspect_graph.py]:
  doc["rocprofiler-sdk-json-tool"]["buffer_records"]["kernel_dispatch"][i] =
      {correlation_id:{internal,external}, start_timestamp, end_timestamp, thread_id,
       dispatch_info:{kernel_id, dispatch_id, queue_id, grid_size, workgroup_size,
                      graph_exec_id:{handle}, graph_node_id:{handle}, capture_correlation_id}}
  ...["buffer_records"]["hip_api_traces"][i] = {correlation_id:{internal}, operation, start_timestamp,
                                                 end_timestamp, thread_id}
  ...["buffer_records"]["names"] = [{kind, operations:[...]}]
  ...["callback_records"]["kernel_symbols"][i]["payload"] = {kernel_id, kernel_name}
Timestamps: CLOCK_BOOTTIME ns [F: lib/common/utility.hpp default_clock_id].
"""
import json
from pathlib import Path


def _handle(v):
    if isinstance(v, dict):
        return v.get("handle", 0)
    return v or 0


def _xyz(v):
    if isinstance(v, dict):
        return (v.get("x", 0), v.get("y", 0), v.get("z", 0))
    return tuple(v) if isinstance(v, (list, tuple)) else (v,)


def clean_name(n):
    return n[:-3] if n.endswith(".kd") else n


def load(path):
    doc = json.loads(Path(path).read_text())
    data = doc["rocprofiler-sdk-json-tool"]
    br = data["buffer_records"]
    cb = data.get("callback_records", {})
    symbols = {}
    for r in cb.get("kernel_symbols", []):
        pl = r.get("payload", r)
        symbols[pl["kernel_id"]] = clean_name(pl.get("kernel_name") or pl.get("formatted_kernel_name") or "")
    hip_ops = []
    for k in br.get("names", []):
        if k.get("kind") == "HIP_RUNTIME_API":
            hip_ops = k.get("operations", [])
            break
    api = {}
    for r in br.get("hip_api_traces", []):
        op = r.get("operation")
        api[r["correlation_id"]["internal"]] = {
            "start": r["start_timestamp"], "end": r["end_timestamp"], "thread": r.get("thread_id"),
            "op": hip_ops[op] if isinstance(op, int) and op < len(hip_ops) else str(op)}
    disp = []
    for r in br.get("kernel_dispatch", []):
        info = r["dispatch_info"]
        cap = info.get("capture_correlation_id", 0)
        cap = cap.get("internal", 0) if isinstance(cap, dict) else (cap or 0)
        disp.append({
            "start": r["start_timestamp"], "end": r["end_timestamp"],
            "corr": r["correlation_id"]["internal"], "thread": r.get("thread_id"),
            "kernel_id": info["kernel_id"], "name": symbols.get(info["kernel_id"], "<missing symbol>"),
            "queue": _handle(info.get("queue_id")), "grid": _xyz(info.get("grid_size")),
            "wg": _xyz(info.get("workgroup_size")), "graph_exec": _handle(info.get("graph_exec_id")),
            "graph_node": _handle(info.get("graph_node_id")), "capture_corr": cap})
    disp.sort(key=lambda d: d["start"])
    meta = {k: data.get(k) for k in ("metadata",) if k in data}
    return {"dispatch": disp, "api": api, "symbols": symbols, "meta": meta}
