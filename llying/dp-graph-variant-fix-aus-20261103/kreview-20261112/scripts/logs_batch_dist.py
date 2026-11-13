#!/usr/bin/env python3
"""Decode batch-size distribution of GLM-5.2 PD-disaggregated SGLang runs, derived from the raw decode logs.

  python3 /perf_apps/liyingli/bench_agentx/kreview-20261112/scripts/logs_batch_dist.py            # everything
  python3 .../logs_batch_dist.py --check runs-n04-33/c208-1109T1152-kopt-pp8 [--check-n 5]         # hand check
  python3 .../logs_batch_dist.py --only c208                                                      # subset, no files written

Writes  out/logs/BATCH-DIST.md  and  out/logs/batch_dist.json  (stdlib only, logs are streamed line by line).

Method (details are repeated in the header of BATCH-DIST.md):
  * every `Decode batch` line of DP rank r is one "rank-tick": the state of rank r after `decode_log_interval`
    (N, read from server_args in the same log) of ITS OWN decode steps; the gap to the rank's previous line
    divided by N is the step time.
  * the state of the other ranks at that tick is their log line nearest in time (either side); a rank with
    no line within `tol` is counted as idle (0 running).
  * busiest = max over ranks, graph batch = smallest captured size >= busiest (capture list is read from the
    `Capture target verify CUDA graph end. bs=[...]` line of the same log).
  * tick weight = 1 / (#ranks with a line within tol) so that N engine steps count once whichever rank logged
    them; request weight = tick weight * total running at the tick; time weight = tick weight * gap.
"""

import argparse
import bisect
import calendar
import glob
import json
import os
import re
import statistics as st
import sys
from multiprocessing import Pool

ROOT = "/perf_apps/liyingli/vfix-logs"
OUT = "/perf_apps/liyingli/bench_agentx/kreview-20261112/out/logs"
CMD = "python3 /perf_apps/liyingli/bench_agentx/kreview-20261112/scripts/logs_batch_dist.py"

# (group, path relative to ROOT, note).  v2/runs/* and the remaining runs-n06-25 c104 runs are added by discovery.
MANIFEST = [
    ("A", "runs-n04-33/c208-1109T1152-kopt-pp8", ""),
    ("A", "runs-n04-33/c208-1104T2113", ""),
    ("A", "runs-n06-25/c208-1104T0523", ""),
    ("A", "runs-n06-25/dprof-c208-1104T0758", "decode-profile run, 25 min window"),
    ("A", "opt/partial/c208-1110T0900", "partial copy"),
    ("A", "opt/partial/c208-1110T2109", "partial copy"),
    ("B", "runs-n02-33/c104-1105T0937", ""),
    ("B", "runs-n06-25/c104-1103T1909", ""),
]
MIN_WINDOW_S = 300  # runs whose usable window is shorter are listed as skipped

FINDINGS = [
    "At C208 the TP8/DP8 decode never runs anywhere near 208/8 = 26 requests per rank. Total running p50 is 47-50 with one PP8 prefill (kopt-pp8, MegaMoeTP mt4q2), 63 in the truncated MegaMoeTP run, 75-89 with two prefill instances (2P1D). Busiest-rank p50 is 10-15, p90 15-23, p99 21-28.",
    "TP8/DP8 graph batch ranges by engine steps: 1P(PP8)+1D runs 80% in 7-16 and 95% in 4-18 (5-18 for mt4q2; 7-18 / 4-20 for the truncated MegaMoeTP run); 2P1D runs 80% in 10-22 and 95% in 5-24 (the 25-minute dprof run 5-22 / 3-26). Request-weighted the ranges move up by one or two sizes: 7-16 / 5-20 for kopt-pp8, 12-20 / 6-24 for 2P1D. In MoE tokens (M = 32 x graph batch) by steps: 224-512 (80%) and 128-576 (95%) for the PP8 runs, 320-704 and 160-768 for 2P1D.",
    "TP4/DP4 sweep, 80% / 95% of engine steps: K=5 C32 2-5 / 1-6; K=5 C48 3-8 / 2-10; K=5 C64 6-16 / 2-16; K=5 C88 (30-minute window, mem-fraction 0.93) 6-20 / 4-24; K=3 C88 10-22 / 5-24; K=3 C104 10-22..24 / 6-26..30 (seven 60-minute runs; the two v2-1node runs with 10 s TTFT sit lower at 7-20 / 5-24); K=3 C112 10-22 / 7-28. With DP balance on, the upper ends drop: K=5 C32 2-4 / 1-5, C48 3-7 / 2-8, C64 6-12 / 2-12, K=3 C88 10-18 / 5-20, C104 10-20 / 6-22. The 1P2D runs (two TP4 engines) sit at 3-10 / 2-12 (C96) and 5-12 / 4-16..18 (C128).",
    "Captured sizes above 32 exist only on TP4 (40, 44, ... 64) and carry at most 2.4% of request-steps and 1.6% of steps in any run; sizes above 24 carry at most 13% of request-steps in the 60-minute runs. The busiest rank never exceeded the largest captured size and `cuda graph` is True on every line.",
    "A large share of the rows a graph executes is padding: real / padded rows is 0.58 for the two complete PP8 TP8 runs, 0.63-0.67 for the other TP8 runs, 0.59-0.75 for TP4 without DP balance and 0.78-0.82 with it. The busiest rank holds 1.42-1.66x the mean rank (p50) on TP8, 1.26-1.60x on TP4 without DP balance, 1.17-1.33x with it.",
    "Step time is a staircase in the graph batch at K=3 ('Median step-time increment' table): 6 to 7 costs +0.1 ms, 7 to 8 +0.9, 8 to 10 +4.3 (minimum +2.8 over 24 engines), then +1.5, +2.1, +1.1, +2.9, +1.7, +2.1, +1.5 for the following sizes. The largest single cost is the move from 8 to 10 (32 to 40 attention rows per rank), at the same graph batch on TP4 and TP8. One run (runs-n02-33/c104-1105T0937, fp8 per-token config) is non-monotonic: 12 is slower than 14-16 and 20 slower than 22. Which kernel causes this cannot be seen in these logs.",
    "Context per KV-holding request grows through the window: p50 95-116K tokens in the first 10 minutes, 165-250K in the last 10 minutes of the 60-minute runs (132-174K at the end of the 25-30 minute runs). Indexer cost measured early in a run is not representative of the end.",
    "The decode starts empty at the window start in every run (first `Decode batch` line 2-5 s after the profiling phase starts), so the first minutes of the scored window are a ramp at small batches (first 10 minutes: busiest p50 7 in the TP8 runs against 9-18 later).",
    "In the kopt-pp8 run each rank's KV pool is 92% full at the median (99% at p90) while the rank runs 6.3 requests and holds 11.2 more in the transfer queue; 9% of engine ticks have at least one rank with nothing running. In the two 60-minute 2P1D runs the pool is 71% full at the median with 4.8-6.8 requests in transfer.",
    "Check on the whole chain: logged accept length divided by the request-weighted step time (p50 and p90) reproduces the client interactivity p50 / p90 of every scored run to within 3.1 tok/s (ratio 0.99-1.03). The decode summary lines in the score files agree with these numbers once weighted by wall time instead of steps (their medians are 1-2 higher for that reason); their step-ms-by-busiest lines agree to within 1.2 ms.",
    "Not all group B runs are single-node: by the node IPs in engine-config.env the runs-n06-25 c104 runs, v2-4p4 and v2-1p2d have prefill and decode on different nodes (layout column).",
]

# --------------------------------------------------------------------------------------------- parsing
LINE = re.compile(
    r"^(?:(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z )?\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) DP(\d+) TP(\d+)\] (.*)$"
)
DEC = re.compile(
    r"Decode batch(?: \[\d+\])?, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), (.*)$"
)
KV = {
    "acc": re.compile(r"accept len: ([\d.]+)"),
    "pre_usage": re.compile(r"pre-allocated usage: ([\d.]+)"),
    "npre": re.compile(r"#prealloc-req: (\d+)"),
    "nxfer": re.compile(r"#transfer-req: (\d+)"),
    "thr": re.compile(r"gen throughput \(token/s\): ([\d.]+)"),
    "nq": re.compile(r"#queue-req: (\d+)"),
}
CG = re.compile(r"cuda graph: (True|False)")
CAP_END = re.compile(r"Capture (target verify|draft decode|draft extend) CUDA graph (?:end|begin)\..*?bs=\[([\d, ]+)\]")
CAP_TOK = re.compile(r"Capture (target verify|draft decode|draft extend) CUDA graph begin\..*?num_tokens_per_req=(\d+)")
MAXTOK = re.compile(r"max_total_num_tokens=(\d+).*?max_running_requests=(\d+)")
ARGS = {
    k: re.compile(r"'%s': ([^,}]+)" % k)
    for k in (
        "speculative_num_steps",
        "speculative_num_draft_tokens",
        "tp_size",
        "decode_log_interval",
        "max_running_requests",
        "cuda_graph_bs",
        "cuda_graph_max_bs",
        "schedule_policy",
    )
}
_EP = {}


def epoch(s19):
    """'YYYY-MM-DD?HH:MM:SS' (UTC) -> epoch seconds; the date part is cached."""
    d = s19[:10]
    if d not in _EP:
        _EP[d] = calendar.timegm((int(d[:4]), int(d[5:7]), int(d[8:10]), 0, 0, 0))
    return _EP[d] + int(s19[11:13]) * 3600 + int(s19[14:16]) * 60 + int(s19[17:19])


def iso(t):
    import time

    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))


def parse_decode_log(path, keep_raw=False):
    """Stream one decode log.  Returns meta + per-rank list of records
    (t, running, token, pre_usage, npre, nxfer, nq, acc, thr, cuda_graph[, raw])."""
    ranks, meta = {}, {"path": path, "capture": {}, "rows": {}, "args": {}, "sec_res": False}
    n_dec = n_bad = n_nots = 0
    t_first = t_last = None
    with open(path, errors="replace") as fh:
        for line in fh:
            if "Decode batch" in line:
                m = LINE.match(line)
                d = DEC.search(m.group(6)) if m else None
                if not d:
                    n_bad += 1
                    continue
                if m.group(1):
                    t = epoch(m.group(1)) + (float(m.group(2)) if m.group(2) else 0.0)
                else:
                    t = float(epoch(m.group(3)))
                    n_nots += 1
                tail = d.group(4)
                vals = {}
                for k, rx in KV.items():
                    x = rx.search(tail)
                    vals[k] = float(x.group(1)) if x else None
                cg = CG.search(tail)
                rec = (
                    t,
                    int(d.group(1)),
                    int(d.group(2)),
                    vals["pre_usage"],
                    int(vals["npre"] or 0),
                    int(vals["nxfer"] or 0),
                    int(vals["nq"] or 0),
                    vals["acc"],
                    vals["thr"],
                    (cg.group(1) == "True") if cg else None,
                )
                if keep_raw:
                    rec = rec + (line.rstrip("\n"),)
                ranks.setdefault(int(m.group(4)), []).append(rec)
                n_dec += 1
                t_last = t
                if t_first is None:
                    t_first = t
            elif "Capture " in line and "CUDA graph" in line:
                x = CAP_END.search(line)
                if x:
                    meta["capture"].setdefault(x.group(1), [int(v) for v in x.group(2).split(",")])
                x = CAP_TOK.search(line)
                if x:
                    meta["rows"].setdefault(x.group(1), int(x.group(2)))
            elif "max_total_num_tokens=" in line and "max_total" not in meta:
                x = MAXTOK.search(line)
                if x:
                    meta["max_total"], meta["max_running_rank"] = int(x.group(1)), int(x.group(2))
            elif "server_args=" in line and not meta["args"]:
                for k, rx in ARGS.items():
                    x = rx.search(line)
                    if x:
                        meta["args"][k] = x.group(1).strip().strip("'")
                m = LINE.match(line)
                meta["t_boot"] = epoch(m.group(1)) if m and m.group(1) else None
    for v in ranks.values():
        v.sort(key=lambda r: r[0])
    meta.update(n_dec=n_dec, n_bad=n_bad, t_first=t_first, t_last=t_last, sec_res=n_nots > n_dec / 2)
    return meta, ranks


# --------------------------------------------------------------------------------------------- run metadata
def read_env(path):
    env = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                env[k] = v.replace("\\", "")
    return env


PH0 = "Phase profiling (profiling) started"
PH1 = "Phase profiling (profiling) sending complete"
NS = re.compile(r"phase=CreditPhase\.PROFILING.*?start_ns=(\d+), sent_end_ns=(\d+)")
TOD = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d+)")
DOCK = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z ")


def find_window(run_dir, meta):
    """Measurement window = AgentX 'Phase profiling started' .. 'sending complete' (what score_summary.sh uses).
    Returns (t0, t1, source); t1 None when the phase never completed."""
    t0 = t1 = None
    src = None
    for f in sorted(glob.glob(os.path.join(run_dir, "logs", "*agentx.log"))):
        for line in open(f, errors="replace"):
            if "Phase profiling" not in line:
                continue
            m = DOCK.match(line)
            if not m:
                continue
            t = epoch(m.group(1)) + float(m.group(2) or 0)
            if PH0 in line and t0 is None:
                t0 = t
            elif PH1 in line and t1 is None:
                t1 = t
        if t0:
            return t0, t1, "logs/*agentx.log docker timestamps"
    f = os.path.join(run_dir, "agentx", "benchmark.log")
    if os.path.exists(f):
        tod0 = tod1 = None
        for line in open(f, errors="replace"):
            if "rofiling" not in line and "PROFILING" not in line:
                continue
            x = NS.search(line)
            if x:
                return int(x.group(1)) / 1e9, int(x.group(2)) / 1e9, "agentx/benchmark.log start_ns/sent_end_ns"
            m = TOD.match(line)
            if m and PH0 in line and tod0 is None:
                tod0 = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + float("0." + m.group(4))
            elif m and PH1 in line and tod1 is None:
                tod1 = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + float("0." + m.group(4))
        ref = meta.get("t_boot") or meta.get("t_first")
        if tod0 is not None and ref:
            day = ref - ref % 86400
            t0 = day + tod0
            if t0 < ref - 120:
                t0 += 86400
            if tod1 is not None:
                t1 = day + tod1
                while t1 < t0:
                    t1 += 86400
            return t0, t1, "agentx/benchmark.log time of day (+ date of the decode log)"
    return None, None, None


def read_score_json(run_dir):
    fs = sorted(glob.glob(os.path.join(run_dir, "agentx", "agentx_conc*.json")))
    if not fs:
        return None
    try:
        a = json.load(open(fs[0]))
        r, lat = a["request_metrics"], a["request_metrics"]["latency"]
        return {
            "source": os.path.relpath(fs[0], ROOT),
            "conc": a.get("conc"),
            "tok_s_gpu": r["throughput"]["per_gpu"]["total_tput_tps"],
            "out_tok_s_gpu": r["throughput"]["per_gpu"]["output_tput_tps"],
            "req_s": r["qps"]["mean"],
            "done": a.get("num_requests_successful"),
            "intvty_p50": lat["intvty"]["p50"],
            "intvty_p90": lat["intvty"]["p90"],
            "itl_p50_ms": lat["itl"]["p50"] * 1e3,
            "itl_p90_ms": lat["itl"]["p90"] * 1e3,
            "ttft_p50": lat["ttft"]["p50"],
            "ttft_p90": lat["ttft"]["p90"],
            "ttft_mean": lat["ttft"]["mean"],
            "e2e_p90": lat["e2el"]["p90"],
            "input_tok_mean": r["tokens"]["input"]["mean"],
            "input_tok_p50": r["tokens"]["input"]["p50"],
            "input_tok_p90": r["tokens"]["input"]["p90"],
            "output_tok_mean": r["tokens"]["output_actual"]["mean"],
            "num_decode_gpu": a.get("num_decode_gpu"),
            "num_prefill_gpu": a.get("num_prefill_gpu"),
            "prefill_pp": a.get("prefill_pp"),
            "prefill_tp": a.get("prefill_tp"),
            "prefill_num_workers": a.get("prefill_num_workers"),
        }
    except Exception as e:  # noqa
        return {"error": repr(e), "source": fs[0]}


def read_score_files():
    """All opt/score-*.txt and score-vfix-*.txt, keyed by file label."""
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "opt", "score-*.txt")) + glob.glob(os.path.join(ROOT, "score-*.txt"))):
        d = {"label": os.path.basename(f)[6:-4], "file": os.path.relpath(f, ROOT)}
        for line in open(f, errors="replace"):
            line = line.rstrip("\n")
            if line.startswith("== run="):
                m = re.match(r"== run=(\S+) window (.*?) \.\. ?(.*)$", line)
                if m:
                    d["run"], d["run_base"] = m.group(1), os.path.basename(m.group(1))
                    d["w0"], d["w1"] = m.group(2).strip(), m.group(3).strip()
            elif line.startswith("| C"):
                c = [x.strip() for x in line.strip("|").split("|")]
                d["row"] = line
                try:
                    d["conc"] = int(c[0][1:])
                    d["tok_s_gpu"] = float(c[1].replace(",", ""))
                    d["out_tok_s_gpu"] = float(c[2])
                    d["intvty_p50"], d["intvty_p90"] = [float(x) for x in c[5].split("/")]
                    d["itl_p50_ms"], d["itl_p90_ms"] = [float(x) for x in c[6].split("/")]
                    d["ttft_p50"], d["ttft_p90"], d["ttft_mean"] = [float(x) for x in c[7].split("/")]
                except Exception:
                    pass
            elif line.startswith("decode: running total"):
                d["decode_line"] = line
                m = re.search(r"running total median (\d+), busiest rank median (\d+) p90 (\d+)", line)
                if m:
                    d["tot_med"], d["busy_med"], d["busy_p90"] = map(int, m.groups())
            elif line.startswith("decode step ms by busiest-rank running:"):
                d["step_line"] = line
                d["step_ms"] = {int(a): float(b) for a, b in re.findall(r"@(\d+) (\d+)", line)}
            elif line.startswith("prefill"):
                d.setdefault("prefill_lines", []).append(line)
        out.append(d)
    return out


def prefill_pp(run_dir):
    """Pipeline stages of prefill0, from the PP<s> rank labels in the first lines of its log."""
    fs = sorted(glob.glob(os.path.join(run_dir, "logs", "*-prefill0.log")))
    top = -1
    if fs:
        with open(fs[0], errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 4000:
                    break
                m = re.search(r" PP(\d+)\] ", line[:80])
                if m:
                    top = max(top, int(m.group(1)))
    return top + 1 if top >= 0 else None


def describe(env, meta, sj):
    """Short config description from engine-config.env + the decode log's own server_args."""
    a = meta["args"]
    n = int(a.get("tp_size", 0) or 0)
    pw = int(env.get("PREFILL_WORKERS", "1") or 1)
    pg = len(env.get("PREFILL0_GPUS", "").split(",")) if env.get("PREFILL0_GPUS") else 0
    dw = int(env.get("DECODE_WORKERS", "1") or 1)
    pp = meta.get("prefill_pp") or (len(env["PP_LAYER_PARTITION"].split(",")) if env.get("PP_LAYER_PARTITION") else None)
    two = env.get("PREFILL_NODE_IP") != env.get("DECODE_NODE_IP")
    lay = f"{pw}P({pg}GPU" + (f",PP{pp}" if pp else "") + f")+{dw}D(TP{n}/DP{n})" + (" 2-node" if two else " 1-node")
    blob = " ".join(env.values())
    tags = []
    if "KOPT_MEGATP=1" in blob:
        m = re.search(r"AITER_MEGAMOE_TP_LB_MT=(\d+)", blob)
        q = re.search(r"AITER_MEGAMOE_TP_LB_Q=(\d+)", blob)
        tags.append("MegaMoeTP" + (f"(mt{m.group(1)}" + (f"q{q.group(1)}" if q else "") + ")" if m else ""))
    if "kopt-20261109/configs" in blob or "overlay-kopt" in blob:
        tags.append("kopt")
    if "schedule-policy hrrn" in blob:
        tags.append("prefill-hrrn")
    if "MC_DISABLE_HIP=1" in blob:
        tags.append("nohip")
    if "FP8_PER_TOKEN" in blob:
        tags.append("fp8-pertoken")
    if "--max-total-tokens" in env.get("DECODE_EXTRA_ARGS", ""):
        tags.append("max-total-tokens=" + re.search(r"--max-total-tokens (\d+)", env["DECODE_EXTRA_ARGS"]).group(1))
    m = re.search(r"--mem-fraction-static ([\d.]+)", env.get("DECODE_EXTRA_ARGS", ""))
    if m and m.group(1) != "0.95":
        tags.append("memfrac=" + m.group(1))
    # DP balance = router-side decode request weight: DECODE_REQUEST_WEIGHT (v2 kit) or
    # INFERA_DECODE_REQUEST_WEIGHT in ROUTER_ENV_EXTRA (vfix kit); absent means the router default (off).
    w = env.get("DECODE_REQUEST_WEIGHT")
    if w is None:
        m = re.search(r"REQUEST_WEIGHT=(\d+)", blob)
        w = m.group(1) if m else None
    dpbal = "off (weight not set)" if w is None else ("off" if w == "0" else f"on({w})")
    return lay, tags, dpbal, two


# --------------------------------------------------------------------------------------------- statistics
def wq(pairs, qs):
    """Weighted quantiles of (value, weight) pairs: smallest value whose cumulative weight >= q * total."""
    pairs = sorted(p for p in pairs if p[1] > 0)
    if not pairs:
        return [None] * len(qs)
    tot = sum(w for _, w in pairs)
    out, acc, i = [], 0.0, 0
    for q in qs:
        target = q * tot
        while i < len(pairs) - 1 and acc + pairs[i][1] < target - 1e-12:
            acc += pairs[i][1]
            i += 1
        out.append(pairs[i][0])
    return out


def cover(hist, frac):
    """Narrowest contiguous range of bucket keys (fewest distinct keys, then largest mass) holding >= frac."""
    keys = sorted(hist)
    tot = sum(hist.values())
    if tot <= 0:
        return None
    best = None
    for i in range(len(keys)):
        s = 0.0
        for j in range(i, len(keys)):
            s += hist[keys[j]]
            if s >= frac * tot - 1e-12:
                cand = (j - i, -s, keys[i], keys[j])
                if best is None or cand < best:
                    best = cand
                break
    return [best[2], best[3], round(-best[1] / tot, 4)]


def ceil_cap(cap, x):
    i = bisect.bisect_left(cap, x)
    return cap[i] if i < len(cap) else None


def plateau_window(ranks):
    """Fallback window: first..last 10 s bin whose total running is within 20% of the plateau
    (plateau = median of the bins in the upper half)."""
    bins = {}
    for r, recs in ranks.items():
        for rec in recs:
            bins.setdefault(int(rec[0] // 10), {}).setdefault(r, []).append(rec[1])
    tot = {b: sum(st.mean(v) for v in d.values()) for b, d in bins.items()}
    vals = sorted(tot.values())
    plat = st.median(vals[len(vals) // 2 :])
    ok = sorted(b for b, v in tot.items() if v >= 0.8 * plat)
    return ok[0] * 10.0, ok[-1] * 10.0 + 10.0, plat


def analyze(meta, ranks, w0, w1):
    N = int(meta["args"].get("decode_log_interval", 40))
    n_ranks = int(meta["args"].get("tp_size") or 0) or (max(ranks) + 1)
    cap = meta["capture"].get("target verify") or []
    rows_req = meta["rows"].get("target verify")
    rk = sorted(ranks)
    times = {r: [x[0] for x in ranks[r]] for r in rk}
    gaps = [b[0] - a[0] for r in rk for a, b in zip(ranks[r], ranks[r][1:]) if w0 <= b[0] <= w1]
    gaps = [g for g in gaps if g > 0] or [1.0]
    med_gap = st.median(gaps)
    tol = max(1.5 * med_gap, 1.0) + (1.0 if meta["sec_res"] else 0.0)
    max_gap = 0.25 * N  # 250 ms/step: beyond this the rank was idle for part of the interval

    def snap(r, i):
        t = times[r][i]
        runs, act = [0] * n_ranks, 0
        for q in range(n_ranks):
            if q == r:
                runs[q] = ranks[r][i][1]
                act += 1
                continue
            tq = times.get(q)
            if not tq:
                continue
            j = bisect.bisect_left(tq, t)
            best = None
            for k in (j - 1, j):
                if 0 <= k < len(tq) and (best is None or abs(tq[k] - t) < abs(tq[best] - t)):
                    best = k
            if best is not None and abs(tq[best] - t) <= tol:
                runs[q] = ranks[q][best][1]
                act += 1
        return runs, act

    ticks = []  # dict per rank-tick
    for r in rk:
        prev = None
        recs = ranks[r]
        for i, rec in enumerate(recs):
            t = rec[0]
            if t > w1:
                break
            if i + 1 < len(recs) and recs[i + 1][0] < w0:
                continue  # more than one line before the window
            runs, act = snap(r, i)
            cur = (runs, act)
            if w0 <= t <= w1:
                busiest, total = max(runs), sum(runs)
                gbs = ceil_cap(cap, busiest) if cap else None
                gap = (t - recs[i - 1][0]) if i > 0 else None
                p_busy = max(prev[0]) if prev else None
                p_gbs = ceil_cap(cap, p_busy) if (prev and cap) else None
                ticks.append(
                    dict(
                        t=t, r=r, rec=rec, runs=runs, act=act, busiest=busiest, total=total, gbs=gbs,
                        gap=gap if (gap is not None and (0 < gap or meta["sec_res"]) and gap <= max_gap) else None,
                        stable=(prev is not None and p_gbs == gbs),
                        busy_stable=(prev is not None and p_busy == busiest),
                        own_stable=(i > 0 and recs[i - 1][1] == rec[1]),
                        w=1.0 / act,
                    )
                )
            prev = cur
    if not ticks:
        return None
    ticks.sort(key=lambda x: x["t"])
    W = sum(x["w"] for x in ticks)
    res = {
        "n_ranks": n_ranks, "steps_per_tick": N, "capture_list": cap, "rows_per_req": rows_req,
        "window": [iso(w0), iso(w1)], "window_s": round(w1 - w0, 1),
        "rank_ticks": len(ticks), "engine_ticks": round(W, 1), "engine_steps_est": round(W * N),
        "median_gap_s": round(med_gap, 4), "align_tol_s": round(tol, 3),
        "lines_per_rank": {str(r): sum(1 for x in ranks[r] if w0 <= x[0] <= w1) for r in rk},
        "ticks_with_idle_rank_frac": round(sum(x["w"] for x in ticks if x["act"] < n_ranks) / W, 4),
        "cuda_graph_false_frac": round(sum(1 for x in ticks if x["rec"][9] is False) / len(ticks), 5),
        "over_capture_frac": round(sum(x["w"] for x in ticks if cap and x["gbs"] is None) / W, 5),
    }
    q5 = [0.1, 0.5, 0.9, 0.99, 1.0]
    # 1. running distributions
    pooled = [(x["runs"][q], x["w"]) for x in ticks for q in range(n_ranks)]
    res["per_rank_running_pooled"] = dict(zip(["p10", "p50", "p90", "p99", "max"], wq(pooled, q5)))
    res["per_rank_running"] = {
        str(q): dict(zip(["p10", "p50", "p90", "max"], wq([(x["runs"][q], x["w"]) for x in ticks], [0.1, 0.5, 0.9, 1.0])))
        for q in range(n_ranks)
    }
    res["per_rank_running_own_lines"] = {
        str(r): dict(zip(["p10", "p50", "p90", "max"], wq([(x["rec"][1], 1.0) for x in ticks if x["r"] == r], [0.1, 0.5, 0.9, 1.0])))
        for r in rk
    }
    res["total_running"] = dict(zip(["p10", "p50", "p90", "p99", "max"], wq([(x["total"], x["w"]) for x in ticks], q5)))
    res["total_running_mean"] = round(sum(x["total"] * x["w"] for x in ticks) / W, 2)
    res["busiest"] = dict(zip(["p10", "p50", "p90", "p99", "max"], wq([(x["busiest"], x["w"]) for x in ticks], q5)))
    res["busiest_mean"] = round(sum(x["busiest"] * x["w"] for x in ticks) / W, 2)
    ratio = [(x["busiest"] * n_ranks / x["total"], x["w"]) for x in ticks if x["total"] > 0]
    a, b = wq(ratio, [0.5, 0.9])
    res["busiest_over_mean"] = {"p50": round(a, 3), "p90": round(b, 3)}
    # busiest as seen by a request (request-weighted)
    res["busiest_req_weighted"] = dict(zip(["p10", "p50", "p90", "p99", "max"], wq([(x["busiest"], x["w"] * x["total"]) for x in ticks], q5)))
    # 2. graph batch histograms
    big = (cap[-1] + 1) if cap else 0
    h_t, h_r, h_s, h_own = {}, {}, {}, {}
    for x in ticks:
        g = x["gbs"] if x["gbs"] is not None else big
        h_t[g] = h_t.get(g, 0.0) + x["w"]
        h_r[g] = h_r.get(g, 0.0) + x["w"] * x["total"]
        h_own[g] = h_own.get(g, 0.0) + x["rec"][1]
        if x["gap"] is not None:
            h_s[g] = h_s.get(g, 0.0) + x["w"] * x["gap"]

    def table(h):
        tot = sum(h.values()) or 1.0
        acc, out = 0.0, {}
        for g in sorted(h):
            acc += h[g]
            out[str(g)] = [round(h[g] / tot, 5), round(acc / tot, 5)]
        return out

    res["graph_bs_hist"] = {"by_ticks": table(h_t), "by_requests": table(h_r), "by_time": table(h_s), "by_own_running_xcheck": table(h_own)}
    res["graph_bs_cover"] = {
        k: {"80": cover(h, 0.8), "95": cover(h, 0.95), "p50": wq(list(h.items()), [0.5])[0]}
        for k, h in (("by_ticks", h_t), ("by_requests", h_r), ("by_time", h_s))
    }
    # 6. fill
    if cap:
        fills = [(x["total"] / (n_ranks * x["gbs"]), x["w"]) for x in ticks if x["gbs"]]
        fw = sum(w for _, w in fills)
        res["fill_mean_of_ticks"] = round(sum(f * w for f, w in fills) / fw, 4)
        res["fill_ratio_of_sums"] = round(
            sum(x["total"] * x["w"] for x in ticks if x["gbs"]) / sum(n_ranks * x["gbs"] * x["w"] for x in ticks if x["gbs"]), 4
        )
        res["fill_if_exact_busiest"] = round(
            sum(x["total"] * x["w"] for x in ticks if x["busiest"]) / sum(n_ranks * x["busiest"] * x["w"] for x in ticks if x["busiest"]), 4
        )
    # 4. step time
    def med(v):
        return round(st.median(v), 2) if v else None

    by_g_st, by_g_all, by_b, by_b_st, by_g_thr = {}, {}, {}, {}, {}
    ratio_thr = []
    sm_all = []  # (step ms, tick weight, request weight): timestamp gaps, or throughput-derived for 1 s logs
    for x in ticks:
        rec = x["rec"]
        g = x["gbs"] if x["gbs"] is not None else big
        x["ms"] = None
        if x["gap"] and not meta["sec_res"]:
            ms = x["ms"] = x["gap"] / N * 1e3
            by_g_all.setdefault(g, []).append(ms)
            by_b.setdefault(x["busiest"], []).append(ms)
            if x["stable"]:
                by_g_st.setdefault(g, []).append(ms)
            if x["busy_stable"]:
                by_b_st.setdefault(x["busiest"], []).append(ms)
        # independent estimate: tokens generated in the interval / logged throughput
        if x["own_stable"] and rec[1] > 0 and rec[7] and rec[8] and rec[8] > 1.0:
            gap_thr = N * rec[1] * rec[7] / rec[8]
            if gap_thr <= max_gap:
                if x["stable"]:
                    by_g_thr.setdefault(g, []).append(gap_thr / N * 1e3)
                if x["gap"] and not meta["sec_res"]:
                    ratio_thr.append(gap_thr / x["gap"])
                if meta["sec_res"]:
                    ms = x["ms"] = gap_thr / N * 1e3
                    by_b.setdefault(x["busiest"], []).append(ms)
                    if x["busy_stable"]:
                        by_b_st.setdefault(x["busiest"], []).append(ms)
        if x["ms"] is not None:
            sm_all.append((x["ms"], x["w"], x["w"] * x["total"]))
    res["step_ms_by_graph_bs"] = {
        str(g): {
            "stable_median": med(by_g_st.get(g)), "stable_n": len(by_g_st.get(g, [])),
            "all_median": med(by_g_all.get(g)), "all_n": len(by_g_all.get(g, [])),
            "thr_median": med(by_g_thr.get(g)), "thr_n": len(by_g_thr.get(g, [])),
        }
        for g in sorted(set(by_g_all) | set(by_g_thr))
    }
    res["step_ms_by_busiest"] = {str(b): {"median": med(v), "n": len(v)} for b, v in sorted(by_b.items())}
    res["step_ms_by_busiest_stable"] = {str(b): {"median": med(v), "n": len(v)} for b, v in sorted(by_b_st.items())}
    # padding test: above 8 the capture list has even sizes only, so busiest 2k-1 and 2k share a graph while
    # 2k and 2k+1 do not.  Compare the step-time increments inside a pair and across pairs.
    src_b = {b: st.median(v) for b, v in by_b_st.items() if len(v) >= 30}
    inside = [src_b[b + 1] - src_b[b] for b in src_b if b >= 9 and b % 2 == 1 and b + 1 in src_b and b + 1 <= 32]
    across = [src_b[b + 1] - src_b[b] for b in src_b if b >= 10 and b % 2 == 0 and b + 1 in src_b and b + 1 <= 32]
    res["padding_pair_test"] = {
        "what": "mean step-ms increment busiest 2k-1 -> 2k (same captured graph) vs 2k -> 2k+1 (next graph), busiest 9..32, ticks with unchanged busiest, n >= 30 per size",
        "inside_pair_mean": round(st.mean(inside), 2) if inside else None, "inside_n": len(inside),
        "across_pair_mean": round(st.mean(across), 2) if across else None, "across_n": len(across),
    }
    a = wq([(m, w) for m, w, _ in sm_all], [0.1, 0.5, 0.9])
    b = wq([(m, rw) for m, _, rw in sm_all], [0.1, 0.5, 0.9])
    res["step_ms"] = {"source": "throughput-derived (1 s timestamps)" if meta["sec_res"] else "timestamp gaps",
                      "p10": a[0] and round(a[0], 2), "p50": a[1] and round(a[1], 2), "p90": a[2] and round(a[2], 2),
                      "mean": round(sum(m * w for m, w, _ in sm_all) / sum(w for _, w, _ in sm_all), 2) if sm_all else None}
    res["step_ms_req_weighted"] = {"p10": b[0] and round(b[0], 2), "p50": b[1] and round(b[1], 2), "p90": b[2] and round(b[2], 2)}
    if ratio_thr:
        rs = sorted(ratio_thr)
        res["steps_per_tick_check"] = {
            "what": "median of N*running*accept_len/throughput / timestamp gap on ticks with unchanged own running; 1.0 means N steps per tick",
            "n": len(rs), "p10": round(rs[len(rs) // 10], 4), "p50": round(st.median(rs), 4), "p90": round(rs[len(rs) * 9 // 10], 4),
        }
    acc = [(x["rec"][7], x["rec"][1]) for x in ticks if x["rec"][7] and x["rec"][1] > 0]
    res["accept_len_req_weighted"] = al = round(sum(a * w for a, w in acc) / sum(w for _, w in acc), 3) if acc else None
    if al and b[1]:
        # what a request would see if it stayed on steps of that duration: tokens/s = accept_len / step time
        res["implied_intvty"] = {"p50": round(al * 1e3 / b[1], 1), "p90_slow": round(al * 1e3 / b[2], 1),
                                 "what": "accept_len / request-weighted step time p50 and p90 (tok/s per request); compare with the client-side interactivity p50 / p90"}
    tw = [(x["total"], x["w"] * x["gap"]) for x in ticks if x["gap"]]
    res["total_running_time_weighted_p50"] = wq(tw, [0.5])[0] if tw else None
    res["busiest_time_weighted"] = dict(zip(["p50", "p90"], wq([(x["busiest"], x["w"] * x["gap"]) for x in ticks if x["gap"]], [0.5, 0.9])))
    # 5. context length
    mt = meta.get("max_total")

    def ctx(sub):
        o = {}
        hold, task, run, tok, use = [], [], [], [], []
        for x in sub:
            rec = x["rec"]
            tok.append((rec[2], 1.0))
            if mt:
                use.append((rec[2] / mt, 1.0))
            h = rec[1] + rec[5] + rec[6]
            if h > 0:
                hold.append((rec[2] / h, 1.0))
            h2 = rec[1] + rec[4] + rec[5]
            if h2 > 0:
                task.append((rec[2] / h2, 1.0))
            if mt and rec[3] is not None and rec[1] > 0:
                run.append((max(0.0, rec[2] - rec[3] * mt) / rec[1], float(rec[1])))
        o["n_lines"] = len(sub)
        for name, v in (("tok_per_holder", hold), ("tok_per_run_prealloc_transfer", task), ("tok_per_running_reqw", run), ("kv_tokens_per_rank", tok), ("kv_usage", use)):
            a = wq(v, [0.1, 0.5, 0.9])
            o[name] = {"p10": a[0] and round(a[0], 4 if name == "kv_usage" else 0), "p50": a[1] and round(a[1], 4 if name == "kv_usage" else 0),
                       "p90": a[2] and round(a[2], 4 if name == "kv_usage" else 0)}
        o["running_mean"] = round(st.mean(x["rec"][1] for x in sub), 2) if sub else None
        o["transfer_mean"] = round(st.mean(x["rec"][5] for x in sub), 2) if sub else None
        o["prealloc_mean"] = round(st.mean(x["rec"][4] for x in sub), 2) if sub else None
        o["queue_mean"] = round(st.mean(x["rec"][6] for x in sub), 2) if sub else None
        return o

    res["context"] = {
        "window": ctx(ticks),
        "first10min": ctx([x for x in ticks if x["t"] < w0 + 600]),
        "last10min": ctx([x for x in ticks if x["t"] > w1 - 600]),
        "max_total_num_tokens_per_rank": mt,
    }
    # evolution in 10-minute slices
    ev = []
    k = 0
    while w0 + k * 600 < w1 - 1:
        sub = [x for x in ticks if w0 + k * 600 <= x["t"] < w0 + (k + 1) * 600]
        if sub:
            ws = sum(x["w"] for x in sub)
            c = ctx(sub)
            ev.append({
                "from_min": k * 10,
                "total_running_p50": wq([(x["total"], x["w"]) for x in sub], [0.5])[0],
                "busiest_p50": wq([(x["busiest"], x["w"]) for x in sub], [0.5])[0],
                "busiest_p90": wq([(x["busiest"], x["w"]) for x in sub], [0.9])[0],
                "graph_bs_p50": wq([(x["gbs"] or big, x["w"]) for x in sub], [0.5])[0],
                "step_ms_p50": med([x["ms"] for x in sub if x.get("ms") is not None]),
                "tok_per_holder_p50": c["tok_per_holder"]["p50"],
                "kv_tokens_per_rank_p50": c["kv_tokens_per_rank"]["p50"],
                "kv_usage_p90": c["kv_usage"]["p90"],
                "engine_ticks": round(ws, 1),
            })
        k += 1
    res["evolution_10min"] = ev
    return res, ticks


# --------------------------------------------------------------------------------------------- driver
def discover():
    runs = list(MANIFEST)
    have = {p for _, p, _ in runs}
    for d in sorted(glob.glob(os.path.join(ROOT, "v2", "runs", "*"))):
        p = os.path.relpath(d, ROOT)
        if p not in have:
            runs.append(("B", p, ""))
    for d in sorted(glob.glob(os.path.join(ROOT, "runs-n06-25", "*c104-*"))):
        p = os.path.relpath(d, ROOT)
        if p not in have:
            runs.append(("B-extra", p, "not in the task list; same TP4 C104 family"))
    return runs


def do_run(job):
    group, rel, note = job
    run_dir = os.path.join(ROOT, rel)
    logs = sorted(glob.glob(os.path.join(run_dir, "logs", "*-decode*.log")))
    env = read_env(os.path.join(run_dir, "engine-config.env"))
    sj = read_score_json(run_dir)
    out = []
    if not logs:
        return [{"group": group, "run": rel, "skipped": "no decode log", "note": note}]
    for lg in logs:
        eng = re.search(r"-(decode\d*)\.log$", lg).group(1)
        key = rel + ("" if len(logs) == 1 else ":" + eng)
        meta, ranks = parse_decode_log(lg)
        row = {"group": group, "run": rel, "key": key, "engine": eng, "note": note, "decode_log": os.path.relpath(lg, ROOT),
               "log_bytes": os.path.getsize(lg), "decode_lines": meta["n_dec"], "unparsed_decode_lines": meta["n_bad"],
               "second_resolution_timestamps": meta["sec_res"], "server_args": meta["args"],
               "captured_graphs": meta["capture"], "rows_per_req_by_graph": meta["rows"], "score_json": sj}
        if not ranks:
            row["skipped"] = "decode log has no Decode batch lines"
            out.append(row)
            continue
        meta["prefill_pp"] = prefill_pp(run_dir)
        lay, tags, dpbal, two = describe(env, meta, sj)
        K = meta["args"].get("speculative_num_steps")
        row.update(layout=lay, tags=tags, dpbal=dpbal, K=int(K) if K and K.isdigit() else None,
                   sim_accept_len=env.get("SIM_ACCEPT_LEN"), image=env.get("IMAGE"),
                   conc=(sj or {}).get("conc") or (int(m.group(1)) if (m := re.search(r"c(\d+)(?:-|$)", os.path.basename(rel))) else None))
        m = re.search(r"-c(\d+)$", rel)
        if m:
            row["conc"] = int(m.group(1))
        t0, t1, src = find_window(run_dir, meta)
        wnote = ""
        if t0 is None:
            sf = job_scores.get(os.path.basename(rel)) or job_scores.get(re.sub(r"-kopt-pp8$", "", os.path.basename(rel)))
            if sf and sf.get("w0") and len(sf["w0"]) >= 19:
                # the score file's date is the copy's mtime date and can be a day off: keep the time of day,
                # take the date from the decode log
                tod = sf["w0"][11:19]
                day = iso(meta["t_first"])[:10]
                t0 = float(epoch(day + " " + tod))
                if t0 > meta["t_last"]:
                    t0 -= 86400
                src = f"start from {sf['file']} (time of day; date from decode log)"
                if sf.get("w1") and len(sf["w1"]) >= 19:
                    t1 = float(epoch(iso(t0)[:10] + " " + sf["w1"][11:19]))
                    while t1 < t0:
                        t1 += 86400
        if t0 is None:
            if not any(rel == p for _, p, _ in MANIFEST):
                row["skipped"] = "no AgentX profiling phase and no score-file window (gate / probe run); not a scoring run"
                out.append(row)
                continue
            t0, t1, plat = plateau_window(ranks)
            src = f"UNKNOWN window: plateau heuristic (total running >= 80% of plateau {plat:.0f})"
        if t1 is None:
            t1 = meta["t_last"]
            wnote = "profiling phase never completed (crash / abort / partial copy): window truncated at the last decode line"
        elif t1 > meta["t_last"] + 30:
            wnote = f"decode log ends {t1 - meta['t_last']:.0f} s before the window end"
            t1 = meta["t_last"]
        row.update(window_source=src, window_note=wnote)
        if t1 - t0 < MIN_WINDOW_S:
            row["skipped"] = f"usable window only {t1 - t0:.0f} s ({iso(t0)} .. {iso(t1)}; {src}; {wnote})"
            out.append(row)
            continue
        a = analyze(meta, ranks, t0, t1)
        if a is None:
            row["skipped"] = "no decode lines inside the window"
        else:
            row["stats"] = a[0]
        out.append(row)
    return out


job_scores = {}


def fmt(x, nd=0):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return str(x)


def rng(c):
    return "n/a" if not c else (f"{c[0]}" if c[0] == c[1] else f"{c[0]}-{c[1]}") + f" ({100 * c[2]:.0f}%)"


def rng_plain(c):
    return "n/a" if not c else (f"{c[0]}" if c[0] == c[1] else f"{c[0]}-{c[1]}")


def shape_rng(c, mult):
    return "n/a" if not c else (f"{c[0] * mult}" if c[0] == c[1] else f"{c[0] * mult}-{c[1] * mult}")


def short(key):
    return key.replace("runs-", "").replace("v2/runs/", "v2:").replace("opt/partial/", "partial:")


def write_md(rows, scores, path):
    L = []
    A = L.append
    ok = [r for r in rows if "stats" in r]
    A("# Decode batch-size distribution from raw decode logs (GLM-5.2, SGLang PD-disaggregated, MI355X)\n")
    A(f"Regenerate everything (stdlib only, ~1 min, streams the logs): `{CMD}`  ")
    A(f"Hand check of the tick alignment against raw lines: `{CMD} --check runs-n04-33/c208-1109T1152-kopt-pp8`  ")
    A("Machine-readable copy of every number below (and more): `batch_dist.json` in this directory.\n")
    A("## Findings (written 2026-11-12 from the tables below; re-read the tables if the inputs change)\n")
    for line in FINDINGS:
        A("* " + line)
    A("")
    A("## Method and what is / is not verified\n")
    A("* **Source**: only `logs/*-decode*.log` (`Decode batch` lines), `engine-config.env`, the AgentX phase markers and `agentx/agentx_conc*.json`. No earlier summary is used as input; the `score-*.txt` numbers appear only in comparison columns.")
    A("* **Steps per log line**: `decode_log_interval` is printed in each log's `server_args` (40 in every run here). In the source (`sglang/srt/managers/scheduler_components/batch_result_processor.py` + `metrics_reporter.py::report_decode_stats`) the per-rank counter `forward_ct_decode` advances once per processed decode batch and a line is logged when it is a multiple of the interval. A rank with no running request processes no decode batch, so it neither counts nor logs: a line is 40 of *that rank's non-idle* steps. Empirical check per run (`steps/tick check` in the detail tables): 40 x running x accept-len / logged throughput divided by the timestamp gap, on ticks where the rank's running count did not change, must be 1.0.")
    A("* **Capture list**: read from each log's `Capture target verify CUDA graph end. bs=[...]` line, not assumed. TP8 runs: 1-8, 10-32 step 2 (max 32 because max_running_requests 256 / 8 ranks). TP4 runs: 1-8, 10-32 step 2, **40, 44, 48, ... 64** (step 4 after 40; there is no 36). Rows per request come from `num_tokens_per_req` on the `Capture target verify CUDA graph begin` line (4 for K=3, 6 for K=5).")
    A("* **Graph batch is inferred, not logged**: graph batch = smallest captured size >= the busiest rank's `#running-req`. That all ranks pad to the busiest rank is taken from the task statement; the log only shows `cuda graph: True` (it is True on every line of every run, and the busiest rank never exceeds the largest captured size). Indirect support from the logs ('Padding check' in the detail sections): above 8, where only even sizes are captured, the step time grows about 0.3-0.7 ms from busiest 2k-1 to 2k (same graph) and about 1.0-1.7 ms from 2k to 2k+1 (next graph), in every run. It is not zero inside a pair, so either the step time also depends on the real request count or the +-20-step alignment blurs it; the logs cannot separate the two.")
    A("* **Alignment**: each `Decode batch` line of rank r is one rank-tick at its timestamp (Docker nanosecond prefix where present). The other ranks' running counts at that tick are taken from their log line nearest in time, before or after; if a rank has no line within `tol` = max(1.5 x median line gap, 1 s) it is counted idle (0 running). Ranks log at different phases of the 40-step cycle, so the nearest line is at most about 20 steps away. Every rank's lines are used as ticks; each is weighted 1 / (#ranks that logged within tol) so that 40 engine steps count once.")
    A("* **Parser / alignment check (2026-11-12)**: `--check runs-n04-33/c208-1109T1152-kopt-pp8` prints 5 ticks spread over the window with the raw line used for every rank. In all 5 the per-rank counts, total, busiest and graph batch equal what the raw lines say, and the nearest lines are within 0.72 s (half a 40-step interval). One of them (12:45:26.449, DP5) was re-derived independently with grep/awk on the raw log: [9, 8, 5, 8, 8, 9, 3, 7], total 57, busiest 9, graph batch 10. DP0 and DP2 log within 0.2 ms of each other in every sampled tick, which fits all ranks stepping together. Gaps in a rank's lines (DP3, DP7 in that run) have `#running-req: 1` before and after, which fits 'no line = idle'.")
    A("* **Weightings**: *ticks* = that tick weight (~ engine steps). *requests* = tick weight x total running at the tick (~ request-steps, i.e. output tokens; this is what per-request interactivity sees). *time* = tick weight x the rank's line gap (wall time spent at that graph batch; lines with gap > 10 s dropped).")
    A("* **Ranges**: '80% range' = the contiguous run of captured batch sizes with the fewest distinct sizes holding >= 80% of the weight (ties: most weight); the share actually covered is in parentheses.")
    A("* **Step time** = gap between consecutive lines of the same rank / 40. 'stable' = ticks whose graph batch was the same at the previous line of that rank. It is wall time per engine step, so it includes whatever the scheduler does between forwards. `opt/partial/c208-1110T2109` has no Docker timestamp prefix (1 s resolution only); for it the step time is 40 x running x accept-len / logged throughput on ticks with unchanged running count. In the other logs that estimate agrees with the timestamp gaps to a few tenths of a ms per graph batch (both columns are in the detail tables).")
    A("* **Link to the client score**: accept length / request-weighted step time (p50 and p90) reproduces the client interactivity p50 / p90 of every scored run to within about 2 tok/s ('TP4/DP4 sweep' table), so the step-time and weighting logic is consistent with what AgentX measured.")
    A("* **#token**: `max_total_num_tokens - (available + evictable)` of that rank's KV pool (`pool_stats_observer.py::_get_token_info`), i.e. KV slots of running requests plus requests in the transfer queue (their KV is pre-allocated: `pre-allocated usage` sums `extend_end` over the transfer queue, `disaggregation/decode.py::num_tokens_pre_allocated`). Requests in `#prealloc-req` hold no KV yet. So context per request is reported as #token / (running + transfer + queue) ('per holder'); the task's formula #token / (running + prealloc + transfer) and a running-only estimate (#token - pre-allocated usage x max_total_num_tokens) / running (usage is logged to 2 decimals, so +-0.5% of the pool) are in the JSON and detail tables. These are per-rank averages over requests, not per-request values. The source tree read is the kreview copy; it was not checked to be byte-identical to the images that produced the logs.")
    A("* **Window**: AgentX `Phase profiling (profiling) started` .. `sending complete` (the same markers `score_summary.sh` uses), taken from the run's own agentx log / `benchmark.log`. Decode activity starts at the window start in every run (warm-up produces no decode batches), so the ramp-up of running requests is inside the window. Per-run window source is in the detail section.\n")

    A("## Envelope by configuration (complete runs; min of the range starts .. max of the range ends over the runs of the group)\n")
    hdr = ["decode", "K (rows/req)", "C", "DP-bal", "engines", "total run p50", "busiest p50 / p90 / p99", "gbs p50 ticks / req", "gbs 80% ticks", "gbs 95% ticks", "gbs 80% req", "gbs 95% req",
           "attn rows/rank 95% req", "MoE M 80% req", "MoE M 95% req", "fill", "ctx/req p50 K first / last 10 min"]
    A("| " + " | ".join(hdr) + " |")
    A("|" + "---|" * len(hdr))
    grp = {}
    for r in ok:
        if r.get("window_note"):
            continue
        s = r["stats"]
        grp.setdefault((-s["n_ranks"], r["K"], r.get("conc") or 0, "off" if r["dpbal"].startswith("off") else "on", s["rows_per_req"]), []).append(r)

    def env(rs, w, f, mult=1):
        cs = [r["stats"]["graph_bs_cover"][w][f] for r in rs]
        lo, hi = min(c[0] for c in cs) * mult, max(c[1] for c in cs) * mult
        return f"{lo}-{hi}" if lo != hi else f"{lo}"

    def span(vals, nd=0):
        lo, hi = min(vals), max(vals)
        return f"{lo:.{nd}f}" if lo == hi else f"{lo:.{nd}f}-{hi:.{nd}f}"

    for k in sorted(grp):
        rs = grp[k]
        n, rw = -k[0], k[4] or 0
        st_ = [r["stats"] for r in rs]
        A("| " + " | ".join([
            f"TP{n}/DP{n}", f"{k[1]} ({rw})", str(k[2]), k[3], str(len(rs)), span([x["total_running"]["p50"] for x in st_]),
            " / ".join(span([x["busiest"][q] for x in st_]) for q in ("p50", "p90", "p99")),
            span([x["graph_bs_cover"]["by_ticks"]["p50"] for x in st_]) + " / " + span([x["graph_bs_cover"]["by_requests"]["p50"] for x in st_]),
            env(rs, "by_ticks", "80"), env(rs, "by_ticks", "95"), env(rs, "by_requests", "80"), env(rs, "by_requests", "95"),
            env(rs, "by_requests", "95", rw), env(rs, "by_requests", "80", rw * n), env(rs, "by_requests", "95", rw * n),
            span([x["fill_ratio_of_sums"] for x in st_], 2),
            span([x["context"]["first10min"]["tok_per_holder"]["p50"] / 1e3 for x in st_]) + " / " + span([x["context"]["last10min"]["tok_per_holder"]["p50"] / 1e3 for x in st_]),
        ]) + " |")
    A("")
    A("## Summary (one row per decode engine)\n")
    A("Columns: per-rank / total / busiest are `#running-req`; `gbs` = implied captured graph batch; ranges give the share covered in parentheses; `fill` = real rows / padded rows (ratio of sums); `step ms` = wall time per engine step; `ctx/req` = #token per KV-holding request. p-values are tick-weighted unless marked req. Scores are read from each run's `agentx/agentx_conc*.json`. Group A = TP8/DP8 decode on its own node; B = TP4/DP4 decode (runs asked for); B-extra = other TP4 C104 runs found next to them. TRUNCATED = the profiling phase did not finish (crash / abort / partial copy), no score.\n")
    hdr = ["run", "grp", "layout", "K (rows/req)", "C", "DP-bal", "window min", "tok/s/GPU", "out tok/s/GPU", "intvty p50/p90", "TTFT p50/p90 s",
           "total run p50/p90", "per-rank p10/p50/p90/max", "busiest p10/p50/p90/p99/max", "busiest/mean p50/p90",
           "gbs 80% ticks", "gbs 95% ticks", "gbs 80% req", "gbs 95% req", "fill", "step ms p50", "ctx/req p50 K"]
    A("| " + " | ".join(hdr) + " |")
    A("|" + "---|" * len(hdr))
    order = lambda r: (r["group"], bool(r.get("window_note")), r["stats"]["n_ranks"] * -1, r.get("K") or 0, r.get("conc") or 0, r["dpbal"] != "off" and not r["dpbal"].startswith("off"), r["key"])  # noqa
    for r in sorted(ok, key=order):
        s, sj = r["stats"], r.get("score_json") or {}
        c = s["graph_bs_cover"]
        A("| " + " | ".join([
            short(r["key"]) + (" [" + ",".join(r["tags"]) + "]" if r["tags"] else "") + (" **TRUNCATED**" if r.get("window_note") else ""), r["group"], r["layout"],
            f"{r['K']} ({s['rows_per_req']})", str(r.get("conc")), r["dpbal"].split(" ")[0], f"{s['window_s'] / 60:.0f}",
            fmt(sj.get("tok_s_gpu"), 0), fmt(sj.get("out_tok_s_gpu"), 1),
            f"{fmt(sj.get('intvty_p50'), 1)} / {fmt(sj.get('intvty_p90'), 1)}", f"{fmt(sj.get('ttft_p50'), 2)} / {fmt(sj.get('ttft_p90'), 2)}",
            f"{s['total_running']['p50']} / {s['total_running']['p90']}",
            "/".join(str(s["per_rank_running_pooled"][k]) for k in ("p10", "p50", "p90", "max")),
            "/".join(str(s["busiest"][k]) for k in ("p10", "p50", "p90", "p99", "max")),
            f"{s['busiest_over_mean']['p50']:.2f} / {s['busiest_over_mean']['p90']:.2f}",
            rng(c["by_ticks"]["80"]), rng(c["by_ticks"]["95"]), rng(c["by_requests"]["80"]), rng(c["by_requests"]["95"]),
            f"{s.get('fill_ratio_of_sums', 0):.2f}", fmt(s["step_ms"]["p50"], 1),
            fmt((s["context"]["window"]["tok_per_holder"]["p50"] or 0) / 1e3, 0),
        ]) + " |")
    A("")
    A("## Shapes (target-verify graph): attention rows per rank = rows/req x gbs; MoE tokens M = n_ranks x rows/req x gbs\n")
    hdr = ["run", "ranks x rows/req", "gbs p50 ticks / req", "attn rows 80% ticks", "attn rows 95% ticks", "attn rows 80% req", "attn rows 95% req",
           "M 80% ticks", "M 95% ticks", "M 80% req", "M 95% req", "M 80% time"]
    A("| " + " | ".join(hdr) + " |")
    A("|" + "---|" * len(hdr))
    for r in sorted(ok, key=order):
        s = r["stats"]
        c, rw, n = s["graph_bs_cover"], s["rows_per_req"] or 0, s["n_ranks"]
        A("| " + " | ".join([
            short(r["key"]), f"{n} x {rw}", f"{c['by_ticks']['p50']} / {c['by_requests']['p50']}",
            shape_rng(c["by_ticks"]["80"], rw), shape_rng(c["by_ticks"]["95"], rw), shape_rng(c["by_requests"]["80"], rw), shape_rng(c["by_requests"]["95"], rw),
            shape_rng(c["by_ticks"]["80"], rw * n), shape_rng(c["by_ticks"]["95"], rw * n), shape_rng(c["by_requests"]["80"], rw * n),
            shape_rng(c["by_requests"]["95"], rw * n), shape_rng(c["by_time"]["80"], rw * n),
        ]) + " |")
    A("")
    A("Other captured graphs seen in the logs (the draft model runs them inside the same step; how often each runs per step was not derived here): "
      + "; ".join(sorted({f"TP{r['stats']['n_ranks']} K={r['K']}: " + ", ".join(f"{g} {t} row(s)/req" for g, t in sorted(r['rows_per_req_by_graph'].items())) for r in ok})) + "\n")

    A("## TP4/DP4 sweep by concurrency (complete runs only), and TP8/DP8 runs\n")
    hdr = ["K", "C", "DP-bal", "run", "out tok/s/GPU", "intvty p50/p90", "total run p50", "busiest p50/p90/p99", "gbs p50 ticks/req", "gbs 80% ticks", "gbs 95% ticks", "gbs 80% req", "gbs 95% req",
           "M 95% req", "fill", "step ms p50/p90 (req-weighted)", "accept_len/step: p50 / p90-slow"]
    for title, sel in (("TP4/DP4", lambda r: r["stats"]["n_ranks"] == 4), ("TP8/DP8", lambda r: r["stats"]["n_ranks"] == 8)):
        A(f"{title}:\n")
        A("| " + " | ".join(hdr) + " |")
        A("|" + "---|" * len(hdr))
        for r in sorted([r for r in ok if sel(r) and (not r.get("window_note") or r["stats"]["n_ranks"] == 8)],
                        key=lambda r: (r.get("K") or 0, r.get("conc") or 0, not r["dpbal"].startswith("off"), r["key"])):
            s, sj = r["stats"], r.get("score_json") or {}
            c, rw, n = s["graph_bs_cover"], s["rows_per_req"] or 0, s["n_ranks"]
            ii = s.get("implied_intvty") or {}
            A("| " + " | ".join([
                str(r["K"]), str(r.get("conc")), r["dpbal"].split(" ")[0], short(r["key"]) + (" TRUNCATED" if r.get("window_note") else ""), fmt(sj.get("out_tok_s_gpu"), 1),
                f"{fmt(sj.get('intvty_p50'), 1)} / {fmt(sj.get('intvty_p90'), 1)}", str(s["total_running"]["p50"]),
                "/".join(str(s["busiest"][k]) for k in ("p50", "p90", "p99")), f"{c['by_ticks']['p50']} / {c['by_requests']['p50']}",
                rng(c["by_ticks"]["80"]), rng(c["by_ticks"]["95"]), rng(c["by_requests"]["80"]), rng(c["by_requests"]["95"]),
                shape_rng(c["by_requests"]["95"], rw * n), f"{s.get('fill_ratio_of_sums', 0):.2f}",
                f"{fmt(s['step_ms_req_weighted']['p50'], 1)} / {fmt(s['step_ms_req_weighted']['p90'], 1)}",
                f"{fmt(ii.get('p50'), 1)} / {fmt(ii.get('p90_slow'), 1)}",
            ]) + " |")
        A("")
    A("`accept_len/step` is what a request would get if it sat on steps of the request-weighted p50 / p90 duration for its whole life; it is an upper-bound style cross-check of the client interactivity columns, not a replacement (a request averages over many steps, and the client number includes streaming and network).\n")

    A("## Step ms by implied graph batch (median over ticks whose graph batch did not change since the rank's previous line; blank = fewer than 40 such ticks)\n")
    sizes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    A("| run | K | " + " | ".join(str(g) for g in sizes) + " |")
    A("|---|---|" + "---|" * len(sizes))
    for r in sorted(ok, key=order):
        s = r["stats"]
        if r.get("window_note") and s["n_ranks"] == 4:
            continue
        col = "thr" if r["second_resolution_timestamps"] else "stable"
        cells = []
        for g in sizes:
            v = s["step_ms_by_graph_bs"].get(str(g), {})
            cells.append(f"{v[col + '_median']:.1f}" if v.get(col + "_n", 0) >= 40 else "")
        A(f"| {short(r['key'])} | {r['K']} | " + " | ".join(cells) + " |")
    A("")

    A("Median step-time increment between neighbouring captured sizes over the engines above that have both sizes (ms; min .. max; n engines):\n")
    trans = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), (22, 24), (24, 26)]
    A("| K (rows/req) | " + " | ".join(f"{a} to {b}" for a, b in trans) + " |")
    A("|---|" + "---|" * len(trans))
    for K in sorted({r["K"] for r in ok}):
        cells = []
        for a, b in trans:
            v = []
            for r in ok:
                if r["K"] != K or r["second_resolution_timestamps"] or (r.get("window_note") and r["stats"]["n_ranks"] == 4):
                    continue
                m = r["stats"]["step_ms_by_graph_bs"]
                if m.get(str(a), {}).get("stable_n", 0) >= 40 and m.get(str(b), {}).get("stable_n", 0) >= 40:
                    v.append(m[str(b)]["stable_median"] - m[str(a)]["stable_median"])
            cells.append(f"{st.median(v):+.2f} ({min(v):+.1f} .. {max(v):+.1f}; {len(v)})" if v else "")
        rw = next(r["stats"]["rows_per_req"] for r in ok if r["K"] == K)
        A(f"| {K} ({rw}) | " + " | ".join(cells) + " |")
    A("")

    # comparison with score files
    A("## Cross-check against the `score-*.txt` lines\n")
    A("The score-file decode numbers come from analyze_pd_logs.py, which bins by 10 s of wall time (last line per rank in the bin, busiest = max over the ranks seen). Mine are per 40 engine steps. Wall-time weighting gives more weight to large batches (their steps are slower), so the score-file medians sit 1-2 above the step-weighted ones; the time-weighted column shows that this accounts for the difference.\n")
    hdr = ["run", "score file", "window agrees", "tok/s/GPU json / score", "total run p50 step-wtd / time-wtd / score", "busiest p50 step-wtd / time-wtd / score", "busiest p90 step-wtd / time-wtd / score",
           "step ms by busiest: max abs diff (n sizes)", "largest diffs @busiest mine/score"]
    A("| " + " | ".join(hdr) + " |")
    A("|" + "---|" * len(hdr))
    by_base = {}
    for sc in scores:
        if sc.get("run_base"):
            by_base.setdefault(sc["run_base"], []).append(sc)
    for r in ok:
        base = os.path.basename(r["run"])
        scs = by_base.get(base) or by_base.get(re.sub(r"-kopt-pp8$", "", base)) or []
        scs = [x for x in scs if ("/v2/" in x.get("run", "")) == r["run"].startswith("v2/")]
        for sc in scs:
            s, sj = r["stats"], r.get("score_json") or {}
            wa = "n/a"
            if sc.get("w0") and len(sc["w0"]) >= 19:
                wa = "yes" if (sc["w0"][11:19] == s["window"][0][11:19]) else f"NO ({sc['w0']} vs {s['window'][0]})"
            diffs = []
            for b, v in (sc.get("step_ms") or {}).items():
                mine = s["step_ms_by_busiest"].get(str(b))
                if mine and mine["n"] >= 30:
                    diffs.append((abs(mine["median"] - v), b, mine["median"], v))
            diffs.sort(reverse=True)
            A("| " + " | ".join([
                short(r["key"]), sc["label"], wa, f"{fmt(sj.get('tok_s_gpu'), 0)} / {fmt(sc.get('tok_s_gpu'), 0)}",
                f"{s['total_running']['p50']} / {s['total_running_time_weighted_p50']} / {sc.get('tot_med', 'n/a')}",
                f"{s['busiest']['p50']} / {s['busiest_time_weighted']['p50']} / {sc.get('busy_med', 'n/a')}",
                f"{s['busiest']['p90']} / {s['busiest_time_weighted']['p90']} / {sc.get('busy_p90', 'n/a')}",
                (f"{diffs[0][0]:.1f} ({len(diffs)})" if diffs else "n/a"),
                ", ".join(f"@{b} {m:.1f}/{v:.0f}" for _, b, m, v in diffs[:3]) or "n/a",
            ]) + " |")
    A("")

    A("## All score files (summary row + decode lines as written in the file; decode log availability)\n")
    have = {os.path.basename(r["run"]): r for r in rows}
    hdr = ["score file", "run", "window", "C", "tok/s/GPU", "out tok/s/GPU", "intvty p50/p90", "ITL p50/p90 ms", "TTFT p50/p90/mean s",
           "decode: total med / busiest med / p90", "decode log analysed here"]
    A("| " + " | ".join(hdr) + " |")
    A("|" + "---|" * len(hdr))
    for sc in scores:
        base = sc.get("run_base", "")
        r = have.get(base) or have.get(base + "-kopt-pp8")
        if r and ("/v2/" in sc.get("run", "")) != r["run"].startswith("v2/"):
            r = None
        A("| " + " | ".join([
            sc["label"], base, f"{sc.get('w0', '')} .. {sc.get('w1', '')}", str(sc.get("conc", "n/a")), fmt(sc.get("tok_s_gpu"), 0), fmt(sc.get("out_tok_s_gpu"), 1),
            f"{fmt(sc.get('intvty_p50'), 1)} / {fmt(sc.get('intvty_p90'), 1)}", f"{fmt(sc.get('itl_p50_ms'), 1)} / {fmt(sc.get('itl_p90_ms'), 1)}",
            f"{fmt(sc.get('ttft_p50'), 2)} / {fmt(sc.get('ttft_p90'), 2)} / {fmt(sc.get('ttft_mean'), 2)}",
            (f"{sc['tot_med']} / {sc['busy_med']} / {sc['busy_p90']}" if "tot_med" in sc else "n/a (scoring failed or line missing)"),
            (short(r["key"]) if r and "stats" in r else ("log present, skipped" if r else "no (log not kept)")),
        ]) + " |")
    A("")
    A("`decode step ms by busiest-rank running` lines of the score files:\n")
    A("```")
    for sc in scores:
        if sc.get("step_line"):
            A(f"{sc['label']:34s} {sc['step_line'][len('decode step ms by busiest-rank running: '):]}")
    A("```\n")

    sk = [r for r in rows if "skipped" in r]
    if sk:
        A("## Skipped\n")
        for r in sk:
            A(f"* `{r.get('key', r['run'])}`: {r['skipped']}")
        A("")

    A("## Per-run detail\n")
    for r in sorted(ok, key=lambda r: (r["group"], r["key"])):
        s, sj = r["stats"], r.get("score_json") or {}
        A(f"### {r['key']}\n")
        A(f"* group {r['group']}; {r['layout']}; K={r['K']} (draft tokens {r['server_args'].get('speculative_num_draft_tokens')}, target-verify rows/req {s['rows_per_req']}, SIM_ACCEPT_LEN {r.get('sim_accept_len')}, logged accept len req-weighted {s['accept_len_req_weighted']}); C={r.get('conc')}; DP balance: {r['dpbal']}; tags: {', '.join(r['tags']) or '-'}; image {r.get('image')}" + (f"; note: {r['note']}" if r["note"] else ""))
        A(f"* decode log `{r['decode_log']}` ({r['log_bytes'] / 1e6:.1f} MB, {r['decode_lines']} Decode batch lines, {r['unparsed_decode_lines']} unparsed" + ("; **second-resolution timestamps only**" if r["second_resolution_timestamps"] else "") + ")")
        A(f"* window {s['window'][0]} .. {s['window'][1]} UTC ({s['window_s'] / 60:.1f} min) from {r['window_source']}" + (f"; **{r['window_note']}**" if r["window_note"] else ""))
        A(f"* {s['rank_ticks']} rank-ticks = {s['engine_ticks']} engine ticks = about {s['engine_steps_est']:,} engine steps; lines per rank in window: " + ", ".join(f"DP{k} {v}" for k, v in s["lines_per_rank"].items()) + f"; median line gap {s['median_gap_s']} s, alignment tol {s['align_tol_s']} s; ticks with at least one idle rank {100 * s['ticks_with_idle_rank_frac']:.1f}%; `cuda graph: False` lines {100 * s['cuda_graph_false_frac']:.2f}%; busiest above the largest captured size {100 * s['over_capture_frac']:.2f}% of ticks")
        if s.get("steps_per_tick_check"):
            k = s["steps_per_tick_check"]
            A(f"* steps/tick check (1.0 = {s['steps_per_tick']} steps per line): p10 {k['p10']}, p50 {k['p50']}, p90 {k['p90']} over {k['n']} ticks")
        if sj and "error" not in sj:
            A(f"* score (`{sj['source']}`): {fmt(sj['tok_s_gpu'], 0)} tok/s/GPU, {fmt(sj['out_tok_s_gpu'], 1)} out tok/s/GPU, {fmt(sj['req_s'], 3)} req/s, intvty p50/p90 {fmt(sj['intvty_p50'], 1)}/{fmt(sj['intvty_p90'], 1)}, ITL p50/p90 {fmt(sj['itl_p50_ms'], 1)}/{fmt(sj['itl_p90_ms'], 1)} ms, TTFT p50/p90/mean {fmt(sj['ttft_p50'], 2)}/{fmt(sj['ttft_p90'], 2)}/{fmt(sj['ttft_mean'], 2)} s; request input tokens mean/p50/p90 {fmt(sj['input_tok_mean'], 0)}/{fmt(sj['input_tok_p50'], 0)}/{fmt(sj['input_tok_p90'], 0)}, output mean {fmt(sj['output_tok_mean'], 0)}")
        else:
            A("* score: no `agentx/agentx_conc*.json` in the run directory (run did not finish or the copy is partial)")
        A("")
        A("Running requests (tick-weighted; per-rank values use the aligned snapshot, idle = 0):\n")
        A("| | p10 | p50 | p90 | p99 | max |")
        A("|---|---|---|---|---|---|")
        for name, d in (("per rank (pooled)", s["per_rank_running_pooled"]), ("total across ranks", s["total_running"]), ("busiest rank", s["busiest"]), ("busiest rank, request-weighted", s["busiest_req_weighted"])):
            A(f"| {name} | {d['p10']} | {d['p50']} | {d['p90']} | {d['p99']} | {d['max']} |")
        A("")
        A("Per rank p10/p50/p90/max: " + "; ".join(f"DP{k} {v['p10']}/{v['p50']}/{v['p90']}/{v['max']}" for k, v in s["per_rank_running"].items()))
        A(f"  \nMeans: total {s['total_running_mean']}, busiest {s['busiest_mean']}; busiest/mean-rank ratio p50 {s['busiest_over_mean']['p50']}, p90 {s['busiest_over_mean']['p90']}.")
        A(f"  \nReal rows / padded rows: mean over ticks {s.get('fill_mean_of_ticks')}, ratio of sums {s.get('fill_ratio_of_sums')} (if graphs existed for every batch size: {s.get('fill_if_exact_busiest')}).\n")
        c = s["graph_bs_cover"]
        A("Implied graph batch, share and cumulative share, plus step time:\n")
        A("| gbs | attn rows/rank | MoE M | ticks % | cum % | requests % | cum % | time % | cum % | step ms stable median (n) | step ms all (n) | step ms from throughput (n) |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|")
        H = s["graph_bs_hist"]
        rw, n = s["rows_per_req"] or 0, s["n_ranks"]
        for g in sorted(H["by_ticks"], key=int):
            t, q, tm = H["by_ticks"][g], H["by_requests"].get(g, [0, 0]), H["by_time"].get(g, [0, 0])
            sm = s["step_ms_by_graph_bs"].get(g, {})
            lab = g if not s["capture_list"] or int(g) <= s["capture_list"][-1] else f">{s['capture_list'][-1]}"
            A(f"| {lab} | {int(g) * rw} | {int(g) * rw * n} | {100 * t[0]:.1f} | {100 * t[1]:.1f} | {100 * q[0]:.1f} | {100 * q[1]:.1f} | {100 * tm[0]:.1f} | {100 * tm[1]:.1f} | "
              + ("n/a (1 s timestamps) | n/a | " if r["second_resolution_timestamps"] else f"{fmt(sm.get('stable_median'), 1)} ({sm.get('stable_n', 0)}) | {fmt(sm.get('all_median'), 1)} ({sm.get('all_n', 0)}) | ")
              + f"{fmt(sm.get('thr_median'), 1)} ({sm.get('thr_n', 0)}) |")
        A("")
        A(f"Ranges: ticks 80% {rng(c['by_ticks']['80'])}, 95% {rng(c['by_ticks']['95'])}; requests 80% {rng(c['by_requests']['80'])}, 95% {rng(c['by_requests']['95'])}; time 80% {rng(c['by_time']['80'])}, 95% {rng(c['by_time']['95'])}.\n")
        A(f"Step ms ({s['step_ms']['source']}): tick-weighted p10/p50/p90 {s['step_ms']['p10']}/{s['step_ms']['p50']}/{s['step_ms']['p90']}, request-weighted p10/p50/p90 {s['step_ms_req_weighted']['p10']}/{s['step_ms_req_weighted']['p50']}/{s['step_ms_req_weighted']['p90']}"
          + (f"; accept_len / step = {s['implied_intvty']['p50']} tok/s at the request-weighted p50 step and {s['implied_intvty']['p90_slow']} at the p90 step (client interactivity p50/p90: {fmt(sj.get('intvty_p50'), 1)}/{fmt(sj.get('intvty_p90'), 1)})" if s.get("implied_intvty") else "") + ".\n")
        A("Step ms by exact busiest-rank running (median, all ticks, n >= 30): " + ", ".join(f"@{b} {v['median']:.1f}" for b, v in s["step_ms_by_busiest"].items() if v["n"] >= 30) + "\n")
        A("Same, only ticks whose busiest count was unchanged since the rank's previous line (n >= 30): " + ", ".join(f"@{b} {v['median']:.1f}" for b, v in s["step_ms_by_busiest_stable"].items() if v["n"] >= 30))
        pt = s["padding_pair_test"]
        A(f"  \nPadding check: mean step-time increment from busiest 2k-1 to 2k (same captured graph, k >= 5) {pt['inside_pair_mean']} ms over {pt['inside_n']} pairs; from 2k to 2k+1 (next captured graph) {pt['across_pair_mean']} ms over {pt['across_n']} pairs.\n")
        cx = s["context"]
        A(f"Context / KV per rank (max_total_num_tokens per rank {cx['max_total_num_tokens_per_rank']:,}):\n" if cx["max_total_num_tokens_per_rank"] else "Context / KV per rank:\n")
        A("| period | #token per holder (run+transfer+queue) p10/p50/p90 | #token per (run+prealloc+transfer) p50 | running-only estimate p50 (req-weighted) | KV tokens per rank p10/p50/p90 | KV usage p50/p90 | mean running / transfer / prealloc per rank |")
        A("|---|---|---|---|---|---|---|")
        for name, key in (("whole window", "window"), ("first 10 min", "first10min"), ("last 10 min", "last10min")):
            d = cx[key]
            if not d["n_lines"]:
                continue
            A(f"| {name} | {fmt(d['tok_per_holder']['p10'])} / {fmt(d['tok_per_holder']['p50'])} / {fmt(d['tok_per_holder']['p90'])} | {fmt(d['tok_per_run_prealloc_transfer']['p50'])} | {fmt(d['tok_per_running_reqw']['p50'])} | "
              f"{fmt(d['kv_tokens_per_rank']['p10'])} / {fmt(d['kv_tokens_per_rank']['p50'])} / {fmt(d['kv_tokens_per_rank']['p90'])} | {fmt(d['kv_usage']['p50'], 2)} / {fmt(d['kv_usage']['p90'], 2)} | {d['running_mean']} / {d['transfer_mean']} / {d['prealloc_mean']} |")
        A("")
        A("Evolution in 10-minute slices: " + "; ".join(
            f"+{e['from_min']}m total {e['total_running_p50']} busiest {e['busiest_p50']}/{e['busiest_p90']} gbs {e['graph_bs_p50']} step {fmt(e['step_ms_p50'], 1)}ms ctx {fmt((e['tok_per_holder_p50'] or 0) / 1e3, 0)}K KV/rank {fmt((e['kv_tokens_per_rank_p50'] or 0) / 1e6, 2)}M usage-p90 {fmt(e['kv_usage_p90'], 2)}"
            for e in s["evolution_10min"]) + "\n")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def check(rel, n):
    """Print n aligned ticks with the raw lines they were built from."""
    run_dir = os.path.join(ROOT, rel.split(":")[0])
    logs = sorted(glob.glob(os.path.join(run_dir, "logs", "*-decode*.log")))
    lg = logs[0] if ":" not in rel else [x for x in logs if x.endswith("-" + rel.split(":")[1] + ".log")][0]
    meta, ranks = parse_decode_log(lg, keep_raw=True)
    t0, t1, src = find_window(run_dir, meta)
    if t0 is None:
        t0, t1, _ = plateau_window(ranks)
        src = "plateau"
    t1 = t1 or meta["t_last"]
    res, ticks = analyze(meta, ranks, t0, t1)
    print(f"log {lg}\nwindow {iso(t0)} .. {iso(t1)} ({src}); N={res['steps_per_tick']} tol={res['align_tol_s']} s capture={res['capture_list']}")
    times = {r: [x[0] for x in ranks[r]] for r in ranks}
    step = max(1, len(ticks) // (n + 1))
    for k in range(1, n + 1):
        x = ticks[k * step]
        print(f"\n--- tick #{k * step}: t={iso(x['t'])}.{int(x['t'] % 1 * 1e6):06d} logged by DP{x['r']}  runs={x['runs']} active={x['act']} busiest={x['busiest']} total={x['total']} "
              f"graph_bs={x['gbs']} gap={x['gap']} step_ms={x['gap'] / res['steps_per_tick'] * 1e3 if x['gap'] else None} weight={x['w']:.3f}")
        for q in sorted(ranks):
            j = bisect.bisect_left(times[q], x["t"])
            cands = [c for c in (j - 1, j) if 0 <= c < len(times[q])]
            b = min(cands, key=lambda c: abs(times[q][c] - x["t"]))
            print(f"   DP{q} nearest line dt={times[q][b] - x['t']:+.3f}s: {ranks[q][b][-1][:118]}")
        i = bisect.bisect_left(times[x['r']], x["t"])
        if i > 0:
            print(f"   previous line of DP{x['r']}: {ranks[x['r']][i - 1][-1][:118]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check")
    ap.add_argument("--check-n", type=int, default=5)
    ap.add_argument("--only", help="substring filter on run path; prints a short table, writes nothing")
    ap.add_argument("-j", type=int, default=3)
    a = ap.parse_args()
    if a.check:
        return check(a.check, a.check_n)
    scores = read_score_files()
    for sc in scores:
        if sc.get("run_base") and "/v2/" not in sc.get("run", ""):
            job_scores.setdefault(sc["run_base"], sc)
    jobs = discover()
    if a.only:
        jobs = [j for j in jobs if a.only in j[1]]
    with Pool(min(4, max(1, a.j))) as p:
        rows = [r for rs in p.map(do_run, jobs, chunksize=1) for r in rs]
    if a.only:
        for r in rows:
            if "stats" in r:
                s = r["stats"]
                print(r["key"], s["window"], "total", s["total_running"], "busiest", s["busiest"], "cover", s["graph_bs_cover"]["by_ticks"], s["graph_bs_cover"]["by_requests"],
                      "fill", s.get("fill_ratio_of_sums"), "step", s["step_ms"], "chk", s.get("steps_per_tick_check"))
            else:
                print(r.get("key", r["run"]), "SKIPPED", r["skipped"])
        return
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "batch_dist.json"), "w") as fh:
        json.dump({"command": CMD, "runs": rows, "score_files": scores}, fh, indent=1)
    write_md(rows, scores, os.path.join(OUT, "BATCH-DIST.md"))
    print(f"wrote {OUT}/BATCH-DIST.md and batch_dist.json: {sum(1 for r in rows if 'stats' in r)} engines analysed, {sum(1 for r in rows if 'skipped' in r)} skipped")


if __name__ == "__main__":
    main()
