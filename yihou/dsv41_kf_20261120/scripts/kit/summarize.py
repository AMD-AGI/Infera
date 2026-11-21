#!/usr/bin/env python3
"""Summarize one AgentX point and append one CSV line to the ledger. Stdlib only (python >= 3.8).

Usage: summarize.py <run dir> [--result-json F] [--artifacts D] [--gpus N] [--mode official|fast|unknown]
                    [--tag T] [--ledger CSV | --no-ledger] [--json]

<run dir> is runs/<tag> (layout of run_point.sh: bench/{inferencex_result.json,aiperf_artifacts/,metrics_*.txt,
server.log,manifest.json}) or any dir holding aiperf_artifacts/ (colleague layouts: use --result-json/--artifacts).

Reports (sources in brackets):
  score       tok/s/GPU from the harness [inferencex_result.json request_metrics.throughput.per_gpu.total_tput_tps] and
              recomputed from raw records [profile_export.jsonl: profiling-phase, error-free records;
              (sum input_sequence_length incl. cached prefix + sum output_sequence_length) / (max end - min start) / GPUs],
              plus the same with server-reported usage_prompt/completion_tokens; relative diffs printed.
  qps, mean ISL/OSL, TTFT p50/p90, ITL p50/p90, p90 interactivity [harness; TTFT p50/p90 also recomputed raw]
  prefix hit  [aiperf server_metrics_export.json, profiling-phase counter totals: vllm:prefix_cache_hits/queries, external_*,
              prompt_tokens_by_source] and [/metrics text snapshot deltas before->after, which include warmup]
  acceptance  AL = 1 + d(vllm:spec_decode_num_accepted_tokens)/d(vllm:spec_decode_num_drafts) (both sources), per-position rates
  KV usage    vllm:kv_cache_usage_perc gauge avg/p90/max (max over engines); KV pool tokens from server.log
  contract    input_config: duration 3600, warmup_requests_per_lane 10, random_seed 42, dataset, scenario agentx; error ratio
"""
import argparse
import csv
import datetime as dt
import fcntl
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

DATASET = "semianalysis_cc_traces_weka_062126"
LEDGER_DEFAULT = Path(__file__).resolve().parents[2] / "results" / "ledger.csv"
COLUMNS = [
    "utc", "tag", "mode", "contract_ok", "conc", "gpus", "score_harness", "score_raw", "raw_vs_harness_pct",
    "score_usage", "qps", "isl_mean", "osl_mean", "ttft_p50_s", "ttft_p90_s", "itl_p50_ms", "itl_p90_ms",
    "intvty_p90", "hit_gpu", "hit_pq", "hit_ext", "hit_theo", "hit_gpu_snap", "al", "al_snap", "kv_use_avg", "kv_use_p90",
    "kv_use_max", "kv_pool_tokens", "running_avg", "waiting_avg", "preemptions", "ok", "total", "errors",
    "duration_s", "image", "dir", "notes",
]


def g(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (k - lo)


def rel(a, b):
    if a is None or b in (None, 0):
        return None
    return (a - b) / b * 100.0


def first_existing(cands):
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def find_inputs(run, a):
    run = Path(run)
    arts = Path(a.artifacts) if a.artifacts else first_existing(
        [run / "bench/aiperf_artifacts", run / "aiperf_artifacts", run / "agentic/aiperf_artifacts",
         run / "agentic_raw/aiperf_artifacts"])
    if arts is None:
        hits = sorted(glob.glob(str(run / "**/aiperf_artifacts/profile_export.jsonl"), recursive=True))
        arts = Path(hits[0]).parent if hits else None
    res = Path(a.result_json) if a.result_json else first_existing([run / "bench/inferencex_result.json",
                                                                     run / "inferencex_result.json"])
    if res is None:
        for d in (run / "bench", run, run.parent):
            for f in sorted(d.glob("*.json")) if d.is_dir() else []:
                try:
                    if f.stat().st_size < 5_000_000 and '"request_metrics"' in f.read_text(errors="ignore"):
                        res = f
                        break
                except OSError:
                    continue
            if res:
                break
    bench = run / "bench" if (run / "bench").is_dir() else run
    return arts, res, bench


def raw_records(jsonl):
    """Stream profile_export.jsonl; same selection as InferenceX artifacts.load_records_with_accounting."""
    r = dict(total=0, warmup=0, errors=0, n=0, isl=0, osl=0, up=0, uc=0, start=None, end=None, ttft=[])
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            r["total"] += 1
            md = o.get("metadata", {})
            phase = md.get("benchmark_phase")
            warm = phase is not None and phase != "profiling"
            r["warmup"] += warm
            if o.get("error"):
                r["errors"] += 1
            if o.get("error") or warm:
                continue
            m = o.get("metrics", {})

            def v(k):
                x = m.get(k)
                return x.get("value") if isinstance(x, dict) else x
            r["n"] += 1
            r["isl"] += int(v("input_sequence_length") or 0)
            r["osl"] += int(v("output_sequence_length") or 0)
            r["up"] += int(v("usage_prompt_tokens") or 0)
            r["uc"] += int(v("usage_completion_tokens") or 0)
            t = v("time_to_first_token")
            if t is not None and t > 0:
                r["ttft"].append(t / 1000.0)
            s, e = md.get("request_start_ns"), md.get("request_end_ns")
            if s:
                r["start"] = s if r["start"] is None else min(r["start"], s)
            if e:
                r["end"] = e if r["end"] is None else max(r["end"], e)
    r["duration"] = (r["end"] - r["start"]) / 1e9 if r["start"] and r["end"] else None
    return r


def export_metrics(path):
    """Profiling-phase totals/gauges from aiperf server_metrics_export.json (summed / maxed over series)."""
    out = {"counters": {}, "by_source": {}, "per_pos": {}, "gauges": {}}
    if not path or not path.is_file():
        return None
    d = json.loads(path.read_text())
    out["phase"] = d.get("metrics_phase")
    for name, m in (d.get("metrics") or {}).items():
        if not name.startswith("vllm:"):
            continue
        for s in m.get("series", []):
            st = s.get("stats") or {}
            lab = s.get("labels") or {}
            if m.get("type") == "counter" and "total" in st:
                out["counters"][name] = out["counters"].get(name, 0.0) + float(st["total"])
                if name == "vllm:prompt_tokens_by_source":
                    src = lab.get("source", "?")
                    out["by_source"][src] = out["by_source"].get(src, 0.0) + float(st["total"])
                if name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                    pos = lab.get("position", "?")
                    out["per_pos"][pos] = out["per_pos"].get(pos, 0.0) + float(st["total"])
            elif m.get("type") == "gauge":
                gl = out["gauges"].setdefault(name, {"avg": [], "p90": [], "max": []})
                for k in ("avg", "p90", "max"):
                    if st.get(k) is not None:
                        gl[k].append(float(st[k]))
    return out


PROM = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)')


def prom_counters(path):
    """Sum each metric over label sets in a Prometheus text snapshot; strip the _total suffix."""
    sums, by_source, per_pos = {}, {}, {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        mm = PROM.match(line)
        if not mm or not mm.group(1).startswith("vllm:"):
            continue
        name, labels, val = mm.group(1), mm.group(2) or "", mm.group(3)
        try:
            x = float(val)
        except ValueError:
            continue
        if name.endswith("_created"):
            continue
        base = name[:-6] if name.endswith("_total") else name
        sums[base] = sums.get(base, 0.0) + x
        if base == "vllm:prompt_tokens_by_source":
            src = re.search(r'source="([^"]*)"', labels)
            k = src.group(1) if src else "?"
            by_source[k] = by_source.get(k, 0.0) + x
        if base == "vllm:spec_decode_num_accepted_tokens_per_pos":
            pos = re.search(r'position="([^"]*)"', labels)
            k = pos.group(1) if pos else "?"
            per_pos[k] = per_pos.get(k, 0.0) + x
    return sums, by_source, per_pos


def snapshot_deltas(bench):
    """Delta after-before summed over every scraped endpoint i (metrics_before_i.txt / metrics_after_i.txt)."""
    tot, src, pos, n = {}, {}, {}, 0
    for b in sorted(bench.glob("metrics_before_*.txt")):
        a = bench / b.name.replace("before", "after")
        if not a.is_file() or b.stat().st_size == 0 or a.stat().st_size == 0:
            continue
        (sb, srb, pb), (sa, sra, pa) = prom_counters(b), prom_counters(a)
        n += 1
        for dst, x, y in ((tot, sb, sa), (src, srb, sra), (pos, pb, pa)):
            for k in y:
                dst[k] = dst.get(k, 0.0) + y[k] - x.get(k, 0.0)
    return (tot, src, pos) if n else None


def kv_pool_tokens(bench, run):
    for p in (bench / "server.log", Path(run) / "server.log", Path(run).parent / "server.log"):
        if p.is_file():
            vals = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", p.read_text(errors="ignore"))
            if vals:
                return "+".join(v.replace(",", "") for v in vals)
    return None


def ratio(c, num, den):
    a, b = c.get(num), c.get(den)
    return a / b if a is not None and b else None


def contract(agg, mode, r):
    notes, ic = [], (agg or {}).get("input_config") or {}
    ph = next((p for p in ic.get("phases", []) if p.get("name") == "profiling"), {})
    ds = (ic.get("datasets") or [{}])[0]
    want_dur, want_wu = (3600.0, 10) if mode == "official" else ((1200.0, 1) if mode == "fast" else (None, None))
    if not ic:
        return None, ["no input_config"]
    if want_dur is not None and float(ph.get("duration") or 0) != want_dur:
        notes.append(f"duration={ph.get('duration')}")
    if want_wu is not None and ph.get("warmup_requests_per_lane") != want_wu:
        notes.append(f"warmup/lane={ph.get('warmup_requests_per_lane')}")
    if ds.get("random_seed") != 42:
        notes.append(f"seed={ds.get('random_seed')}")
    if ds.get("dataset") != DATASET:
        notes.append(f"dataset={ds.get('dataset')}")
    if ic.get("scenario") not in ("agentx", None):
        notes.append(f"scenario={ic.get('scenario')}")
    if r and r["total"] and r["errors"] / max(r["total"] - r["warmup"], 1) > 0.10:
        notes.append(f"errors={r['errors']}")
    return (not notes), notes


def fmt(x, nd=4):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{nd}f}" if abs(x) < 1000 else f"{x:.1f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--result-json")
    ap.add_argument("--artifacts")
    ap.add_argument("--gpus", type=int)
    ap.add_argument("--mode", default=None, choices=["official", "fast", "unknown"])
    ap.add_argument("--tag")
    ap.add_argument("--ledger", default=str(LEDGER_DEFAULT))
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--json", action="store_true", help="also print the full summary as JSON")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    run = Path(a.run).resolve()
    arts, resf, bench = find_inputs(run, a)
    man = {}
    if (bench / "manifest.json").is_file():
        man = json.loads((bench / "manifest.json").read_text())
    res = json.loads(resf.read_text()) if resf else {}
    agg = json.loads((arts / "profile_export_aiperf.json").read_text()) if arts and (arts / "profile_export_aiperf.json").is_file() else {}
    mode = a.mode or man.get("mode") or "unknown"
    tag = a.tag or man.get("tag") or run.name
    gpus = a.gpus or res.get("num_gpus") or man.get("gpus_denominator")
    rm = "request_metrics."

    r = raw_records(arts / "profile_export.jsonl") if arts and (arts / "profile_export.jsonl").is_file() else None
    s = {"tag": tag, "mode": mode, "dir": str(run), "result_json": str(resf) if resf else None,
         "artifacts": str(arts) if arts else None, "gpus": gpus, "conc": res.get("conc") or man.get("concurrency")}
    s["score_harness"] = g(res, rm + "throughput.per_gpu.total_tput_tps")
    if r and r["duration"] and gpus:
        s["score_raw"] = (r["isl"] + r["osl"]) / r["duration"] / gpus
        s["score_usage"] = (r["up"] + r["uc"]) / r["duration"] / gpus
        s["raw_duration_s"] = r["duration"]
        s["raw_n"] = r["n"]
        s["raw_isl_mean"] = r["isl"] / r["n"] if r["n"] else None
        s["raw_osl_mean"] = r["osl"] / r["n"] if r["n"] else None
        s["raw_ttft_p50_s"] = pct(r["ttft"], 0.5)
        s["raw_ttft_p90_s"] = pct(r["ttft"], 0.9)
        s["raw_qps"] = r["n"] / r["duration"]
    s["raw_vs_harness_pct"] = rel(s.get("score_raw"), s["score_harness"])
    s["usage_vs_harness_pct"] = rel(s.get("score_usage"), s["score_harness"])
    s["qps"] = g(res, rm + "qps.mean")
    s["isl_mean"] = g(res, rm + "tokens.input.mean")
    s["osl_mean"] = g(res, rm + "tokens.output_actual.mean")
    s["osl_expected_mean"] = g(res, rm + "tokens.output_expected.mean")
    s["ttft_p50_s"] = g(res, rm + "latency.ttft.p50")
    s["ttft_p90_s"] = g(res, rm + "latency.ttft.p90")
    i50, i90 = g(res, rm + "latency.itl.p50"), g(res, rm + "latency.itl.p90")
    s["itl_p50_ms"] = i50 * 1000 if i50 is not None else None
    s["itl_p90_ms"] = i90 * 1000 if i90 is not None else None
    s["intvty_p90"] = g(res, rm + "latency.intvty.p90")
    s["duration_s"] = g(res, rm + "throughput.duration_seconds")
    s["hit_theo"] = g(res, rm + "cache.theoretical_cache_hit_rate")
    s["ok"] = res.get("num_requests_successful", r["n"] if r else None)
    s["total"] = res.get("num_requests_total", r["total"] if r else None)
    s["errors"] = g(res, "request_accounting.records_error_dropped", r["errors"] if r else None)
    s["image"] = res.get("image") or man.get("server_image_label")

    ex = export_metrics(arts / "server_metrics_export.json") if arts else None
    if ex:
        c = ex["counters"]
        s["metrics_phase"] = ex.get("phase")
        # hit_gpu: token-based, same definition as InferenceX server_metrics.cache.gpu_cache_hit_rate
        # (prompt_tokens_by_source local_cache_hit / all prompt tokens); hit_pq: vllm:prefix_cache_hits/queries.
        s["hit_pq"] = ratio(c, "vllm:prefix_cache_hits", "vllm:prefix_cache_queries")
        s["hit_ext"] = ratio(c, "vllm:external_prefix_cache_hits", "vllm:external_prefix_cache_queries")
        acc, dr = c.get("vllm:spec_decode_num_accepted_tokens"), c.get("vllm:spec_decode_num_drafts")
        s["al"] = 1 + acc / dr if acc is not None and dr else None
        s["accept_per_pos"] = {k: v / dr for k, v in sorted(ex["per_pos"].items())} if dr else None
        bs = ex["by_source"]
        tot = sum(bs.values())
        s["prompt_by_source_frac"] = {k: v / tot for k, v in bs.items()} if tot else None
        s["hit_gpu"] = bs.get("local_cache_hit", 0.0) / tot if tot else None
        s["preemptions"] = c.get("vllm:num_preemptions")
        kv = ex["gauges"].get("vllm:kv_cache_usage_perc", {})
        s["kv_use_avg"] = max(kv["avg"]) if kv.get("avg") else None
        s["kv_use_p90"] = max(kv["p90"]) if kv.get("p90") else None
        s["kv_use_max"] = max(kv["max"]) if kv.get("max") else None
        run_g = ex["gauges"].get("vllm:num_requests_running", {})
        wait_g = ex["gauges"].get("vllm:num_requests_waiting", {})
        s["running_avg"] = sum(run_g["avg"]) if run_g.get("avg") else None
        s["waiting_avg"] = sum(wait_g["avg"]) if wait_g.get("avg") else None
        s["harness_hit_gpu"] = g(res, "server_metrics.cache.gpu_cache_hit_rate")
    sd = snapshot_deltas(bench)
    if sd:
        c2, _, _ = sd
        s["hit_pq_snap"] = ratio(c2, "vllm:prefix_cache_hits", "vllm:prefix_cache_queries")
        tot2 = sum(sd[1].values())
        s["hit_gpu_snap"] = sd[1].get("local_cache_hit", 0.0) / tot2 if tot2 else None
        s["hit_ext_snap"] = ratio(c2, "vllm:external_prefix_cache_hits", "vllm:external_prefix_cache_queries")
        acc, dr = c2.get("vllm:spec_decode_num_accepted_tokens"), c2.get("vllm:spec_decode_num_drafts")
        s["al_snap"] = 1 + acc / dr if acc is not None and dr else None
    s["kv_pool_tokens"] = kv_pool_tokens(bench, run) or res.get("kv_cache_pool_tokens")
    ok, notes = contract(agg, mode, r)
    s["contract_ok"] = ok
    s["contract_notes"] = notes

    print(f"== {tag}  mode={mode} conc={s['conc']} gpus={gpus} dir={run}")
    print(f"score tok/s/GPU: harness={fmt(s['score_harness'])} raw={fmt(s.get('score_raw'))} "
          f"({fmt(s['raw_vs_harness_pct'], 3)}%) usage-based={fmt(s.get('score_usage'))} ({fmt(s['usage_vs_harness_pct'], 3)}%)")
    print(f"qps={fmt(s['qps'])} (raw {fmt(s.get('raw_qps'))})  ISL={fmt(s['isl_mean'])} OSL={fmt(s['osl_mean'])} "
          f"(expected {fmt(s['osl_expected_mean'])})  dur={fmt(s['duration_s'])}s  ok/total={s['ok']}/{s['total']} err={s['errors']}")
    print(f"TTFT p50/p90={fmt(s['ttft_p50_s'])}/{fmt(s['ttft_p90_s'])} s (raw {fmt(s.get('raw_ttft_p50_s'))}/"
          f"{fmt(s.get('raw_ttft_p90_s'))})  ITL p50/p90={fmt(s['itl_p50_ms'], 3)}/{fmt(s['itl_p90_ms'], 3)} ms  "
          f"p90 intvty={fmt(s['intvty_p90'], 2)}")
    print(f"prefix hit: gpu(token)={fmt(s.get('hit_gpu'))} (harness {fmt(s.get('harness_hit_gpu'))}, snapshot incl. warmup "
          f"{fmt(s.get('hit_gpu_snap'))}) hits/queries={fmt(s.get('hit_pq'))} (snapshot {fmt(s.get('hit_pq_snap'))}) "
          f"ext={fmt(s.get('hit_ext'))} theo={fmt(s['hit_theo'])} "
          f"by_source={json.dumps({k: round(v, 4) for k, v in (s.get('prompt_by_source_frac') or {}).items()})}")
    print(f"AL={fmt(s.get('al'))} (snapshot {fmt(s.get('al_snap'))}) per-pos="
          f"{json.dumps({k: round(v, 3) for k, v in (s.get('accept_per_pos') or {}).items()})}")
    print(f"KV usage avg/p90/max={fmt(s.get('kv_use_avg'))}/{fmt(s.get('kv_use_p90'))}/{fmt(s.get('kv_use_max'))} "
          f"pool tokens={s['kv_pool_tokens']} running avg={fmt(s.get('running_avg'), 2)} waiting avg={fmt(s.get('waiting_avg'), 2)} "
          f"preemptions={fmt(s.get('preemptions'), 0)}")
    print(f"contract_ok={ok} {'; '.join(notes)}")
    if a.json:
        print(json.dumps(s, indent=2, default=str))

    row = {k: "" for k in COLUMNS}
    row.update({k: fmt(s.get(k), 6) if isinstance(s.get(k), float) else ("" if s.get(k) is None else s.get(k))
                for k in COLUMNS if k in s})
    row["utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row["notes"] = "; ".join([x for x in [a.notes] + notes if x])
    line_fields = [str(row[k]) for k in COLUMNS]
    if a.no_ledger:
        print("ledger line (not written): " + ",".join(line_fields))
        return 0
    led = Path(a.ledger)
    if "yihou" not in str(led.resolve()):
        sys.exit("ledger path must contain 'yihou'")
    led.parent.mkdir(parents=True, exist_ok=True)
    with open(led, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.tell() == 0 and os.path.getsize(led) == 0:
            csv.writer(f).writerow(COLUMNS)
        csv.writer(f).writerow(line_fields)
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"appended to {led}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
