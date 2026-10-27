"""Measurements as the ATOM kit and lm_eval leave them, for the tests of the ledger, the A/B gates and the field
checks: an AgentX run directory (agentx_conc<C>.json, runtime.env, AIPerf's export with its window) from the real
outputs in tests/fixtures, and an lm_eval GSM8K run (samples and results in lm_eval's format).

- fixtures/agentx/agentx_conc40.json: a real AgentX output.
- fixtures/aiperf/profile_export_aiperf.json: a real AIPerf 1.0.11 export of a 3600 s point (the per-metric
  statistics left out, the host replaced).
- fixtures/lm_eval/: a real GSM8K run's samples (its first 20 questions, prompts and responses left out) and its
  results summary (counts made to match those 20).
"""

import copy
import itertools
import json
from datetime import datetime, timedelta
from pathlib import Path

from agent_sys_v2 import ledger

FIXTURES = Path(__file__).parent / "fixtures"
RAW = json.loads((FIXTURES / "agentx/agentx_conc40.json").read_text())
AIPERF = json.loads((FIXTURES / "aiperf/profile_export_aiperf.json").read_text())
SAMPLES = [
    json.loads(line)
    for line in (FIXTURES / "lm_eval/samples_gsm8k_2026-10-26T08-57-57.452731.jsonl")
    .read_text()
    .splitlines()
]
LM_RESULTS = json.loads((FIXTURES / "lm_eval/results_2026-10-26T08-57-57.452731.json").read_text())
NODE = "10.0.0.1"
T0 = datetime(2026, 10, 27, 1, 0, 0)
SLOT = (
    itertools.count()
)  # each measurement starts after the one before: the window, redeploy and warm-up


def raw(scale=1.0, itl=0.01415, errors=0, conc=40, duration=None):
    """An agentx_conc<C>.json: the real one (measured for 3626.6 s) with its throughput scaled, its p90 ITL, failed
    requests and, when given, its measured duration set."""
    agg = copy.deepcopy(RAW)
    agg["conc"] = conc
    per_gpu = agg["request_metrics"]["throughput"]["per_gpu"]
    for k in ("input_tput_tps", "total_tput_tps", "output_tput_tps"):
        per_gpu[k] = round(per_gpu[k] * scale, 5)
    agg["request_metrics"]["latency"]["full_response_itl"]["p90"] = itl
    agg["request_accounting"]["records_error_dropped"] = errors
    if duration is not None:
        agg["request_metrics"]["throughput"]["duration_seconds"] = duration
    return agg


def aiperf(conc, duration, start, node=NODE):
    """AIPerf's export of a point: the real one with its phase, window and endpoint set."""
    export = copy.deepcopy(AIPERF)
    phase = export["input_config"]["phases"][0]
    phase["concurrency"], phase["duration"] = conc, float(duration)
    end = start + timedelta(seconds=duration + 26.7)
    export["start_time"], export["end_time"] = start.isoformat(), end.isoformat()
    export["input_config"]["endpoint"]["urls"] = [f"http://{node}:18000"]
    return export


def measure(ws, name, agg, duration=3600, node=NODE, start=None, fast=None, export=True):
    """A run directory as the ATOM kit's agentx.sh leaves it: agentx_conc<C>.json beside runtime.env and
    aiperf_artifacts/profile_export_aiperf.json. start: the window's start (default: the next slot, 85 min after the
    one before); fast: AIPERF_EXPERIMENTAL_FAST in runtime.env (absent by default, as the ATOM kit writes it)."""
    d = ws / "log" / "agentx" / name
    d.mkdir(parents=True)
    (d / f"agentx_conc{agg['conc']}.json").write_text(json.dumps(agg, indent=1))
    env = f"CONC={agg['conc']}\nDURATION={duration}\nAIPERF_WARMUP_REQUESTS_PER_LANE=10\n"
    if fast is not None:
        env += f"AIPERF_EXPERIMENTAL_FAST={int(fast)}\n"
    (d / "runtime.env").write_text(env)
    if export:
        start = start or T0 + timedelta(minutes=85 * next(SLOT))
        set_to = 1200 if fast else duration
        (d / "aiperf_artifacts").mkdir()
        (d / "aiperf_artifacts/profile_export_aiperf.json").write_text(
            json.dumps(aiperf(agg["conc"], set_to, start, node), indent=1)
        )
    return d


def convert(ws, run, series, role, label, conc=40, round_=1):
    argv = [
        "agentx",
        str(run),
        "--out",
        str(ws / "results"),
        "--series",
        series,
        "--round",
        str(round_),
    ]
    assert (
        ledger.main([*argv, "--role", role, "--label", label, "--time", "2026-10-27T10:00:00Z"])
        == 0
    )
    return ws / "results" / f"{label}-c{conc}.json"


def lm_eval_run(d, total, wrong, stamp="2026-10-27T08-00-00.000000"):
    """An lm_eval GSM8K run as engine/eval.sh leaves it under <run>/gsm8k/<model>/: questions 0..total-1, the ones
    in wrong answered wrong under both filters."""
    out = Path(d) / "__models__GLM-5.2-MXFP4"
    out.mkdir(parents=True)
    lines = []
    for doc in range(total):
        for flt in ("strict-match", "flexible-extract"):
            row = dict(SAMPLES[(doc % 20) * 2 + (flt == "flexible-extract")])
            row.update(doc_id=doc, filter=flt, exact_match=0.0 if doc in wrong else 1.0)
            lines.append(json.dumps(row))
    (out / f"samples_gsm8k_{stamp}.jsonl").write_text("\n".join(lines) + "\n")
    res = copy.deepcopy(LM_RESULTS)
    g = res["results"]["gsm8k"]
    g["sample_len"] = total
    for flt in ("strict-match", "flexible-extract"):
        g[f"exact_match,{flt}"] = (total - len(wrong)) / total
    res["n-samples"]["gsm8k"]["effective"] = total
    res["config"]["limit"] = float(total)
    (out / f"results_{stamp}.json").write_text(json.dumps(res, indent=1))
    return Path(d)


def gsm8k(ws, label, series, total, wrong):
    """An lm_eval run under the workspace's log/, converted with ledger gsm8k into its results/."""
    run = lm_eval_run(ws / "log" / "gsm8k" / label, total, set(wrong))
    argv = ["gsm8k", str(run), "--out", str(ws / "results"), "--label", label, "--series", series]
    assert ledger.main(argv) == 0
    return ws / "results" / f"{label}.gsm8k.json"
