import json, statistics, subprocess, sys, os
W = "work_v3/iterations"; rows = {}
for arm in "ABC":
    for rep in (1, 2, 3, 4):
        p = f"{W}/v3x_{arm}{rep}_u173k"
        if not os.path.exists(f"{p}/result_yihou.json") or not os.path.exists(f"{p}/launch_status.json"):
            continue
        out = subprocess.run([sys.executable, "kit_v3/vllm_inproc/tool3_spec_step/scripts/verify_point.py", p], capture_output=True, text=True).stdout
        v = json.loads(out)[0]; r = json.load(open(f"{p}/result_yihou.json"))
        s = [json.loads(l)["seconds"] * 1000 for l in open(f"{p}/steps_rank_0_yihou.jsonl")]
        g = r["graph_execution_counts_by_rank"]
        rows[(arm, rep)] = dict(verdict=v["verdict"], p50=r["step_seconds_p50"] * 1000,
                                p50_r1=r["step_seconds_p50_by_rank"]["1"] * 1000, mean=statistics.mean(s), max=max(s),
                                tpot=r["effective_token_latency_ms_per_user"], acc=r["realized_accept_length"],
                                it=r["verify_iterations"], replays=all(g[k]["target"] == g[k]["draft"] == r["verify_iterations"] for k in g))
for k, v in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    print(f"{k[0]}{k[1]} {v['verdict']} p50 {v['p50']:.3f}/{v['p50_r1']:.3f} mean {v['mean']:.3f} max {v['max']:.1f} tpot {v['tpot']:.3f} acc {v['acc']!r} it {v['it']} replays_ok {v['replays']}")
arms = {a: [v["p50"] for k, v in rows.items() if k[0] == a] for a in "ABC"}
for a, xs in arms.items():
    if xs: print(a, "n", len(xs), "mean", round(statistics.mean(xs), 3), "min", round(min(xs), 3), "max", round(max(xs), 3), "spread", round(max(xs) - min(xs), 3), "sd", round(statistics.stdev(xs), 3) if len(xs) > 1 else None)
if arms["A"]:
    a = statistics.mean(arms["A"])
    for b in "BC":
        if arms[b]: print(f"{b} vs A: {(statistics.mean(arms[b]) - a) / a * 100:+.2f} %")
