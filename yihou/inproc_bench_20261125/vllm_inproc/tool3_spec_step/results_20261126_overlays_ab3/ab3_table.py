import glob, json, statistics, subprocess, sys, os
W = "work_v3/iterations"; rows = {}
for arm in ("A", "B3"):
    for rep in (1, 2, 3, 4):
        p = f"{W}/v3z_{arm}r{rep}_u173k"
        if not (os.path.exists(f"{p}/result_yihou.json") and os.path.exists(f"{p}/launch_status.json")):
            continue
        v = json.loads(subprocess.run([sys.executable, "kit_v3/vllm_inproc/tool3_spec_step/scripts/verify_point.py", p],
                                      capture_output=True, text=True).stdout)[0]
        r = json.load(open(f"{p}/result_yihou.json"))
        s = [json.loads(l)["seconds"] * 1000 for l in open(f"{p}/steps_rank_0_yihou.jsonl")]
        g = r["graph_execution_counts_by_rank"]
        rows[(arm, rep)] = dict(verdict=v["verdict"], problems=v["problems"], p50=r["step_seconds_p50"] * 1000,
                                p50_r1=r["step_seconds_p50_by_rank"]["1"] * 1000, mean=statistics.mean(s), max=max(s),
                                tpot=r["effective_token_latency_ms_per_user"], acc=r["realized_accept_length"],
                                it=r["verify_iterations"],
                                replays=all(g[k]["target"] == g[k]["draft"] == r["verify_iterations"] for k in g))
for k, v in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    print(f"{k[0]}r{k[1]} {v['verdict']} p50 {v['p50']:.3f}/{v['p50_r1']:.3f} mean {v['mean']:.3f} max {v['max']:.1f} "
          f"tpot {v['tpot']:.3f} acc {v['acc']!r} it {v['it']} replays_ok {v['replays']} {v['problems'] or ''}")
arms = {a: [v["p50"] for k, v in rows.items() if k[0] == a] for a in ("A", "B3")}
for a, xs in arms.items():
    if xs:
        print(a, "n", len(xs), "mean", round(statistics.mean(xs), 3), "min", round(min(xs), 3), "max", round(max(xs), 3),
              "spread", round(max(xs) - min(xs), 3), "sd", round(statistics.stdev(xs), 3) if len(xs) > 1 else None)
if arms["A"] and arms["B3"]:
    a = statistics.mean(arms["A"]); print(f"B3 vs A: {(statistics.mean(arms['B3']) - a) / a * 100:+.2f} %")
    for rep in (1, 2, 3, 4):
        if ("A", rep) in rows and ("B3", rep) in rows:
            x, y = rows[("A", rep)]["p50"], rows[("B3", rep)]["p50"]
            print(f"round {rep}: B3 vs A {(y - x) / x * 100:+.2f} %")
