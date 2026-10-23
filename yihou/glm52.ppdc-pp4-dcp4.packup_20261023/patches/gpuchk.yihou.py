# gpuchk (task ppdc 2026-10-23): total vs per-GPU throughput in an AgentX result + PREFILL_TP in runtime.env.
import json, sys, pathlib
b = pathlib.Path(sys.argv[1]); d = json.load(open(b / "agentx_conc56.json")); t = d["request_metrics"]["throughput"]
def walk(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): walk(v, p + "." + k)
    elif isinstance(x, (int, float)): print(p, x)
walk(t)
for line in open(b / "runtime.env"):
    if line.split("=")[0] in ("TP", "PREFILL_TP", "DECODE_TP", "PREFILL_DP_SIZE", "DECODE_DP_SIZE", "PREFILL_NUM_WORKERS", "DECODE_NUM_WORKERS"): print(line.strip())
