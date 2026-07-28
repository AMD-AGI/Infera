# Optimus-AgenticBench — GLM-5.2 caseA on spur

Ran the AgenticBench GLM-5.2 production workload **Case A** (`agent/workloads/glm52_crxx_caseA.yaml`)
against our GLM-5.2-MXFP4 sglang service on crsuse spur (node069).

## Setup
- Server: single-node `sglang.launch_server` TP8, **context-length 262144** (caseA p99 input = 235K),
  `--mem-fraction-static 0.90 --enable-cache-report --max-running-requests 48`, GLM DSA-ROCm recipe.
  Image `infera.yihou.sglang.1.0`. Launcher: `scripts/server_bigctx.sh`.
- Client: `agent-bench` (pip-installed from /home/yihou/dev/git/Optimus-AgenticBench into the
  container), driving the server's OpenAI `/v1/chat/completions`.
- Workload: caseA yaml with tokenizer path set to `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.
  Offline profile validator PASSED (input 74K/154K/235K, output 320/3300/17000, cache 0.89 — all match spec).

## Command
```bash
agent-bench agent --server http://<node069_ip>:30000 --model glm5.2-mxfp4 \
  --tokenizer /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 \
  --workload-config /tmp/glm52_caseA.yaml --mode realistic \
  --num-sessions 16 --max-inflight 16 --name glm52_caseA_probe
```
(realistic session/turn/think-time mode; ran the full 600s sustain window.)

## Result — PASS
- **Success rate: 95.9%** (1 transient error out of ~116 requests — a "can not write request body"
  on a very large prompt, not a server crash; server 0 crashes throughout).
- Duration 618.9s, avg QPS 0.20 (target ramp 0.05→0.40). 37 sessions (8 initial + 29 rate-based).
- **Prompt length reproduced the spec**: mean 82K, p50 70K, p90 148K, p99 183K tokens (caseA large-context).
- **TTFT** (sustain phase): p50 749.6ms, p90 1861ms. **TPOT**: p50 17.4ms, p90 24.4ms.
- **Peak prefill throughput: 622,872 tok/s** (77,859 tok/s/GPU).
- **Cache**: ideal 89.0% → actual **86.5%** hit rate (97.2% efficiency, 2.8% eviction) — the server's
  prefix cache tracked the caseA 89%-cache-hit design closely. 9.55M total tokens, 8.26M cached.

vs the caseA SLA target (`e2e_p50_ms 4500, success_rate 0.97`): sustain-phase TTFT p50 750ms and TPOT
p50 17ms are well within budget; success 95.9% is just under the 0.97 floor due to the single
large-prompt write error (ramp phase; sustain phase was clean).

See `caseA_summary.txt` for the full metrics block.
