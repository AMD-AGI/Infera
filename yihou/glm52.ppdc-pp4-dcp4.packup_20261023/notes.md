# Notes

1. PP4 prefill started first time with the deepseek_v2 fix; prefill KV pool 10.74 M tokens (one pool across stages).
2. The harness needed: router affinity off (one prefill DP), OffloaderV2 args stripped, prefill mounts as one 3-row tsv,
   agentx_env TP*PP.
3. `ssh` inside a piped interactive session must be `ssh -n` (otherwise it eats the following commands); the spur LB host
   key changes between login hosts - use spur006 directly when it answers.
4. The sweep's ISL falls with concurrency (closed-loop AgentX: long cold requests take longer and fewer complete), so a
   higher score at higher C partly comes from a lighter mix.
5. Not measured: the binding limit at C72-C80 (prefill GPU prefix capacity vs decode step time), correctness under sim
   acceptance, single contributions of PP4 / SPF / no-OffloaderV2 / prefill mem.
