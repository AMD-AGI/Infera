# User directives of 2026-10-14 (verbatim, from the conversation summary)

Times are UTC and [inferred] from env-file headers and `analysis/ledger/poll.md` entries, not from message timestamps.

1. (~01:30) "按照你的想法测0.90和0.92 prefill kv"
   -> rounds pf90 (089), pf92 (206 -> 085 after 206's job ended), same-session control pf88 (274).
2. (~03:30) "找我们hi-cache=450的最好的实验结果，基于此，改用selective throughput，测一下看有没有性能提升"
   -> round wts48 (q1 config + PREFILL_HICACHE_WRITE_POLICY=write_through_selective). User 03:4x "使用136".
3. (~04:20) "pc48b A2看起来性能不错，追加一个实验，在pc48B的基础上，跑conc=64. 另外runshare的实验先停一停"
   -> round pc64 (089, after pf90 A2). No runshare round was running.
4. (~07:15) "直觉上write_through——selective应该支持更高的conc, 测一下wts64, pf88先不用跑。增大gmu理论上时可以给kv cache腾出更多空间的，但是这些空间确实被挤压的一滴不剩了么？你调研一下，如果必要，可以尝试在0.92下不断增加kv_cache_token, 调整block对齐，直到oom"
   -> wts64 (C64 on the live wts48 stack), memory accounting (analysis/memory_accounting.md), pf94/pf93 probe (armed on 136).
5. (~07:15) "插空可以在我们现在最高的吞吐的配置上做一些正确性的测试"
   -> round cx90 (pf90 config, simulated acceptance OFF, scripts/session/correctness_tests.yihou.py).
6. (~07:5x, on prefill_complete) "哦那放弃这条优化路径"
   -> pc64 A2 aborted; no further prefill_complete rounds.
