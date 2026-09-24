Run: `2p1d-triton-dsa-20260922T124849Z`  
状态：`complete`  
P/D 节点：`[{"role": "prefill", "node": "crsuse2-m2m-137", "data_ip": "10.245.153.247"}, {"role": "prefill", "node": "crsuse2-m2m-138", "data_ip": "10.245.157.237"}, {"role": "decode", "node": "crsuse2-m2m-136", "data_ip": "10.245.154.168"}]`  
完整运行信息：`/mnt/m2m_nobackup/xiaobche/2p1d-sweep-triton-dsa-20260922/shared-kit/.tmp/results/2p1d-triton-dsa-20260922T124849Z`

# 2P1D AgentX concurrency sweep

只统计完成测量、24 GPU 元数据校验和前后服务检查的档位。

total tok/s/GPU 包含输入和输出 token（含命中缓存的输入）；output tok/s/GPU 单列输出吞吐。
默认使用模拟 acceptance 3.61，结果只用于性能对比，不代表生成正确性。

| conc | GPUs | total tok/s/GPU | output tok/s/GPU | TTFT p50 s | TTFT p90 s | ITL p50 ms | intvty p50 | profiled | error dropped* |
|---|---|---|---|---|---|---|---|---|---|
| 80 | 24 | 14975.27 | 115.51 | 2.415 | 6.377 | 16.020 | 62.43 | 10641 | 3 |
| 112 | 24 | 17581.16 | 140.52 | 4.155 | 9.529 | 17.520 | 57.07 | 12529 | 3 |
| 144 | 24 | 18519.35 | 154.15 | 10.481 | 22.480 | 16.830 | 59.41 | 13439 | 3 |
| 192 | 24 | 18806.71 | 149.44 | 26.458 | 50.777 | 16.470 | 60.72 | 13305 | 5 |
| 256 | 24 | 15618.00 | 126.09 | 60.131 | 137.236 | 17.050 | 58.65 | 11471 | 53 |

*error dropped 为原始聚合的 records_error_dropped，可能包含预热阶段；不等同于正式测量失败数。详见 measurement_notes.md。

