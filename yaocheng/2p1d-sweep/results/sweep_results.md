Run: `2p1d-20260921T151735Z`  
状态：`complete`  
P/D 节点：`[{"role": "prefill", "node": "crsuse2-m2m-137", "data_ip": "10.245.153.247"}, {"role": "prefill", "node": "crsuse2-m2m-138", "data_ip": "10.245.157.237"}, {"role": "decode", "node": "crsuse2-m2m-136", "data_ip": "10.245.154.168"}]`  
完整运行信息：`/mnt/m2m_nobackup/xiaobche/2p1d-sweep/shared-kit/.tmp/results/2p1d-20260921T151735Z`

# 2P1D AgentX concurrency sweep

只统计完成测量、24 GPU 元数据校验和前后服务检查的档位。

total tok/s/GPU 包含输入和输出 token（含命中缓存的输入）；output tok/s/GPU 单列输出吞吐。
默认使用模拟 acceptance 3.61，结果只用于性能对比，不代表生成正确性。

| conc | GPUs | total tok/s/GPU | output tok/s/GPU | TTFT p50 s | TTFT p90 s | ITL p50 ms | intvty p50 | profiled | error dropped* |
|---|---|---|---|---|---|---|---|---|---|
| 80 | 24 | 14821.96 | 115.94 | 3.219 | 7.940 | 15.560 | 64.27 | 10551 | 1 |
| 112 | 24 | 16886.23 | 136.83 | 4.838 | 11.579 | 17.850 | 56.01 | 12166 | 4 |
| 144 | 24 | 18356.23 | 152.81 | 8.976 | 22.998 | 18.110 | 55.22 | 13337 | 7 |
| 192 | 24 | 17504.51 | 139.46 | 29.408 | 66.617 | 16.370 | 61.08 | 12388 | 7 |
| 256 | 24 | 14346.41 | 115.36 | 68.221 | 168.186 | 17.130 | 58.39 | 10513 | 28 |

*error dropped 为原始聚合的 records_error_dropped，可能包含预热阶段；不等同于正式测量失败数。详见 measurement_notes.md。

