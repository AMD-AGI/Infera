# D radix节省了主KV传输，但DSA indexer仍走整段state

固定镜像中，decode.py构造主KV目标索引时使用 `[total_prefix_len:origin_input_len]`，并把 `decode_prefix_len` 传给发送端。因此主KV可按D本地复用长度缩减。

同一文件的 `_full_kv_pages_payload()` 使用 `[:seq_len]`，`StateType.DSA` 映射到这个函数。DSA indexer state的目标页索引仍覆盖整个prompt。源文件：`/perf_apps/liyingli/bench_agentx/router-capacity-20260928/docker/decode_prefix_diag.py`，对应约1417–1429、1461–1467、1513–1518行。这个文件仅额外添加了三行只读prefix诊断，其余传输逻辑来自固定镜像。

因此，“input − D prefix”的token统计只适合描述主KV的逻辑增量，不能直接等同于所有线上传输字节。实测D节点8条rail的RDMA接收平均速率之和：B2 15.5356 GB/s、B3 4.8938 GB/s，约下降68.5%；B3原生匹配子集的主KV前缀复用约91.8%。硬件计数是节点级，并非逐请求归因；B3末尾采样被抢占截断（702 vs B2 720个样本/rail）。

这是后续传输优化的候选位置，尚不能判定为可直接删除的冗余。若尝试让DSA indexer state也按D前缀裁剪，必须先核对indexer缓存的有效生命周期、P/D源目的页索引对齐及重试/取消行为，再做真实接受率的命中/清D对照和已知答案验证。不能只修改D切片，也不应在本次已冻结的性能矩阵中悄悄加入这一变量。

本轮继续使用现有实现完成Triton和资源比例比较。网络流量下降而吞吐仅小幅变化，也说明在当前负载下不能把前缀复用率直接换算成系统吞吐收益。
