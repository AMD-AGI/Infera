# C1：固定 host 的 Prefill GPU KV 扩容

2026-09-24 用户提供新分配并提出优先扩容；顺序调整为 C1 容量验证，然后根据结果决定是否复用同一 P/D 开启路由优化。不重复采集基线。

## 当前状态

02:21:16 UTC 开始 LAUNCHING，尚未通过健康检查或开始 benchmark。driver PID 1194334，运行根目录 `/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923`，run `runs/c1-fixed-host-31688`，日志 `c1-fixed-host-31688-driver.log`。自动流程包含健康检查、容量/配置验收、smoke、884 请求预热、3600 秒正式采集和分析。该状态仅表示启动，不表示实验成功。

## 设计

| 参数 | 历史 A0 | C1 目标 |
|---|---:|---:|
| P GPU tokens/rank | 3,143,424 | 3,400,000（+8.16%） |
| P host tokens/rank | 4,715,200 | 4,715,200 |
| P mem_fraction_static | 0.85 | 0.90 |
| P max_total_tokens | 自动 | 3,400,000 |
| P hicache_ratio | 1.5 | 1.3868141176470588 |
| 路由 guard 释放 | decode | decode |
| 实际 P/D chunk | 4096/4096 | 4096/4096 |
| 并发 | 80 | 80 |

ratio 按 `(host - 32) / gpu_tokens` 计算，按 64-token 页分配规则保持 host 绝对容量。budget/cap/ratio 共同实现一项 P GPU KV 容量干预；8 ranks 实际 GPU/host 容量必须验收通过才发压。D 沿用 A0 参数；workload、seed、镜像和客户端约束保持已有设置。

直接参考历史 A0，不同时开启路由优化。新节点、驱动、启动历史和缓存编译包带来比较限制，不能宣称严格同节点因果对照。C1 如稳定且有收益，再在相同容量/P-D 上清空逻辑缓存并开启 completion guard，验证路由增量收益；若扩容无效或显存不稳定，停止继续扩容，重新判断路由验证安排。20→10 权重试探仍暂缓。

## 当前节点

- job 31688，emad / Compute-DCPT / batch，StartTime 02:12:04 UTC，计划 EndTime 次日 02:12:04 UTC，抢占保护到 02:42:04 UTC。
- P n02-29：AMDGPU 6.14.14；D n02-33：AMDGPU 6.16.6。
- 两端内核 6.8.0-107-generic，ionic 26.03.3.001，firmware 1.117.5-a-77。
- 启动前 16 GPU busy=0，VRAM 约 0.28 GiB/卡；两节点均有其他用户 xiaoming-dev 容器，不操作它们；正式采集使用现有 GPU 所有权监控，出现干扰则标记无效。
- 两端固定镜像 ID 已核实；16 个已完成 AITER 库经哈希校验安装到独立本地缓存。
- 两端 allocation watchdog 绑定 job、节点、StartTime 和本实验容器前缀，避免抢占后误清理其他任务。

## 判断标准

观察 device/host 命中分布、miss/recompute、P queue、forward、TTFT、total/output tokens/s/GPU；同时检查 OOM、retraction、尾延迟和显存峰值。GPU KV 增大可能只把 host hit 移到 device，并不保证 miss 减少。吞吐、时延及机制证据共同判断是否值得继续。

历史峰值估算扩容后约占 95–96% VRAM，属于需实测验证的余量，不能把预算通过视为已证明无 OOM。发生原生崩溃或验收失败时不开始负载。
