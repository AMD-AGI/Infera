# 第 2～6 项准备工作（尚未启动实验）

当前唯一运行目标是同节点 4K C80 基线；本目录只保存后续实验方案，不自动执行这些变体。

## 可复用基线

固定 n03-33 Prefill / n02-21 Decode、同一镜像和诊断代码、P/D 实际 chunk=4096、mem_fraction_static=0.85、host ratio=1.5、kernel/page_first、acceptance=3.61、C80、每 lane 10 warmup、3600 秒正式窗口。每次从进程退出、GPU/host 资源释放后的新服务开始，不能继续使用上一项热缓存。

完整采集包含镜像与容器命令/环境、server-info、工作负载 revision/hash、tokenizer/config 哈希、trace/request 原始记录、2 秒 engine 和 5 秒 node 采样、阶段账目、错误归属、HiCache/GPU 释放时间。模型元数据哈希不等于全量权重哈希。单次基线不提供运行间波动置信区间。

## 参数变体

| 项目 | 准备的变更 | 发压前必须满足 |
|---|---|---|
| 2 P 容量 | P fraction 0.85→0.90 | 实际 host pool 保持 4,715,200 tokens/rank，确认 workspace 余量与实际 GPU 池 |
| 3a host 容量 | ratio 1.5→2.0 | P GPU 容量不变、backend/layout 不变，确认总 host 内存余量 |
| 3b host backend | kernel→direct 候选 | 先确认相同 page_first layout、dtype 和硬件可用；若必须改 layout，不能再称为只改 backend |
| 4 D 容量 | D fraction 0.85→0.90 | P 不变，核验实际 D 池、graph 和 retraction |
| 5 P 调度公平性 | 先准备候选顺序/预算策略 | 与同一诊断构建的对照比较，补停止接纳原因与候选 age/miss/budget；不直接改当前基线 |
| 6 D 路由 | 先 shadow 后切换 | 补全两个 admission 条件、max_new、每 rank budgets/slots、pending reservations 与时刻；free KV 不能代替完整可接纳性 |

具体参数、预言、否定条件见 cases.json。

## Host 绝对容量的实现细节

从本次诊断镜像提取的 pool_host/base.py 使用十进制 GB：`int(host_size * 1e9 // bytes_per_token)`；之后按 `floor(tokens/page_size)+1` 页分配，即原始 token 数恰好对齐时也多加一页。`hicache_size` 的 CLI 类型是 int。

当前 page_first、78 层、FP8、(512+64) 维得到 44,928 bytes/token。4,715,200 host tokens 对应 211,844,505,600 bytes = 211.8445056 GB。设 212 GB 会变成 4,718,720 tokens，不能声称保持绝对容量完全相同。

已提供 scripts/plan_host_capacity.py：输入新配置实际 device token 容量，计算使实际 host token 容量保持不变的 ratio。新 GPU 池未确认前不能把预测值当成实际值。若必须先做一次容量标定启动，需把额外退出/显存释放成本纳入排期；也可另设相同整数 GB 的成对控制，但不能直接拿当前 ratio 基线当精确等容量对照。

source/ 保存提取文件及镜像 ID、哈希。基线完成后应以其实际容量重新检查这些计划值。
