# 11-11 夜间工作总览（16:54 起）

你的要求：sglang#48302 和 Mooncake #4540 / #4541 按我的 review 意见改动并验证；两个 Mooncake PR 转 ready；sglang 的 CI 只要 PR Test (Base) 全过、其他失败与 main 一致即可；过程及时写进 `llying/` 的报告。

详细记录：
- `llying/sglang-48302-review-20261111/PROGRESS-20261111.zh-CN.md`
- `llying/mooncake-4540-4541-review-20261111/PROGRESS-20261111.zh-CN.md`
- `llying/srt-slurm-630-review-20261111/QA-20261111.zh-CN.md` 末尾（#630 后续提交的测试）

## 结果

| PR | 改动 | 验证 | 状态 |
|---|---|---|---|
| [sglang#48302](https://github.com/sgl-project/sglang/pull/48302) | 合并 11-11 的 main（无冲突，新列下标仍正确）；源码文本测试换成 3 个行为测试；描述补上范围、NVIDIA 行为变化、单 DP rank 的说法、设计取舍和 main 上的新验证 | 单测 13/13 + 4 个相关文件全过；**main 上两轮审计**（GLM-5.2-MXFP4，TP4/DP4 attention）：17,920 + 19,712 次 replay，各 rank 选图 0 次不一致（原规则会有 468 / 448 次），上界 0 次小于真实值，dense 686 次；GSM8K 0.980 / 0.983 / 0.978，乱码 0；开 EAGLE 时 verify 只抓一套图 | 已推送 `4dae7a44`，描述已更新；CI 见下 |
| [Mooncake#4541](https://github.com/kvcache-ai/Mooncake/pull/4541) | 合并 main；通知 QP 也用 `MC_IB_TIMEOUT`；文档写清范围和 TENT 对应 | HIP 编译单测全过；双机 `ibv_query_qp` 读回：数据 QP 和通知 QP 的 timeout 都随设置从 14 变成 18（两对机器都验证过） | **已转 ready，CI 全绿** |
| [Mooncake#4540](https://github.com/kvcache-ai/Mooncake/pull/4540) | 合并 main（解决 CMakeLists 冲突）；测试改为按 `GPU_PREFIX` 构造（原来写死 `cuda:`，HIP 下必挂）；有目的端提示时绕过选卡缓存；文档写清发送端取舍 | HIP 编译下 7 个测试通过；双机：开启后两端都改走目的 GPU 的网卡（功能正确），但在 AUS 上单流吞吐 38.59 → 28.25 GB/s（发送端跨 PCIe 根读，正是文档写的取舍，AUS 这类每卡一网卡的机器本来就不需要它），已如实写进描述 | **已转 ready**；CI 除 CTest 外全过，CTest 是 runner 编译时失联（基础设施），已用空提交重新触发 |

## 需要你决定的事

1. **srt-slurm#630 总评论里有一句不准确**：Hardware smoke 写了 "Mooncake over the ionic rails"，但那次 Mooncake 实际走的是 TCP（镜像里的 libionic 要求比这些机器的 ionic 内核驱动更新的 ABI，找不到 RDMA 网卡）。不影响 #630 要验证的内容。更正文字在 Mooncake 报告文末，要不要改那条评论由你定。
2. **是否 at 维护者**：两个 Mooncake PR 转 ready 后、48302 更新后，我都没有点名任何人。
3. **#4540 上的 "Qoder Assistant"** 显示 action_required（AI 评审机器人等待批准），是否需要处理由你定。
4. **n10-29 上有别人遗留的 root 进程** `vllm serve Kimi-K3 --tensor-parallel-size 8`（运行 2 小时以上，8 张 GPU 显存约 95%）。holder 35609 曾分到这台机器；e1ab 的工具 `clear_foreign_containers.sh` 会在任务开始前清理这类容器，我没有动它。

## 过程中的问题和更正

- **ionic 用户态驱动不匹配**：两个 sglang-rocm 镜像（v0.5.22、v0.5.23）的 `libionic1 54.0` 和这些机器的内核驱动（ABI 1）不兼容，容器里看不到 RDMA 网卡。挂载宿主机的 `libionic.so.1.1.39` 后正常。以后在容器里用 Mooncake RDMA，都要加这个挂载。
- **我的失误（都已纠正）**：
  - 把"改文件"和"git add/commit"放在同一批并行执行，导致 #4540 的合并提交带上了冲突标记（推送前发现，已 amend），以及两次报告没提交上（随后补交）。
  - 审计脚本把带换行的参数串传给 `bash -c`，第一次审计只生效了第一行参数而 OOM；退出时只杀了外层子 shell，两个 GLM 服务残留了约 20 分钟，发现后手动取消。
  - srt-slurm 主工作区里的本地 `srtslurm.yaml` 让全量测试卡在 mock sweep；在干净 worktree 里确认 PR head `72c4d04` 全量 5531 过、3 个环境失败（与之前相同）。
- holder 35609 在 17:55 和 19:55 两次被抢占；所有验证都在两次抢占之间或之后的新节点上补跑完成。

## CI 状态（持续更新）

见文末"最新快照"。

### 最新快照

- 21:00：sglang#48302 —— 95 过、5 失败（全是可选标签的 `pr-gate` / `*-extra-finish`，与其他 PR 一致）、9 项排队（Base 的 `base-b-test-2-gpu-large` 两片、AMD stage-b/c、Xeon、NPU）。Mooncake#4541 全绿；#4540 重新触发的 run 自 20:27 起在 GitHub 排队（同期 GitHub API 频繁 502）。
