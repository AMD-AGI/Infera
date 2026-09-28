# 执行期间的代码复核

所有复核修改均不改变正在测量的模型、Router 二进制或调度参数。

- 首次选择 score 的单元、HTTP、ZMQ 与配置渲染检查已完成，结果及已知限制见 P-SCORE-REVIEW.zh-CN.md。B1 没有取得性能收益，后续保持 score 关闭。
- 多 worker 正确性探针根据诊断记录确认实际 P/D rank 覆盖，不再假设 N 个新会话必然覆盖 N 个 rank。未覆盖时限次补充不同文档的新会话；最多四批，仍不足则明确失败。每条回复仍检查已知答案、P/D room 与会话亲和。
- host-load 的提交/完成关联键加入 worker 身份，避免不同容器相同 PID、node_id 被误配。冲突样例包含两个 worker，PID 都为 1、node_id 都为 7；验证得到两条记录，耗时分别为 100 ms、200 ms，均值 150 ms，没有覆盖丢失。样例产物：`/tmp/campaign-host-identity-a0whivm8/analysis/runtime-summary.json`。
- 多 worker 捕获按独立诊断目录读取，记录 source_worker/source_node；请求分析按 worker + DP rank 分组。节点资源每台机器只采一次，GPU 利用率与吞吐分母按实际 worker 配置记录。

后续每个点仍需运行时验证：配置、实际 rank/worker、真实接受率下已知答案、MTP 模拟值、有效 chunk 4096、fusion 开启、IndexShare 关闭、进程身份稳定，以及正式窗口错误和取消记录。静态复核不能替代这些检查。

后续placement切换要求上一正式点存在通过检查的review-ready记录。配置生成器用缺失/失败检查记录验证拒绝，再验证12/24/24卡布局与当前32054作业号，全部通过（临时产物`/tmp/campaign-reviewed-layout-zexkcjoh`）。真实接受率gate仍可直接衔接模拟阶段，正式点则必须先完成请求、进程、GPU分母与D前缀覆盖检查。
