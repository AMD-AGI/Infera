# 服务释放状态更新

2026-09-24 09:43 UTC 后核查。

旁路观察结束后只停止了发压和采样，模型服务曾保留空闲。用户指出 HiCache 延迟释放后，立即尝试先停止 Prefill，再停止 Decode 和辅助容器；SSH 在容器身份检查前失败，因此不能记录为停止成功。

- Prefill n02-29：Slurm 报告 09:39:55 意外重启，状态 DOWN。旧进程随重启结束，原 host HiCache 应随重启释放；未直接核实新启动后的容器/VRAM。
- Decode n06-25：节点已被分配，当前用户没有活动作业，pam_slurm_adopt 拒绝访问。旧 Decode 容器是否清理未知。
- 作业31699已不能查询；集群 accounting 禁用，无法据此还原终止原因。
- 未找到 watchdog 清理事件，不能用预设了 watchdog 代替实际清理证据。

需要恢复节点访问或由管理员核实残留。原 Prefill 容器 ID 为 `18b776a617b69e04689549bf994c56f14eec581b2e7ab9865e50888bbcbcc4ff`；原 Decode 容器 ID 为 `da0fdd177948d889b2e07ef9af94229a892eee3aba7665a0c8300029f8419676`。名称分别为 `llying-adaptive-31699-prefill-0` 和 `llying-adaptive-31699-decode-0`。
