# Harness 来源

从 `yihou/glm52.p8d8.agentx-sweep.packup_20260920/scripts/bench-harness/`
复制 launch、engine、stop、agentx_bench 及其四个 Python 工具。
原文件 SHA256 记录在 `UPSTREAM_SHA256SUMS.txt`。

本目录的调整：

- `agentx_bench.sh`：挂载根目录改为本 checkout 的真实仓库根目录，适配新位置。
  原始输出和容器工作目录放到 `.tmp/raw/`；`agentx_env.py` 将 UV、pip、XDG、HF
  和 AIPerf cache、客户端临时目录配置在本套件 `.tmp/` 内。
  `agentx_env.py` 另外核对实际 worker 的 RDMA map、Mooncake/probe/HSA 参数和
  MTP 参数，拒绝已撤回或与 PD handoff 不兼容的配置。
- `launch.sh`、`preflight.sh` 的默认输出放到 `.tmp/results/`；
  `engine.sh` 的 AITER JIT 缓存放到 `.tmp/cache/aiter/<node>/<image-id>/`。
- `stop.sh`：移除按 `infera-preflight-` 前缀清理容器的逻辑，避免停止其他测试的 preflight。
- `check_nodes.sh`、`preflight.sh`：复制自当前 checkout 的 `bench/glm5p2_pd/`。
  前者增加必须检测到 8 张 GPU 的检查；后者保留 WRITE byte-verification 和自身容器清理。

所有入口都由上一层 wrapper 传入本套件的 `CONFIG`、`TOPOLOGY`；
请勿不带配置直接运行这里的脚本。镜像和模型不随目录分发，使用要求见套件 README。
