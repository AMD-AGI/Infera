# 参考镜像额外补丁

这两份补丁原样同步自
`yihou/glm52.p8d8.agentx-sweep.packup_20260920/patches/`，用于追溯和重建。
它们已经包含在 `config.sh` 固定的 `sha256:fd7220a57b7d…` 镜像中，运行脚本不会重复应用。

| 文件 | 作用 |
|---|---|
| `pr37152.sources.yihou.diff` | ROCm HiCache 拷贝分组/轮次以及 K-only host pool 的 HIP JIT 开关；只含三份生产源码的修改 |
| `nextn-fusion-fork-4350d37c5.patch` | 为 GLM-5.2 NextN draft 补上自己的 shared-experts fusion architecture 名称 |

NextN 原始 diff 的行号来自另一个源码版本；重建时应按类名定位，不要忽略 patch
失败或模糊地套用。HiCache 补丁在参考归档时仍是未合入的 PR，具体适用范围与
保留理由见 [PD_FIXES_REPORT.md](../PD_FIXES_REPORT.md)。

其余 DSA/PD 修复已经位于本仓库 `deploy/docker/patches/`，不在本目录重复复制。
当前仓库的 Rust router 和 `bench/glm5p2_pd` 已同步同 DP rank 路由代码及参数接线。
完整镜像的能力检查入口为 `scripts/verify_pd_fixes.sh`；仅使用上游原始 SGLang
镜像或仅构建通用 Dockerfile，不等同于这个完整参考镜像。
