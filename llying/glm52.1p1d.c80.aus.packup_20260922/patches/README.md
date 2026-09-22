# 本次AUS镜像携带的补丁

基础镜像由固定Infera源码83e0f6c8和SGLang 20260916 nightly构建；本kit的build/Dockerfile.yihou-overlay另外叠加：

| 文件 | 修改 |
|---|---|
| pr37152.sources.yihou.diff | ROCm HiCache JIT copy rounds、Python元素尺寸检查和K-only host pool HIP gate；仅三处源码，不含上游测试文件 |
| nextn-fusion-fork-4350d37c5.patch | NextN draft shared-experts fusion修复的来源记录，来自xiaobochen-amd/sglang commit 4350d37c5ba7aaa9dfaa91261f57e1aae1fd49da |

NextN实际由build/apply_nextn_patch.py按类声明定位应用，不直接git apply该历史diff。HiCache patch与yihou C40归档文件的SHA256相同：7153a693b24e734ef0f1ca4fc4cb5dbb7a0f486eb205ddbfbe4916ed1f381fb2。

构建时检查源码标记和Python语法，随后scripts/verify_image.py检查两节点镜像；实际运行还验证了HiCache备份与回读。历史镜像ID为5c2716ea18a792b2c1c87e688e0e00afe1d20329cf45640a162a04e2c443160e。

**此overlay不包含router PD/DP rank affinity patch。** 用户选择沿用跨rank。与yihou历史镜像相比，底层nightly及构建来源不同，不能仅凭复用两份修复就声称完整环境等价。

构建日志见../logs/build-base.log和../logs/build-overlay.log；完整步骤见../REPRODUCE.md。
