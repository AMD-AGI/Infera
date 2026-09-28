# B 组（decode radix cache 开启）验证结果

日期：2026-09-24。方案见 [SOLUTION.zh-CN.md](SOLUTION.zh-CN.md)。

## 结论

GLM-5.2 在 PD + DPA + MTP 下开启 decode radix cache 后，启动检查、正确性探测和两轮完整 GSM8K 全部通过。

- 随机 few-shot：strict-match 0.9682，flexible-extract 0.9704。
- 固定 few-shot：strict-match 0.9757，flexible-extract 0.9765；95.1% 的请求在 decode 上复用了前缀。
- 两轮都高于 InferenceX 的阈值 0.9。

尚未验证的项：长前缀下的准确率、淘汰压力、AgentX 负载和性能。

## 配置

与 yihou t2f（`yihou/glm52-agentx-t2sweep-simacc.packup_20260920`，`config.yihou.full.dcar.sh`）对齐。

| 项目 | 值 |
|---|---|
| 机器 | prefill：crsuse2-m2m-135，GPU 2–5；decode：crsuse2-m2m-138，GPU 2–5；RDMA ionic_2..5 |
| 并行 | 两边都是 TP4/DP4，开 DPA；`PD_DP_RANK_AFFINITY=1` |
| decode MTP | EAGLE 5/1/6，`--disable-custom-all-reduce`，`index_share_for_mtp_iteration=false` |
| 其他 | prefill 开 HiCache（ratio 1.5）；chunk 8192/rank；max-running 128；fp8 KV；page 64；tilelang DSA |
| 与 t2f 的差异 | 镜像为 `…-nextnfix-hicache-decrad`（t2f 用的是 `v0519-yihou-0917`）；使用**真实接受率**（t2f 为模拟 3.61）；decode 开 KV events 和 `SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1` |

配置文件：[config.decrad.b.sh](../scripts/bench-harness/results/decrad/config.decrad.b.sh)（在 [config.decrad.a.sh](../scripts/bench-harness/results/decrad/config.decrad.a.sh) 基础上叠加）。测试脚本是 yihou 同节点 kit 自带的 harness，拷贝到 `scripts/bench-harness/`，对跨节点部署没有行为差异。

## 启动检查

全部通过，证据在 `evidence/gate/b-launch/`。

- decode 日志依次输出三条 EXPERIMENTAL 提示：spec + radix、DP attention、decode radix enabled。不再出现 `KV cache is forced as chunk cache`。
- decode 的 4 个 DP rank 都使用 `UnifiedRadixCache`（FULL 组件）。
- decode 的 `server_info`：`disaggregation_decode_enable_radix_cache=true`，`disable_radix_cache=false`，`kv_events_config` 不为空。
- Router 为 decode 登记了 `kv_events_endpoint`，`kv_block_size=64`，`dp_size=4`，并订阅了 4 个 rank。

## 正确性探测

使用 [gate_probe.py](../scripts/gate_probe.py)，结果在 `evidence/gate/b-gate.json`。

- **yihou 的 3 个 coherence 探测：** 2+2 回答 `4`；`PROBE_OK_7391` 原样复述；质数题推理通顺（512 token 用完仍在推理，content 为空，不属于乱码）。
- **前缀复用（2 轮，每轮约 1.1 万 token 共享前缀）：**
  - 所有回答都正确（Nairobi、Z2186），没有乱码。
  - Router 日志显示，第二个请求在 decode 上命中 169/170 块；只 flush decode 之后命中为 0。这证明 decode 复用确实发生。
- **逐 token 一致性不能作为判据：** 两个都没有 decode 复用的对照请求（r1 和 r1b），输出也不一样，只是开头措辞不同，答案相同。命中与未命中的对比，一轮一致，一轮不一致。出现差异的请求都落在不同的 DP rank 上（例如 dp3 对 dp0），推测是跨 rank 的数值差异，未进一步验证。因此改用 GSM8K 做统计判定。

## GSM8K（完整 1319 题，真实接受率，并发 64）

| 轮次 | strict-match | flexible-extract | decode 命中的请求 | 命中块比例 | 耗时 |
|---|---|---|---|---|---|
| 随机 few-shot（InferenceX 原版） | 0.9682 ± 0.0048 | 0.9704 ± 0.0047 | 100/1319 | 1.3% | 5.5 分钟 |
| 固定 few-shot（`first_n`） | 0.9757 ± 0.0042 | 0.9765 ± 0.0042 | 1255/1319（95.1%），中位 10 块，约 640 token | 82% | 4.6 分钟 |

- InferenceX 原版的 few-shot 每题随机抽取，请求之间基本没有共享前缀，所以第一轮基本没有走到复用路径。
- 固定 few-shot 版本只改了抽样方式：[gsm8k_fixedshot.yaml](../scripts/eval/gsm8k_fixedshot.yaml)，通过 [gsm8k_task.sh](../scripts/bench-harness/eval/gsm8k_task.sh) 运行。未命中的请求基本是第一批 64 个并发请求，它们发出时还没有任何缓存。
- 两轮分数不能严格对比，因为 few-shot 示例本身不同。
- 结果目录：`evidence/gate/b-gsm8k/`、`evidence/gate/b2-gsm8k-fixedshot/`；Router 日志：`b-router.log`、`b2-router-fixedshot.log`。

## 发现的问题

1. **镜像中 Infera 有两份。** 一份在 `/opt/infera`（镜像的 WorkingDir，引擎用 `python3 -m` 启动时优先导入），一份装在 site-packages。最初的 Dockerfile 只给前者打了补丁。
   - B 组的引擎导入的是 `/opt/infera`：日志中出现补丁后的代码路径，且 radix 开关被追加。所以以上结果有效。
   - Dockerfile 已改为两份都打补丁，镜像已重建。
2. **Infera 已有的问题：** `_decode_radix_cache_unsupported_reason` 调用了 SGLang 0.5.19 中已不存在的 `ServerArgs.get_model_config()`。见 [ROOT-CAUSE.zh-CN.md](ROOT-CAUSE.zh-CN.md) 的"其他发现"一节。
3. **停机时的显存释放：** 停止后，135 的 GPU 2–5 需要约 100 秒才释放显存。再次启动前需要确认 VRAM 已回到 0。

`evidence/gate/a-launch*` 是一次中途取消的 A 组启动（decode radix 关闭），没有结果。
