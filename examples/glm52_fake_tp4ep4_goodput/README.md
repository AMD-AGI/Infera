# GLM-5.2 TP4/EP4 unprofiled goodput sweep

在指定 fake-decode packup 上记录不同并发的 ITL、TPOT、TTFT、端到端延迟、输出吞吐与逐请求速率，选择满足70/80 tokens/s目标的最大已测并发。数据在 `experiments/` 的独立实验目录下。

**统一报告入口：[REPORT.md](REPORT.md)。** 已将1–128的成功测量、32/40复测、48补测、256启动OOM和此前c32算子profile合并整理。无profile成功测量共15轮、6336请求。256的0.84内存方案只完成准备，未执行；当前不再新增测试。

服务保留 TP4/EP4/DP1、GPU 0–3、MXFP4 模型、FP8 KV、mem-fraction=0.85、ISL10000/OSL500、EAGLE steps5 / draft6 / topk1、模拟接受长度3.61及原始三份 fake-decode 补丁。无 profiler API 调用，无 profiling 环境变量，无 ROCm graph packet capture 覆盖；CUDA Graph 使用原默认 packet 行为。

## 测量口径

- 每请求 decode 速率 = `(output_tokens - 1) / (request_latency - TTFT)`，也就是 `1000 / TPOT_ms`。70 tokens/s 对应 TPOT ≤14.2857 ms，80 对应 ≤12.5 ms。
- 同时给出 P50/P90 TPOT 门限和满足速率门限的请求比例；报告的最大值仅限已测并发，复测时要求每一轮均满足对应条件。
- SLA goodput（实例级）= 达标请求的输出 token 总数 / 测量墙钟时间，与每请求输出速率分别报告。
- 原生 SGLang ITL 将一个 SSE chunk 的间隔均摊到该 chunk 的新增 token。MTP 会批量返回 token，因此另外保存真实 chunk gap 和每 chunk token 数；不能把均摊 ITL 解释成真实 token 到达间隔。
- TTFT/E2E 从实际发出 HTTP 请求开始计时，不包含客户端等待 concurrency semaphore 的时间。
- 每点测量请求数为 `max(128, 8×conc)`，向上取整到 conc 的整数倍。预热 `max(16, conc)` 个请求，按原生客户端规则预热 OSL32，测量 OSL500。边界复测用独立label标记，不混入指定八点的首轮数据。
- 首阶段服务max-running=64，图列表为 `[1,2,4,8,16,24,32,40,48,56,64]`；后续成功服务max-running=128，图列表为 `[1,16,32,64,128]`。两套容量的结果在合并数据中分别标记，未混成同配置复测。并发是客户端上限；实际batch、尾部缩批与retraction由日志保留。
- 同服务连续扫点时，native `accept_length` 是服务生命周期累计值；实验的目标模拟接受长度固定3.61。

`bench_with_details.py` 基于镜像内原生 benchmark，仅增加客户端已测延迟/start time/success 与 SSE gap/token count 的保存，没有模型或服务端 profiling 插桩。修改 diff、原始脚本、SHA256 均保存在实验目录。分析脚本从逐请求数据重算 TPOT/ITL，并与原生统计交叉校验。

原始 prompt 文件只有160条。请求数超过160时，runner 在对应 round 目录循环扩展这些 synthetic prompt，确保 native random/tokenize 路径实际提交计划中的请求数；成功数及实际 token 长度会逐点验证。

## 复现（在 GPU 节点的仓库根目录）

```bash
ROOT="$PWD/examples/glm52_fake_tp4ep4_goodput"
STAMP=$(date -u +%Y%m%dT%H%M%S)
RUN="$ROOT/experiments/$STAMP"
NAME="glm52-goodput-$STAMP"
mkdir -p "$ROOT/experiments"
python3 "$ROOT/prepare.py" "$RUN" \
  --cache-dir /data/cyao1002/glm52_profile_20260910_n04_c32 \
  --name "$NAME" --port 31864
bash "$RUN/scripts/launch.sh" sim server 0.85 64
python3 "$ROOT/run_sweep.py" "$RUN"
python3 "$ROOT/analyze.py" "$RUN"
# 示例：确认候选边界点，运行前按首轮数据选择 conc。
python3 "$ROOT/run_sweep.py" "$RUN" --label confirm \
  --concurrencies 32 40 --repeats 2
python3 "$ROOT/analyze.py" "$RUN"
docker stop --timeout 300 "$NAME"
```

容器名和运行目录必须全新，已有容器不会被替换。launcher 检查八张 GPU 显存空闲；只使用 GPU 0–3。cache 路径沿用本任务已验证的相同镜像编译产物，也可指定新的空目录。

实际镜像固定为 `416d51effc431…`，与历史 packup 的 `b9a83742…` 不同；SGLang `402df1e1e…` 与 AITER `2c71811b…` commit 相同。环境、图配置和模型 config/index 哈希保存在 `state/`。本测试仍使用 synthetic KV / simulated acceptance，不表示真实 prefill 或 PD 正确性。

## 输出

- `REPORT.md`、`summary/analysis/`：统一报告、跨实验CSV/JSON和曲线；合并过程保留历史实验文件。
- `rounds/*/benchmark.jsonl`：全部请求的原始 timing/输出。
- `rounds/*/requests.csv`：逐请求 TPOT、decode/e2e 速率、SLO 是否达标。
- `rounds/*/metrics.json`、`analysis/metrics.csv`：完整延迟分位数、goodput、实际并发等。
- `analysis/SUMMARY.md`、`analysis/summary.json`：扫点汇总与门限判断。
- `rounds/*/server-window.log`、`observations.json`：实际 batch、graph/retraction 证据。

离线重算统一数据的命令见 [REPORT.md](REPORT.md)。`prepare.py` 支持 `--max-running-requests`、`--graph-batch-sizes` 和 `--mem-fraction`；改变这些参数需按实验元数据解释结果，256的OOM不能用未执行方案代替。

```bash
python3 -m unittest discover -s examples/glm52_fake_tp4ep4_goodput -p 'test_*.py'
```
