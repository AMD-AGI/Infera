# GLM-5.2 MXFP4 8P4D，prefill 改为 PP4：原生 ATOM + Infera

[`../8p4d-atom-infera`](../8p4d-atom-infera/README.md) 的变体，按
[`pcp_and_alternatives.md`](../8p4d-atom-infera/analysis-new/pcp_and_alternatives.md) 的替代方案把 prefill
由 TP8 + DPA 改为 TP1×PP4：2 个 prefill 实例（每个 TP1×PP4，层划分 20/20/20/18，共 8 卡）与 1 个 decode
（TP4 + DCP4，4 卡），共 12 卡。引擎参数按 ATOM `recipes/Agentic-GLM-5.2.md` 的 1P1D 配置（upstream `c3a88b5c`），
镜像、patch、AgentX 方法与上一组相同。过程与问题见 `.record/`。

2026-10-09 CRS 复测见 [`plan/crs-baseline.md`](plan/crs-baseline.md)。136/138 的镜像、AgentX 三档配置及数据集、
GSM8K 数据已准备完成。用户已于本日授权 GPU 测试，并确认 C48/C80/C120 各 3600 s；
正确性验证已通过（GSM8K 97.5%）；C48/C80/C120 均完成一小时测量，三档数据有效性检查通过。
原始数据已归档并校验，测试容器已清理。
已完成结果及与原环境的比较见 [`results/crs-20261009-report.md`](results/crs-20261009-report.md)。
镜像与依赖清单见 [`.record/crs-image-manifest.json`](.record/crs-image-manifest.json)。

## 目录

- `config.sh`：节点、IP、镜像、模型、端口、引擎与 AgentX 参数；所有脚本支持用 `KEY=VALUE` 覆盖。
- `scripts/`：`build_image.sh`、`up.sh`、`down.sh`、`smoke.sh`、`agentx.sh`、`sweep.sh`、`summarize.py`。
  `agentx.sh PREPARE_ONLY=1 CONC=48` 只准备依赖、数据和 replay 命令，不需要运行中的服务，也不发送推理请求。
  `common.sh`、`sweep.sh`、`summarize.py` 由上一组复制；`up.sh`、`down.sh`、`smoke.sh`、`agentx.sh` 改为多个 prefill 实例。
- `.tmp/`（不入库）：`gsm8k.sh`、`needle_probe.sh`、`vram_sample.sh`，运行目录 `runs/<RUN_ID>/`、日志 `logs/`。
  `up.sh` 启动每个容器后由控制节点收集 `docker logs -f`，写入 `runs/<RUN_ID>/logs/<name>.live.log`，
  作业被抢占、来不及执行 `down.sh` 时日志仍保留到抢占时刻；`down.sh` 另存完整的 `<name>.log`。
- `results/`：通过的 AgentX 聚合 JSON 与汇总表。

镜像沿用 `infera-atom:nightly_202609221542`（基础镜像 `rocm/atom-dev:nightly_202609221542`，ATOM `d9f0720e2f99`），
由 `../8p4d-atom-infera/docker/Dockerfile` 与 `../8p4d-atom-infera/patch/` 构建，未改动。InferenceX checkout、
AgentX venv 与数据集缓存使用 `../8p4d-atom-infera/.tmp/cache`（`CACHE_DIR`）。

## 默认配置

节点随作业重新运行而变化，`config.sh` 的默认值为最近一次分配的节点（作业 32947 第二次运行的 n02-[25,29]）。

- Prefill：`smci355-ccs-aus-n02-29`。实例 i 使用 GPU 组 i（`0,1,2,3` 与 `4,5,6,7`），HTTP 端口 `19001 + 2i`，
  握手端口 `21301 + 20i + stage`。`-tp 1 -pp 4 --enforce-eager --max-num-seqs 512 --max-num-batched-tokens 8192`，
  显存比例 0.85，`VLLM_PP_LAYER_PARTITION=20,20,20,18`，`ATOM_MOONCAKE_MATCHED_RAILS=auto`。
  LMCache：`LMCACHE_MAX_LOCAL_CPU_SIZE=160`（ATOM 按层数把 160×4 GiB 分给 4 个 stage，2 个实例共 1280 GiB 锁页内存，
  小于 GTT 上限 1511 GiB），每个 stage 运行 lookup server（`LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3`），
  `OFFLOAD_MIN_LOAD_TOKENS=0`、`OFFLOAD_PROFILE=1`、`LMCACHE_NUMA_MODE=auto`。
- Decode：`smci355-ccs-aus-n02-25`，GPU 0-3，`-tp 4 --decode-context-parallel-size 4`，显存比例 0.94，
  `--max-num-batched-tokens 2048`，`AITER_REUSE_IDENTICAL_COMM_GROUPS=1`；`--max-num-seqs` 与 cudagraph 尺寸按 `2*CONC`。
- 两侧：FP8 KV、block 16、prefix caching、MTP（CONC ≥ 48 为 K3、forced acceptance 2.99，两侧都传入；
  `MTP_AL=` 关闭）、`--timeout-keep-alive 900`、在线 PTPC FP8 量化（层 0-77 的专家与 MTP 层 78 保持原精度）。
- 路由：Infera Python 路由，round-robin。两个 prefill 实例各有独立的 KV 与 LMCache，同一会话的各轮在两个实例之间轮流分配。
- etcd（23379）、路由（18000）、AgentX 客户端在控制节点（decode 节点）；decode 19002，握手端口 21311。

与上一组（`../8p4d-atom-infera/results/`，C48/C80/C120）的差异：

| 项目 | 上一组 | 本组 |
|---|---|---|
| Prefill 并行 | 1 个 TP8 + DPA（8 个 DP rank，各自的 KV 与 LMCache），会话亲和 | 2 个 TP1×PP4，round-robin |
| Prefill token 预算 | 16384 | 8192 |
| LMCache | 每个 DP rank 160 GiB（8 个私有池，共 1280 GiB），`OFFLOAD_MIN_LOAD_TOKENS=8192` | 每个实例 640 GiB（2 个池，共 1280 GiB），4 个 stage 各自 lookup，`OFFLOAD_MIN_LOAD_TOKENS=0` |
| Decode 显存比例 / token 预算 | 0.95 / 16384 | 0.94 / 2048，加 `AITER_REUSE_IDENTICAL_COMM_GROUPS=1` |
| block size | 64 | 16 |
| HTTP keep-alive | 5 s（默认） | 900 s |
| forced acceptance | 只在 decode | prefill 与 decode |
| 驱动 | amdgpu 6.14.14 | 早期 Slurm 尝试为 6.16.13；本次 CRS 正式复测为 6.14.14 |

## 运行

脚本在控制节点执行；长时间任务用 `nohup setsid` 脱离 SSH 会话（`cd` 写成单独的语句）。
更换节点时修改 `config.sh` 的 `PREFILL_NODE/IP`、`DECODE_NODE/IP`（或用 `KEY=VALUE` 传入）：

```bash
ssh -J cyao1002@dccs-1334-slurm.prov.aus.ccs.cpe.ice.amd.com smci355-ccs-aus-n02-25
cd /apps/tas/yaoc/research/topic/glm-5.2-pd-opt/Infera-glm-5.2-2p1d/yaocheng/8p4d-atom-infera-prefill-pp4

bash scripts/build_image.sh                       # 两台节点并行拉取基础镜像并构建
bash scripts/up.sh CONC=48 MTP_AL= RUN_ID=acc-c48  # 精度验证用：关闭 forced acceptance
bash scripts/smoke.sh
bash .tmp/gsm8k.sh; bash .tmp/needle_probe.sh
bash scripts/down.sh

# 正式测试：每档冷启动，3600 s，每 lane 预热 10，结束后写 results/summary.{md,csv}
nohup setsid bash scripts/sweep.sh > .tmp/logs/sweep.log 2>&1 < /dev/null &
```
