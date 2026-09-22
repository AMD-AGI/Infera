# AUS 1P1D C80 复现步骤

本目录包含两轮测试：phase A 关闭 HiCache，phase B 仅开启 prefill HiCache。
两轮都采用跨 rank 路由、decode MTP 模拟接受长度 3.61；不是回答正确性测试。
结果和限制见 README.md 与 analysis/hicache-results.md。

## 前提与固定版本

- 节点：P/control `smci355-ccs-aus-n01-33`，D `smci355-ccs-aus-n02-21`，每节点8张MI355X。
- 模型在两节点同一路径 `/perf_apps/data/models/GLM-5.2-MXFP4`，需282个完整权重分片。
- SSH用户有Docker权限，两节点能访问共享 `/perf_apps/liyingli`。从control节点运行以下步骤。
- Infera构建源码固定在 `83e0f6c86cce34718f369d8669b750fa812c62d0`；新提交只携带复现包，不要求checkout HEAD仍是该commit。
- `scripts/build_image.sh` 从该commit导出源码到builder的独立临时目录；基底SGLang镜像也固定到历史digest。其余安装依赖沿用原Dockerfile，不保证重建后的image ID逐字节相同。
- InferenceX和AIPerf版本由harness固定；客户端会自动获取，Git中不包含`.cache/`。

## 1. 构建或验证镜像

在共享的Infera Git checkout内设置路径（按实际checkout位置调整）：

```bash
REPO=/home/liyingli/code/bench_agentx/Infera
KIT="$REPO/llying/glm52.1p1d.c80.aus.packup_20260922"
bash "$KIT/scripts/build_image.sh" --check
bash "$KIT/scripts/build_image.sh"
```

`--check`只验证源码pin和输入文件，无SSH/Docker副作用。若浅克隆没有该commit，先在仓库执行
`git fetch origin 83e0f6c86cce34718f369d8669b750fa812c62d0`。

正常构建会在P节点编译基础镜像、叠加NextN和HiCache修复，再save/load到D节点，比较两端image ID并验证源码标记/import。overlay的`.dockerignore`只传入构建文件和两份补丁。没有加入router affinity patch。

如复用已验证镜像，可跳过重建，分别在两节点执行：

```bash
docker image inspect --format '{{.Id}}' infera-sglang:v0519-llying-aus-0922-nextnfix-hicache
```

历史实测image ID为 `sha256:5c2716ea18a792b2c1c87e688e0e00afe1d20329cf45640a162a04e2c443160e`。

## 2. 准备容器可访问的共享运行目录

历史NFS home权限为700，Docker无法穿过它挂载运行文件；因此脚本、客户端和结果放在`/perf_apps`。不需要复制模型，也不需要改home权限。

```bash
RUN_ROOT="/perf_apps/liyingli/bench_agentx/glm52-c80-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"
tar -C "$KIT" --exclude='scripts/bench-harness/.cache' -cf - scripts |
  tar -C "$RUN_ROOT" -xf -
export MODEL=/perf_apps/data/models/GLM-5.2-MXFP4
export CONFIG="$RUN_ROOT/scripts/config.phase-b-hicache.sh"
set -a
source "$CONFIG"
set +a
```

phase A改选`config.phase-a.sh`，同样重新source配置。phase A保留历史`PD_DP_RANK_AFFINITY=1`设置作为记录，但该镜像不实现它；phase B显式写0，两者实际均为独立P/D rank选择。

## 3. 检查节点、预检和启动

先检查同名服务。配置使用`llying-aus-1p1d-*`容器名；已有本任务服务时，在确认没有运行中的benchmark后用同一CONFIG停止，再启动新一轮。不要停止其他用户的容器。

```bash
# 仅当需要替换本任务的已有服务时执行：
# bash "$RUN_ROOT/scripts/bench-harness/stop.sh" "CONFIG=$CONFIG" "MODEL=$MODEL"

bash "$RUN_ROOT/scripts/bench-harness/check_nodes.sh" \
  smci355-ccs-aus-n01-33 smci355-ccs-aus-n02-21
OUT_DIR="$RUN_ROOT/results/preflight" \
  bash "$RUN_ROOT/scripts/bench-harness/preflight.sh" \
  smci355-ccs-aus-n01-33 smci355-ccs-aus-n02-21
bash "$RUN_ROOT/scripts/bench-harness/launch.sh" \
  "CONFIG=$CONFIG" "MODEL=$MODEL" "OUT_DIR=$RUN_ROOT/results/launch"
```

等待P/D/router全部healthy。首次模型加载、AITER JIT和HiCache host pool分配可能耗时十多分钟。HiCache版每rank约211.84 GB KV host pool加48.55 GB indexer，需确保足够CPU内存。禁止decode HiCache与当前MTP配置同时开启。

本次历史phase B只替换prefill，保留decode/router；新环境按上面的完整启动过程即可建立同配置。

## 4. 运行AgentX并记录HiCache活动

```bash
POINT="$RUN_ROOT/results/agentx-c80-hicache"
mkdir -p "$RUN_ROOT/results/cache-snapshots"
python3 "$RUN_ROOT/scripts/capture_hicache_metrics.py" \
  --output "$RUN_ROOT/results/cache-snapshots/before.prom"
bash "$RUN_ROOT/scripts/bench-harness/agentx_bench.sh" \
  "CONFIG=$CONFIG" "MODEL=$MODEL" CONC=80 DURATION=1200 "OUT_DIR=$POINT"
python3 "$RUN_ROOT/scripts/capture_hicache_metrics.py" \
  --output "$RUN_ROOT/results/cache-snapshots/after.prom"
python3 "$RUN_ROOT/scripts/bench-harness/tools/collect_agentx.py" "$RUN_ROOT/results"
```

输出目录必须不存在；重跑使用新目录。1200秒是profiling发送窗口，不含数据准备和预热，结束后还有收尾。检查runner退出码、原始records的phase/error、收尾取消、HiCache backup/load-back增量；仅healthy或命令退出0不代表所有记录均有效。

语义`eval/smoke.sh`可用于真实接受率配置；当前强制模拟接受率可能生成重复文本，历史phase A该检查失败。不能把此性能流程宣传为正确性验证。

## 5. 保存产物与停止

完整原始结果保留在本轮`RUN_ROOT/results`。需要Git留档时只带汇总和关键证据，参照ARTIFACTS.md；不要将`.cache`或GB级server metrics JSON加入Git。

```bash
bash "$RUN_ROOT/scripts/bench-harness/stop.sh" "CONFIG=$CONFIG" "MODEL=$MODEL"
```

停止脚本只操作本配置的P/D/router/etcd及任务客户端；其他用户容器不在其范围内。历史实测结束时服务曾保留运行，该状态不是新checkout的保证。
