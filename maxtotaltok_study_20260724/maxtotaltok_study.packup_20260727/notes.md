# Notes — 坑、弯路、纠错（what / why / how / context）

完整叙事见父目录 `working_process.md`。这里是蒸馏出的耐用教训。

## A. 机制认知的修正链（R3 → G1 → G2 → G3，重要）

初版结论"DSv4 必须靠超大池 / 崩是天生的"是**错的**，被用户两个质疑逐步纠正到正确机制：

1. **R3（131072 池, max_running 64/rank）崩** → 初判"DSv4 无优雅 retract 所以要大池"。
   - **what**：池撑到 100% → `retract_decode() schedule_batch.py:2423 raise NotImplementedError()` → 崩。
   - **对**：DSv4 无优雅 retract 是真的。**错**：以为"必须超大池"。

2. 用户质疑①"是不是没限制 max-running/context？" → 查日志发现**各轮 max_running 全被固定 64/rank，池砍了准入没砍**。
   自洽配置应"池 + max_running 联动"。

3. 用户质疑②"decode 有准入队列为什么还能打爆？" → 发现准入按 `full_len(prefill~8192)+num_reserved_decode_tokens(512)`
   预留（`disaggregation/decode.py:379`），**不是满 ctx**。per_req 实测 = 131072/12 = **10,923 tok**（含 page 对齐）。

4. **G2（131072 池, max_running 8/rank）不崩但吞吐塌到 8-174 tok/s** → 用 llm-pd-bottleneck-finder 5态分析：
   - **what**：running 被砍到 2.8（自然 16），transfer-in 涨到 9.2。
   - **why**：max_running=8 < 自然 running 16 → transfer 管道预留 slot 速度 > 8-cap 消化 → 池被 transfer-waiting
     塞满、running 饿死。**本质是自己掐死 decode，非 KV 不足、非 wire 慢**。

5. **G3（大池, conc=512 过载）** → decode running 仍只 20% cap、retract=0；过载表现为 **router circuit-breaker
   拒绝**（8867 个 "No available prefill workers"）。→ **8k/1k 1P1D 是 prefill-bound，decode KV 永不是瓶颈**。

**最终三态**：稳（池够）｜崩（池<需求→NotImplementedError）｜塌（max_running<自然running→掐死）。

## B. 环境事故教训（2026-07-26 下午，代价惨重）

1. **换轮没等 VRAM 完全释放 → decode 启动即 `Not enough memory` 死**。
   - **what**：重起 decode 时 avail mem 208 vs 正常 277（前一实例 ~69GB/卡没释放）→ scheduler init OOM 死。
   - **how**：换轮/换池**必须**轮询 `rocm-smi --showmeminfo vram` 到 sum<10GB 再起。这是自己 working_process
     里早写过的纪律，还是违反了。用 `scripts/reset.sh` 内置等待。

2. **decode OOM 死后留 KFD 僵尸进程占 GPU，容器 restart 无效**。
   - **what**：死掉的 decode 留 8 个 `<defunct>` KFD 进程（PID 575427-575434）各占 ~140GB/卡。
   - **why**：僵尸的 GPU 分配由 **host kernel（KFD）** 持有，容器 restart 不回收。
   - **how**：host 上 `rocm-smi --showpids` 找占 VRAM 的 pid → 逐个 `kill -9 <pid>` → VRAM 才回 2.2GB。

3. **router↔decode circuit-breaker 经历 crash/恢复后会 flapping**。
   - **what**：单发 smoke 能过，但 conc=128 warmup 就 "No available decode workers (circuits open)"。
   - **why**：早前 crash/timeout 触发 circuit-breaker 置 open，重起 router 未必清干净。
   - **how**：kill decode leg + router 一起重起（新 pair）。router 进程名是 `sglang::router` 不是
     `sglang_router`，`pkill -f sglang_router` 杀不掉 → 用 `mtt_kill_router.sh` 按端口 pid 清。

## C. 起测硬坑（复现必看）

1. **libionic overlay 必做**：容器内 ionic provider 不匹配 → `ibv_devinfo PORT_ACTIVE=0` → mooncake 静默退
   TCP。overlay host libionic 到 soname + libibverbs 软链 → PORT_ACTIVE=8。容器重建后要重做。

2. **bench_serving import 错**：`ModuleNotFoundError: sglang.benchmark.datasets`。0.5.13 镜像把 sglang 当
   PEP-420 namespace pkg 载（`sglang.__file__`=None），benchmark/ 子树不在 import path。
   **修复**：`export PYTHONPATH=/sgl-workspace/sglang/python`（已在 bench.sh）。

3. **换轮重启顺序**：legs → router → bench。**任何 leg 重启后 router 必须一起重启**（circuit-breaker 记着
   旧 URL）。

4. **decode stat 行字段是 `#full token: N` / `#swa token: N`**（不是 `#token:`），准入预留看
   `pre-allocated usage` + `#transfer-req`。

5. **冷 NFS 首次权重加载慢**（同一节点首次可达 5-9min，warm 后 ~85s）。低 CPU%+冻结日志 ≠ 死锁；
   py-spy 连采两帧看 MoE loader 帧是否移动才能判定。cuda-graph capture ~50s。

## D. 关键数值（0.5.13 DSv4, 8k/1k）

- 每 full-token KV = **32,781.44 B**（复合，含 swa/c4/c128/state）。
- per_req_cost（decode 单请求峰值占用）≈ **10,900 tok**（= ISL+OSL+page 对齐，实测 131072/12=10923）。
- decode 自然 running/rank = conc/dp（conc128→16, conc512→25，受 prefill-bound 限）。
- 绝对数字随 sglang 版本漂：0.5.13 decode gmu0.90 full_token=3,895,296（与历史 0.5.13 参考运行
  pd_1p1d_dpa_8k1k 逐字一致），但 0.5.15 同 gmu 是 9,217,792。**必须实测勿套旧版数**。
