# NVIDIA/srt-slurm#630 review（Infera frontend）

- PR：[NVIDIA/srt-slurm#630](https://github.com/NVIDIA/srt-slurm/pull/630)，`feat(frontends): add an infera frontend for etcd-registered SGLang workers`，作者 llying-001
- 审阅对象：PR head `5bf86633`（本地 checkout `~/code/bench_agentx/srt-slurm-630`）；对照 Infera `v0.3.3` 源码（`~/code/bench_agentx/Infera`）
- 写于 2026-11-10

## 结论

PR #630 的整体设计是合理的：CI 10 项全绿，可以干净合并，没有冲突（只落后 main 1 个提交，#661）。但它有一个真实的 bug，Devin 也标成了阻塞：示例里的 decode worker 会打开一个 srtctl 不知道的端口。两位 reviewer 的意见大部分成立，需要改代码的有 6 条，只需回复说明的有 4 条，另有 2 条可以不改。

---

## 一、我 review 出的问题

按严重程度排列：

1. **【必须修】decode worker 会自己开一个 KV-events 端口。**
   - `infera.py` 用 `process.kv_events_port is None` 判断是否加 `--no-enable-kv-events`，但 topology 给**每个**进程都分配了 KV-events 端口（`core/topology.py` 约 710 行，`kv_events_port=allocator.next(...)` 没有任何条件），所以这个条件永远不成立。
   - 证据：`tests/snapshots/launch/sglang__infera-disagg.txt` 第 64 行的 decode worker 既没有 `--kv-events-config`，也没有 `--no-enable-kv-events`。
   - Infera 0.3.3 里 `--enable-kv-events` 默认打开（`infera/engine/sglang/args.py`）；argv 里没有显式配置时，`worker.py` 会调用 `free_tcp_port_block(dp_size)` 自己找空闲端口。结果是开了一个 srtctl 端口分配器不知道的 ZMQ 端口，还发布了路由器根本不读的事件。
   - 对应的单测 `_process(..., kv_events_port=None)` 是人为构造的进程，真实 topology 里不会出现，所以测不出这个问题。
2. **【应修】就绪判断太宽松。** `/v1/workers` 返回的 worker 缺 `status` 字段时被当成 active，缺 `disagg_mode` 时被当成 decode。以后 Infera 改了字段名，job 会被误判为已就绪。
3. **【应修】`kv-aware` 拿不到 KV events 时只会悄悄退化。** srtctl 启动路由器用的 `infera.server` 默认策略就是 `kv-aware`（`infera/server/args.py:116`）。如果所有 role 都没开 `kv_events`，路由器不会报错，只是退化成按负载路由，只在连续多次未命中缓存后往日志里打一条 warning（`rust/router/src/policy.rs`："routing has degenerated to load-only"）。srtctl 没有对这种情况做校验。
4. **【设计】绕过了仓库的约定。** readiness 端口用 `env KEY=VAL` 前缀直接塞进 argv，绕过了环境变量层；新加的 hook 返回的是原始 argv，backend 不再掌控 `python3 -m <module>` 这部分命令。
5. **【小问题】** etcd 地址重复构造了一遍；安装脚本 import 了 Infera 的私有函数 `_find_binary`。
6. **【验证缺口】** PR 描述里的硬件验证早于 readiness 端口那次改动（`028898b1`）。按 review 改完后，建议在 AUS 上至少重跑一次 1P1D 冒烟测试。

---

## 二、已有意见逐条：是否合理、要不要改

### Devin（机器人 reviewer，2026-11-09）

| # | 意见 | 判断 | 处理 |
|---|---|---|---|
| D1 | 阻塞：decode 没带 `--no-enable-kv-events`（`infera.py:154`） | 合理，已用代码和 snapshot 核实 | **改**：在 `Backend` 上提供 `get_kv_events_config_for_mode(mode)`，默认返回 `None`（SGLang backend 已经有这个方法）。frontend 根据它的返回值是否为 `None` 来决定加不加这个参数，这样两边以同一个方法为准，再更新 snapshot |
| D2 | 阻塞：单测是人为构造的（`tests/test_infera_frontend.py:183`） | 合理 | **改**：用 backend 的真实 topology 构造进程，或者直接在 snapshot 上断言 decode 带了这个参数 |
| D3 | 文档那句描述和实际不符（`docs/infera.md:36`） | 合理 | D1 修好后这句话自然成立，不用改文档 |
| D4 | etcd 地址应复用 `discovery_env()`（`infera.py:60`） | 合理，改动小 | **改**。顺带还能支持 external etcd，去掉现在的拒绝逻辑也可以 |
| D5 | 安装脚本用了私有的 `_find_binary`（`configs/install-infera.sh:18`） | 部分合理。Devin 建议改成检查 `_bin/infera-router` 路径，但这个路径同样是 Infera 的内部目录结构 | 可以改成路径检查，并保留版本 pin，优先级低 |
| — | readiness 端口用 `sys_port` | Devin 同意现在的做法 | 不用处理 |

### nv-kmcgill53（NVIDIA 的 reviewer，2026-11-09）

| # | 意见 | 判断 | 处理 |
|---|---|---|---|
| K1 | `env` 前缀绕过了环境变量层（`infera.py:137`） | 合理。它不会出现在 dry-run 的环境变量段，也没法被覆盖；而 `_worker_environment_defaults`（`cli/mixins/worker_stage.py`）只给 dynamo 注入变量 | **改**：在 `Frontend` 上加一个 `worker_environment(process, runtime)` hook，默认返回 `{}`，对所有 frontend 合并。Infera 返回 `{INFERA_READINESS_PORT: sys_port}`，argv 里的 `env` 前缀就可以去掉 |
| K2 | hook 返回原始 argv，入口不受控（`backends/sglang.py:372`） | 合理，是设计层面的意见 | **改**：按他给的方案，拆成 `worker_module(backend_type)` 和 `worker_launcher_args(...)` 两个方法，`python3 -m` 仍由 backend 拼。和 K1 一起改。不要新增第三种 `worker_launch`，他自己也说不值得 |
| K3 | 就绪等于"已在 etcd 注册"，是否等于"已能服务"？（`infera.py:158`） | 是个问题，答案是肯定的：Infera 的 `start()`（`infera/engine/sglang/worker.py`）先 `_wait_ready()` 等 SGLang `/health` 返回 200，再解析 page size、组装注册信息 | **回复确认**，并在 `docs/infera.md` 的 Readiness 那一行写明。不用改代码 |
| K4 | 缺字段时的默认值太宽松（`infera.py:163`） | 合理 | **改**：缺字段就不计数，并用 v0.3.3 真实的 `/v1/workers` 响应做测试 fixture |
| K5a | KV events 由两个文件分别决定（`infera.py:155`） | 合理，和 D1 是同一件事 | 随 D1 一起改 |
| K5b | Infera 透传参数的方式变了没人能发现 | 有道理，但没法在 srtctl CI 里测：Infera 解析参数时要 import SGLang 的 `ServerArgs`，srtctl 的 CI 环境里没有 SGLang | **回复**：靠版本 pin 和硬件验证来保证，并在文档里写明 |
| K5c | SGLang 的细节写死在 InferaFrontend 里 | 部分合理。metrics 路径本来就调用基类实现（基类问 backend 的 `prometheus_metrics_path`）；端口逻辑和 `static_router.py` 完全一样，是直连 worker 的通用写法 | **回复说明**，或者复用 `static_router` 的实现，不必大改 |
| K5d | `WORKER_LAUNCHERS` 看起来很通用，实际只支持 sglang | 合理 | 在注释或文档里写明"目前只支持 sglang" |
| K6 | 用 `kv-aware` 却没有任何 role 开 `kv_events`（`examples/sglang/infera-disagg.yaml:35`） | 合理，确实会悄悄退化。注意 `infera.server` 的默认策略就是 `kv-aware`，所以没显式设置策略时也要检查 | **改**：在 `validate()` 里检查，生效策略是 `kv-aware`（包括没设置的情况）而所有 role 都没开 `kv_events` 时直接报错 |
| K7 | `page-size` 和 `served-model-name` 在各 role 重复写，彼此没有关联（`:49`） | 部分合理。路由器的 block size 来自每个 worker 注册时上报的实际 page size，不需要和路由器配置对齐；P/D 两侧 page size 本来就必须一致，这是 SGLang 的要求 | 文档里加一句说明即可 |
| K8 | `INFERA_VERSION` 只靠一个全局变量保证路由器和 worker 版本一致（`:29`） | 合理，但风险低 | 在 `validate()` 里加一条：role 或 frontend 级别的 `INFERA_VERSION` 不能和全局的不同 |
| — | `WORKER_LAUNCHERS[...]` 可能抛 `KeyError`（`infera.py:140`） | 实际触发不到：加载配置时 `SrtConfig._validate_frontend` 会检查所有 active role 的 backend 都是 `required_backend` | 可以改成显式报错，优先级低 |
| — | snapshot 里的 `127.0.0.1` 来自 mock | 说的是事实 | 不用改 |
| — | `get_hostname_ip` 每个 worker 都调用一次（`infera.py:146`） | 这个函数已经加了 `@functools.cache`（`core/slurm.py`），#651 也已在分支里 | 回复确认即可 |

### 建议的修改顺序

1. 先改 D1、D2、K5a，修掉唯一的真实 bug。
2. 再改 K1 和 K2，统一启动方式和环境变量。
3. 然后改 K4、K6、K8、D4，补上校验和细节。
4. 最后改文档（K3、K5b、K7），重跑一次硬件冒烟测试，再统一回复两位 reviewer。

---

## 三、零基础讲解：what / why / how

### 先认识几个角色

- **srt-slurm（命令行工具叫 `srtctl`）**：NVIDIA 的工具。写一份 YAML recipe，就能在 Slurm 集群上一键拉起整套推理服务：基础服务（etcd、NATS）、多个推理 worker、前端路由器，最后跑 benchmark。它内部有几块主要抽象：
  - **backend**：推理引擎，比如 sglang、vllm、trtllm，负责拼出 worker 的启动命令；
  - **frontend**：请求入口或路由器，负责转发请求和判断服务是否就绪；
  - **topology / 端口分配器**：决定哪个进程用哪个节点、哪些 GPU、哪些端口。仓库有一条规则：**所有监听端口都必须来自分配器**。
- **frontend 分两种：**
  - **静态路由器**：启动时在命令行里把 worker 的 URL 列表告诉它；
  - **动态 frontend**：worker 自己去一个"发现平面"（比如 etcd）登记，路由器通过监听 etcd 来发现 worker。Dynamo 就属于这一种。
- **Infera**：AMD 的推理服务框架，包含两部分：
  - `infera-router`：用 Rust 写的 KV-aware 路由器。每个 SGLang worker 会通过 ZMQ 广播"我缓存了哪些前缀块"（这就是 KV events），路由器据此把请求发给缓存命中最多的 worker，减少重复计算；
  - `infera.engine.sglang`：一个包装器。它启动 SGLang 原生的 server，等 server 就绪后去 etcd 登记自己。

### Why（为什么要做）

InferenceX 是 SemiAnalysis 的公开推理 benchmark，它的多节点 PD 测试就是用 srt-slurm 拉起的。我们要在 InferenceX 上提交"Infera + SGLang 跑 GLM-5.2"的结果（InferenceX#4629），所以 srt-slurm 必须支持 `frontend.type: infera`。AUS 上的 e2e 实测（Infera 仓库 `llying/inferencex-aus-e2e-20261029/`）就是"srt-slurm + #630"跑出来的。

### How（怎么实现）

一次 job 的流程如下：

```text
recipe: frontend.type: infera
  │
  ├─ infra 节点：srtctl 自动起一个 etcd（infera 只隐含 etcd，不需要 NATS）
  │
  ├─ 每个 worker（prefill / decode）：
  │    SGLang backend 照常拼参数；因为 frontend 是 infera，命令开头
  │    "python3 -m sglang.launch_server" 被换成
  │    "python3 -m infera.engine.sglang --discovery-backend etcd ..."，后面接原来的 SGLang 参数
  │    → 包装器启动 SGLang → 等 /health 正常 → 把 URL、角色、bootstrap 地址、DP 数、KV events 端点写进 etcd
  │
  ├─ frontend 节点：python3 -m infera.server --router-backend rust --discovery-backend etcd ...
  │    → 路由器监听 etcd 发现 worker，直接订阅各 worker 的 ZMQ KV events
  │
  ├─ 就绪判断：srtctl 轮询路由器的 /v1/workers，按角色统计 active 的注册数，够数后开始 benchmark
  │
  └─ 安装：setup_script（install-infera.sh）在每个容器里执行 pip install amd-infera==0.3.3
```

### 几个关键设计选择

- **为什么是动态 frontend：** `infera-router` 只能从 etcd 发现 worker，不接受命令行传入的 URL，所以继承 `DynamicFrontend`。
- **为什么 worker 仍然算 `direct`：** 真正提供服务的是 SGLang 自己的 server，包装器只多做了一步登记。这样直连 worker 已有的各种处理都能原样复用，比如 metrics 走 HTTP 端口、`--skip-server-warmup`、不注入 Dynamo 的环境变量。
- **只加一个新 hook：** `Frontend.direct_worker_launcher()` 默认返回 `None`。只有 Infera 返回包装器命令，SGLang backend 把它接在原参数前面，其他 backend 和其他 frontend 都不受影响。K2 就是对这个 hook 形式的意见。
- **端口都来自分配器：** prefill 的 KV events 端口用分配器给的端口块；readiness 端口用进程的 `sys_port`。D1 的 bug 正好是 decode 这一侧漏掉了这条规则。
- **路由器参数由 srtctl 控制：** 发现方式、etcd 地址、传输方式这些参数由 srtctl 统一设置，用户在 `frontend.args` 里设置它们会直接报错；其余参数（路由策略、权重等）原样透传。

### 代码地图

| 文件 | 作用 |
|---|---|
| `src/srtctl/frontends/infera.py` | 新的 frontend：参数校验、隐含 etcd、worker 启动命令、路由器命令、就绪统计 |
| `src/srtctl/frontends/base.py` | 新增的 hook `direct_worker_launcher` |
| `src/srtctl/backends/sglang.py` | 调用 hook，把包装器命令接在 SGLang 参数前面 |
| `configs/install-infera.sh` | 在容器里安装 Infera |
| `examples/sglang/infera-disagg.yaml` | 示例 recipe |
| `tests/snapshots/launch/sglang__infera-disagg.txt`、`tests/test_infera_frontend.py` | 测试 |
| `docs/infera.md` | 文档 |

---

## 附：sglang#48042 在 MI355X 上的测试结果（同一轮回答开头的内容）

作业 35557，节点 n01-33，镜像 `lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261109`，PR head `59bda49f`。脚本 `/apps/liyingli/jobs/sgl48042_tests.sh`，输出 `/apps/liyingli/jobs/llying-sgl48042-ut-35557.out`。

- **`test_kvcacheio.py`**：148 passed，50 skipped。跳过的 50 个都是 HIP 上本来就跳过的用例。
- **兼容性**：新测试在镜像自带的旧 wrapper 下也是 4/4 通过，所以 CI 不管用新旧哪个 sgl-kernel 都没问题。
- **计时**：一次驱逐，所有拷贝都逐字节正确：

| 数据 | 不设变量（默认，quota 16） | `SGLANG_HICACHE_BLOCK_QUOTA=2` | `=16` |
|---|---|---|---|
| MLA KV（6.14 GB） | 126.6 ms | 265.4 ms | 126.7 ms |
| DSA indexer（0.35 GB） | 7.8 ms | 11.0 ms | 7.7 ms |

需要说明：作业里第 1 步那行"打印签名"的检查写错了，打印出来的是镜像里的旧模块，所以它不能证明 pytest 用的是 PR 版 wrapper。这不影响结论：第 3 步直接加载 PR 版 wrapper，"默认值走 16"已经由计时数据证实；而且 `test_kvcacheio.py` 用新旧两个 wrapper 都能通过。

这些结果还没贴到 PR 上，需要确认后再贴。
