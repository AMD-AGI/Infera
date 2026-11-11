# srt-slurm#630 原理与执行过程详解（零基础版）

这份文档用一个具体例子，从"写一份 recipe"讲到"一个请求怎么被路由、KV 怎么被复用"，配数据流图和接口调用流程图。

- 讲的是**按 review 修改后**的代码（本地 worktree `~/code/bench_agentx/srt-slurm-630-review`）；和 PR 当前 head 的差别集中在第 9 节。
- 例子用仓库自带的 `examples/sglang/infera-disagg.yaml`，命令取自它的 launch snapshot（mock 环境，所以 IP 都是 `127.0.0.1`；真实集群里是各节点的 IP）。
- 图都用纯文本画，任何编辑器都能看。

---

## 0. 一句话

**srtctl 是"按 recipe 在 Slurm 上把一整套推理服务拉起来并压测"的工具；这个 PR 让它多支持一种前端：Infera 路由器。** worker 仍然是 SGLang 自己的 server，只是外面套了一层 Infera 的启动器，负责把自己登记到 etcd；路由器从 etcd 发现 worker，按"谁缓存了这段前缀"来分发请求。

---

## 1. 先认识六个角色

| 角色 | 是什么 | 在这个 PR 里做什么 |
|---|---|---|
| **srtctl** | NVIDIA srt-slurm 的命令行工具，在 Slurm 作业的 head 节点上运行 | 读 recipe、校验、分配端口、用 `srun` 依次拉起 etcd / worker / 路由器 / benchmark，判断何时就绪 |
| **Slurm `srun` + 容器** | 集群调度器；每个服务都是一个 `srun` 步骤，跑在容器里 | 承载下面每一个进程 |
| **etcd** | 一个小型键值数据库 | 当"通讯录"：worker 把自己的地址写进去，路由器从里面读 |
| **SGLang worker** | 真正跑模型的推理引擎（`sglang.launch_server`） | 分 prefill（算 prompt、产生 KV）和 decode（逐 token 生成）两种角色 |
| **Infera 启动器** `infera.engine.sglang` | Infera 提供的一层包装 | 启动 SGLang、等它就绪、把它登记到 etcd |
| **infera-router** | Infera 的路由器（Rust 写的），由 `python3 -m infera.server` 启动 | 从 etcd 发现 worker、订阅 KV events、把请求发给合适的 prefill 和 decode |

两个关键词：
- **KV cache / 前缀复用**：模型处理一段 prompt 会产生 KV cache。如果下一个请求的开头和之前某个请求相同（比如同一段 system prompt），而它被送到了缓存着这段 KV 的 worker，就不用重算这段前缀。
- **KV events**：SGLang 每当缓存了新的块或者删掉旧的块，就通过 ZMQ 广播一条消息（"我存了哪些块的哈希"）。路由器订阅这些消息，就能知道每个 worker 缓存了什么。

---

## 2. 具体例子：一份 recipe

```yaml
setup_script: install-infera.sh        # 每个容器启动前先跑这个脚本：pip 安装 Infera
environment:
  INFERA_VERSION: "0.3.3"              # 装哪个版本（路由器和 worker 必须一致）

frontend:
  type: infera                         # ← 本 PR 新增的前端类型
  args:
    router-policy: "kv-aware"          # 原样传给路由器

engine: sglang                         # infera 只能配 sglang
roles:
  prefill:
    workers: 1
    gpus: 1
    kv_events: true                    # prefill 发布 KV events（kv-aware 路由靠它）
    args: { page-size: 64, served-model-name: "Qwen/Qwen3-0.6B", ... }
  decode:
    nodes: colocate
    workers: 1
    gpus: 1                            # decode 不发布 KV events
    args: { page-size: 64, served-model-name: "Qwen/Qwen3-0.6B", ... }

benchmark:
  type: "sa-bench"                     # 压测客户端
```

srtctl 读完它，最终会在集群上拉起这些进程（来自 launch snapshot）：

| srun 步骤 | 节点 | 进程 |
|---|---|---|
| `service_etcd` | node-01 | etcd，监听 2379 |
| `prefill_0_node-01` | node-01 | Infera 启动器 → SGLang prefill |
| `decode_0_node-02` | node-02 | Infera 启动器 → SGLang decode |
| `infera_router_0` | node-01 | `python3 -m infera.server` → `infera-router`，监听 8000 |
| benchmark | node-01 | `sa-bench` 压 `http://localhost:8000` |

（还有 dcgm-exporter、node-exporter 等监控进程，与本 PR 无关，略。）

---

## 3. 总体时间线

```text
时间 ──────────────────────────────────────────────────────────────────────────────►

[srtctl 加载 recipe]                         ← 第 4 节：校验
   │  不通过就报错退出（dry-run 也会走到这里）
   ▼
[分配 topology 和端口]                        ← 第 5 节
   ▼
[srun: etcd]                                ← 第 6 节
   ▼
[srun: prefill worker] [srun: decode worker]  ← 第 7 节：每个 worker 安装 Infera → 启动 SGLang → 等就绪 → 登记 etcd
   ▼
[srun: 路由器]                               ← 第 8 节：从 etcd 发现 worker、订阅 KV events
   ▼
[srtctl 轮询路由器 /v1/workers，直到 1 prefill + 1 decode 都 active]   ← 第 8.3 节
   ▼
[srun: benchmark 开始发请求]                  ← 第 9 节之前的"一个请求的旅程"
```

---

## 4. 阶段 0：加载和校验（还没占用任何 GPU）

```text
srtctl apply / dry-run
  └─ SrtConfig(...)                         读 recipe
       ├─ _validate_frontend()
       │    ├─ get_frontend("infera") → InferaFrontend      （按 frontend.type 查注册表）
       │    ├─ required_backend == "sglang"？               （检查每个 active role 的引擎）
       │    └─ InferaFrontend.validate(config)
       │         ├─ frontend.args 里不能写 srtctl 自己管的参数（etcd-endpoint、port 等）
       │         ├─ 不能用 external etcd
       │         ├─ router-policy（没写就算 kv-aware）是 kv-aware 时，至少一个 role 要开 kv_events   ← R4 新增
       │         └─ environment / frontend.env / roles.*.env 里的 INFERA_VERSION 必须一致         ← R5 新增
       └─ implied_services() → [etcd]       infera 隐含需要 etcd（不需要 NATS）
```

**例子**：如果把 recipe 里 prefill 的 `kv_events: true` 删掉，加载时就会直接报错：

```text
frontend.type: infera routes kv-aware (infera.server's default router-policy) on KV events, but no
role publishes them, so the router would route on load alone; set kv_events: true on the prefill
(or agg) role, or set frontend.args.router-policy
```

为什么要在这一步拦：路由器在这种情况下不会报错，只会悄悄按负载路由，压测结果会被误读（详见意见清单 R4）。

---

## 5. 阶段 1：分配 topology 和端口

srtctl 先决定每个进程放在哪个节点、用哪些 GPU、用哪些端口。**规则：所有监听端口都由分配器发放**，这样同一节点上的多个进程不会抢端口。

例子里的分配结果（mock 环境）：

| 进程 | 节点 | HTTP 端口（SGLang server） | system 端口 | bootstrap 端口 | KV-events 端口 |
|---|---|---|---|---|---|
| prefill_0 | node-01 | 6100 | 7500 | 7200 | 3000（会用） |
| decode_0 | node-02 | 6100 | 7501 | — | 分配了一个（但不用） |

注意最后一列：**分配器给每个进程都发了 KV-events 端口**，不管它的 role 开没开 `kv_events`。这正是 R1 那个 bug 的根源，后面第 7 节会看到怎么处理。

- **system 端口**：原本是给 Dynamo 运行时用的；infera 没有 Dynamo，所以拿它当 Infera 启动器的 readiness 端口（见第 7 节）。
- **bootstrap 端口**：SGLang PD 模式下，decode 通过 prefill 的 bootstrap 端口来接 KV。

---

## 6. 阶段 2：启动 etcd

```text
/configs/etcd --data-dir /tmp/etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://<infra 节点 IP>:2379
```

worker 和路由器都用 `--etcd-endpoint http://<infra 节点 IP>:2379` 指向它。两边的地址由同一个函数 `etcd_endpoint(runtime)` 拼出，所以一定一致。

---

## 7. 阶段 3：启动 worker（本 PR 改动最多的地方）

### 7.1 srtctl 里的接口调用流程

以 prefill_0 为例：

```text
WorkerStageMixin.start_worker(process=prefill_0)
  │
  ├─ 组装环境变量
  │    env = _worker_environment_defaults(process)
  │    │      ├─ {"HEAD_NODE_IP": ...}
  │    │      └─ frontend.worker_environment(process, runtime)          ← 新 hook（R2）
  │    │           InferaFrontend 返回 {"INFERA_READINESS_PORT": "7500"}   （= process.sys_port）
  │    ├─ + 每个 role 的 env、recipe 的 environment（可以覆盖上面的值）
  │    └─ + backend.get_process_environment(process)
  │
  ├─ 拼命令
  │    cmd = SGLangBackend.build_worker_command(process, ...)
  │    │      ├─ module = frontend.worker_module("sglang")              ← 新 hook（R2）
  │    │      │     InferaFrontend → "infera.engine.sglang"
  │    │      │     （其他前端返回 None → 用 "sglang.launch_server" 或 "dynamo.sglang"）
  │    │      ├─ cmd = ["python3", "-m", module]
  │    │      ├─ cmd += frontend.worker_launcher_args(backend, process, runtime)   ← 新 hook（R2）
  │    │      │     InferaFrontend 返回 Infera 自己的参数（见 7.2），其中：
  │    │      │       if backend.get_kv_events_config_for_mode(mode) is None:    ← R1 修复
  │    │      │           加 "--no-enable-kv-events"
  │    │      ├─ cmd += SGLang 自己的参数（--model-path、--port、--disaggregation-mode ...）
  │    │      └─ if backend.get_kv_events_config_for_mode(mode) 不为 None:
  │    │             cmd += ["--kv-events-config", '{"endpoint": "tcp://*:<分配的端口>", ...}']
  │    │
  └─ start_srun_process(command=cmd, env_to_set=env, bash_preamble=<运行 install-infera.sh>, ...)
```

两个要点：
- **KV events 由同一个方法决定**：backend 加不加 `--kv-events-config`、frontend 加不加 `--no-enable-kv-events`，都看 `get_kv_events_config_for_mode(mode)`，也就是 recipe 里这个 role 的 `kv_events`。两边不可能不一致。
- **readiness 端口走环境变量层**：它出现在 worker 的 `Env:` 日志和 snapshot 的 `env_to_set` 里，用户也能在 recipe 里覆盖；命令里不再有 `env KEY=VAL` 前缀。

### 7.2 拼出来的最终命令

prefill_0（node-01），环境变量里有 `INFERA_READINESS_PORT=7500`：

```text
python3 -m infera.engine.sglang \
  --discovery-backend etcd --etcd-endpoint http://127.0.0.1:2379 \      ← Infera 自己的参数
  --advertise-host 127.0.0.1 \                                           （worker_launcher_args 返回的）
  --request-transport http --kv-event-transport zmq --kv-events off \
  --model-path /model --served-model-name Qwen/Qwen3-0.6B \             ← 下面都是 SGLang 的参数
  --host 0.0.0.0 --port 6100 --nccl-port 17500 --enable-metrics \        （和直接跑 sglang.launch_server 时一样）
  --disaggregation-mode prefill --skip-server-warmup \
  --disaggregation-bootstrap-port 7200 \
  --kv-events-config '{"publisher": "zmq", "topic": "kv-events", "endpoint": "tcp://*:3000"}' \
  --context-length 4096 --mem-fraction-static 0.5 --page-size 64 --tensor-parallel-size 1
```

decode_0（node-02），环境变量里有 `INFERA_READINESS_PORT=7501`：

```text
python3 -m infera.engine.sglang \
  --discovery-backend etcd --etcd-endpoint http://127.0.0.1:2379 \
  --advertise-host 127.0.0.1 \
  --request-transport http --kv-event-transport zmq --kv-events off \
  --no-enable-kv-events \                                               ← R1 修复后新增
  --model-path /model --served-model-name Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 6100 --nccl-port 17501 --enable-metrics \
  --disaggregation-mode decode --skip-server-warmup \
  --context-length 4096 --mem-fraction-static 0.5 --page-size 64 --tensor-parallel-size 1
```

**修复前**，decode 这行既没有 `--kv-events-config` 也没有 `--no-enable-kv-events`。Infera 启动器的 `--enable-kv-events` 默认是开的，看到没有 `--kv-events-config`，就会自己找一个空闲端口，再补上 `--kv-events-config`：于是 decode 上多开了一个分配器不知道的 ZMQ 端口，广播的事件路由器也不读。

### 7.3 容器里发生了什么

```text
srun 步骤启动容器
  │
  ├─ bash_preamble：运行 install-infera.sh
  │    pip install --no-deps nats-py amd-infera==$INFERA_VERSION
  │    检查 infera.engine.sglang 能 import，检查自带的 infera/_bin/infera-router 存在且可执行（R6）
  │
  └─ python3 -m infera.engine.sglang <Infera 参数> <SGLang 参数>
       │
       ├─ 1. 拆参数：parse_known_args() 先拿走自己认识的（--discovery-backend 等），
       │            剩下的全部交给 SGLang 的 ServerArgs 解析
       │
       ├─ 2. 启动子进程：python -m sglang.launch_server <SGLang 参数>
       │            SGLang 在 6100 端口起 HTTP server；prefill 还在 3000 端口起 ZMQ PUB（KV events）
       │
       ├─ 3. 等就绪：反复 GET http://127.0.0.1:6100/health，直到返回 200
       │
       ├─ 4. 问 SGLang 实际用的 page size（/get_server_info），作为路由器的 KV 块大小
       │
       ├─ 5. 登记到 etcd（带 lease，worker 挂掉、lease 过期后记录自动消失）
       │       key   = /infera/workers/10.0.0.1:6100
       │       value = {
       │         "worker_id": "10.0.0.1:6100",
       │         "url": "http://10.0.0.1:6100",
       │         "model_name": "Qwen/Qwen3-0.6B",
       │         "engine": "sglang",
       │         "status": "active",
       │         "disagg_mode": "prefill",
       │         "disagg_meta": {"protocol": "sglang-bootstrap",
       │                         "params": {"bootstrap_addr": "10.0.0.1:7200"}},
       │         "kv_events_endpoint": "tcp://10.0.0.1:3000",
       │         "kv_block_size": 64,
       │         ...
       │       }
       │       decode 的记录类似，但 disagg_mode 是 "decode"，没有 bootstrap_addr，kv_events_endpoint 为 null
       │
       └─ 6. 在 INFERA_READINESS_PORT（7500）上打开一个 readiness 监听
               （srtctl 不读它，srtctl 看的是路由器；给它分配的端口只是为了不和别的进程抢）
```

因为第 3 步在第 5 步之前，**"在 etcd 里出现"就意味着"SGLang 已经能服务"**（意见清单 R9）。

---

## 8. 阶段 4：启动路由器，以及就绪判断

### 8.1 srtctl 里的接口调用流程

```text
FrontendStageMixin.start_frontend()
  └─ InferaFrontend.start_frontends(topology, runtime, config, backend, processes)
       └─ 对每个 frontend 节点：
            cmd = build_router_command(port=8000, runtime, config)
            start_process(command=cmd, env_to_set=recipe 环境 + frontend.env, bash_preamble=<install-infera.sh>)
```

拼出的命令：

```text
python3 -m infera.server --router-backend rust --host 0.0.0.0 --port 8000 \
  --discovery-backend etcd --etcd-endpoint http://127.0.0.1:2379 \      ← srtctl 管的参数（用户不能改）
  --request-transport http --kv-event-transport zmq \
  --router-policy kv-aware \                                            ← 来自 frontend.args
  --router-tokenizer-path /model                                        ← 默认用 worker 加载的同一份模型
```

### 8.2 路由器启动后做什么

```text
python3 -m infera.server
  ├─ 把 tokenizer 路径解析成本地目录（Hugging Face id 会先下载）
  └─ exec infera-router（Rust 二进制）
       ├─ 读 etcd：range /infera/workers/  → 得到当前所有 worker
       ├─ watch etcd：之后有新登记 / 记录消失都会收到通知
       ├─ 对每个带 kv_events_endpoint 的 worker：开一个 ZMQ SUB 连到 tcp://10.0.0.1:3000，订阅主题 "kv-events"
       │     （decode 没有 kv_events_endpoint，不订阅）
       └─ 在 8000 端口提供 OpenAI 兼容接口，以及 /v1/workers、/v1/models 等
```

### 8.3 srtctl 怎么判断"可以开始压测了"

```text
BenchmarkStageMixin.run_benchmark()
  └─ _wait_for_service_ready()
       └─ wait_for_model()                      每隔几秒轮询一次，直到就绪或超时
            └─ DynamicFrontend.probe_ready(host, 8000, n_prefill=1, n_decode=1)
                 └─ probe_json_health(GET http://<路由器>:8000/v1/workers)
                      └─ InferaFrontend.parse_health(响应, expected_prefill=1, expected_decode=1)
```

路由器返回的 `/v1/workers`（v0.3.3 的真实结构，每个字段都会输出）：

```json
{"workers": [
  {"worker_id": "10.0.0.1:6100", "url": "http://10.0.0.1:6100", "model_name": "Qwen/Qwen3-0.6B",
   "engine": "sglang", "status": "active", "disagg_mode": "prefill",
   "disagg_meta": {"protocol": "sglang-bootstrap", "params": {"bootstrap_addr": "10.0.0.1:7200"}},
   "kv_events_endpoint": "tcp://10.0.0.1:3000", "kv_block_size": 64,
   "dp_rank": null, "dp_size": null, "request_transport": "http"},
  {"worker_id": "10.0.0.2:6100", "url": "http://10.0.0.2:6100", "model_name": "Qwen/Qwen3-0.6B",
   "engine": "sglang", "status": "active", "disagg_mode": "decode", ...}
]}
```

`parse_health` 的计数规则：

| 条目 | 计入 |
|---|---|
| `status == "active"` 且 `disagg_mode == "prefill"` | prefill +1 |
| `status == "active"` 且 `disagg_mode` 是 `decode` 或 `mixed` | decode +1（aggregate worker 算 decode） |
| `status` 是别的值（如 `draining`） | 不计 |
| 缺 `status` 或缺 `disagg_mode` | 不计（R3：以后字段改名会让就绪失败，而不是误判为就绪） |

这里 prefill=1、decode=1，都达到期望，srtctl 开始跑 benchmark。

---

## 9. 一个请求的旅程（数据流）

### 9.1 总数据流图

```text
                         ┌───────────────────────── 控制面 ─────────────────────────┐
                         │                                                          │
          登记(put+lease) │                     ┌──────────┐      watch / range     │
   ┌─────────────────────┼────────────────────►│   etcd   │◄───────────────────┐   │
   │                     │                     │ :2379    │                    │   │
   │  ┌──────────────────┘                     └──────────┘                    │   │
   │  │                                                                         │   │
┌──┴──┴──────────────────┐                                         ┌───────────┴───┴───────┐
│ node-01                │      ① HTTP：prefill 腿（带 bootstrap_room） │ infera-router :8000    │
│ infera.engine.sglang   │◄────────────────────────────────────────│                        │
│  └ SGLang prefill :6100│                                         │ 每个 worker 的 KV 视图  │◄── benchmark
│    ZMQ PUB :3000 ──────┼──────── KV events（BlockStored 等）────►│ （哪些块哈希在谁那里）   │    HTTP 请求
│    bootstrap :7200     │                                         │                        │    /v1/chat/completions
└──────────┬─────────────┘                                         └───────────┬────────────┘
           │ ③ KV 传输（引擎之间直接传，                                          │
           │    decode 通过 bootstrap_addr 找到 prefill）                         │ ② HTTP：decode 腿（同一个 bootstrap_room），
           ▼                                                                    │    decode 的输出流式返回给客户端
┌────────────────────────┐                                                      │
│ node-02                │◄─────────────────────────────────────────────────────┘
│ infera.engine.sglang   │
│  └ SGLang decode :6100 │   （decode 不发布 KV events：--no-enable-kv-events）
└────────────────────────┘
```

### 9.2 一步一步看

用两个共享同一段 system prompt 的请求做例子。

**请求 A**（第一次见到这段 system prompt）：
1. benchmark 把请求发到路由器 `:8000`。
2. 路由器用 tokenizer 把 prompt 切成 token，按 64 个 token 一块算出一串块哈希。
3. 选 prefill：kv-aware 策略的打分是 `cost = −(重叠权重 × 命中块数) + 负载`。此时没有任何 worker 缓存了这些块，命中数都是 0，就按负载选，选中 prefill_0。
4. 选 decode：在同一个模型名（`Qwen/Qwen3-0.6B`）的 decode 池里选一个，这里只有 decode_0。
5. **双发**：路由器同时把请求发给 prefill_0 和 decode_0，两边的请求体里都带同一个 `bootstrap_room`，并告诉 decode prefill 的 `bootstrap_addr`。
6. prefill_0 算完 prompt，KV 通过 bootstrap 机制直接传给 decode_0；decode_0 逐 token 生成，结果流式返回给 benchmark。
7. 与此同时，prefill_0 的 SGLang 把"我缓存了这些块"作为 KV events 发到 ZMQ :3000；路由器收到后更新 prefill_0 的 KV 视图。

**请求 B**（开头是同一段 system prompt）：
1. 路由器算出块哈希，发现前面若干块在 prefill_0 的 KV 视图里都有。
2. prefill_0 的命中数高，`cost` 更低，于是选它（只有一个 prefill 时这一步看不出区别，多个 prefill 时就体现出来了）。
3. prefill_0 只需算没命中的那部分 token，前缀直接复用。

**如果没有 KV events**（所有 role 都没开 `kv_events`）：路由器的 KV 视图永远是空的，第 3 步永远命中 0，等于只按负载路由，而且不报错。这就是第 4 节要在加载时拦住的情况。

---

## 10. 端口和"谁分配"一览

| 监听方 | 端口（例子） | 谁分配 | 谁连它 |
|---|---|---|---|
| etcd | 2379 | srtctl（固定端口） | worker（登记）、路由器（watch） |
| SGLang HTTP server | 6100 | srtctl 分配器（每节点） | 路由器（转发请求）、Infera 启动器（查 /health）、metrics 抓取 |
| SGLang ZMQ PUB（KV events，仅 prefill） | 3000 | srtctl 分配器，经 `--kv-events-config` 传入 | 路由器（ZMQ SUB） |
| SGLang bootstrap（仅 prefill） | 7200 | srtctl 分配器 | decode（接 KV） |
| Infera readiness 监听 | 7500 / 7501 | srtctl 分配器（借用 system 端口），经环境变量 `INFERA_READINESS_PORT` 传入 | 没人连（只为不抢端口） |
| infera-router | 8000 | srtctl（frontend 端口） | benchmark、srtctl 就绪检查 |

修复前多出来的一行：decode 上 Infera 启动器自己挑的 ZMQ 端口，**不在这张表里**，分配器不知道它，可能和同节点的别的进程冲突。

---

## 11. 代码在哪里

| 文件 | 内容 |
|---|---|
| `src/srtctl/frontends/infera.py` | `InferaFrontend`：`validate`（4 条规则）、`implied_services`（etcd）、`worker_module` / `worker_launcher_args` / `worker_environment`（worker 怎么启动）、`build_router_command` / `start_frontends`（路由器）、`parse_health`（就绪计数）、各种端口回答 |
| `src/srtctl/frontends/base.py` | `Frontend` 基类上的三个新 hook，默认值让其他前端不受影响 |
| `src/srtctl/backends/base.py` | `Backend.get_kv_events_config_for_mode`，默认 `None`（SGLang、vLLM 各自实现） |
| `src/srtctl/backends/sglang.py` | `build_worker_command`：用 `worker_module` 和 `worker_launcher_args` 拼命令开头 |
| `src/srtctl/cli/mixins/worker_stage.py` | `_worker_environment_defaults`：对所有前端合并 `worker_environment` |
| `configs/install-infera.sh` | 容器里安装 Infera 并检查路由器二进制 |
| `examples/sglang/infera-disagg.yaml` + `tests/snapshots/launch/sglang__infera-disagg.txt` | 示例 recipe 和它在集群上实际会跑的命令 |
| `tests/test_infera_frontend.py` | 单测 |
| `docs/infera.md` | 用户文档 |

---

## 12. PR 原版和 review 修改后的差别

| 方面 | PR 原版（head `5bf86633`） | 修改后 |
|---|---|---|
| 换掉 worker 启动模块的接口 | 一个 hook `direct_worker_launcher()`，返回整段 argv（含 `env KEY=VAL python3 -m infera.engine.sglang ...`），backend 直接拼在前面 | 三个 hook：`worker_module()` 只给模块名，`worker_launcher_args()` 给参数，`worker_environment()` 给环境变量；`python3 -m` 仍由 backend 拼 |
| readiness 端口 | 命令里的 `env INFERA_READINESS_PORT=7500` 前缀 | worker 环境变量（`Env:` 日志和 snapshot 的 `env_to_set` 里能看到，recipe 可覆盖） |
| decode 的 KV events | 看 `process.kv_events_port is None`（永远不成立），decode 什么都不加，Infera 自己开端口 | 看 `backend.get_kv_events_config_for_mode(mode)`，decode 加 `--no-enable-kv-events` |
| 就绪计数 | 缺 `status` 当 active，缺 `disagg_mode` 当 decode | 缺字段不计数 |
| 加载时校验 | 只拦 srtctl 管的参数和 external etcd | 另外拦"kv-aware 但没人发 KV events"和"INFERA_VERSION 不一致" |
| 安装检查 | import Infera 私有函数 `_find_binary` | 直接检查 `infera/_bin/infera-router` |
| 文档 | — | 补了就绪语义、版本和参数透传、各 role 需要一致的参数、只支持 SGLang 的原因 |
