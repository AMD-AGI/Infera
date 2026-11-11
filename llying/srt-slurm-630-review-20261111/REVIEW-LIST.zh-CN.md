# srt-slurm#630 意见合并清单（按优先级）

把我的 review 和两位 reviewer（Devin、nv-kmcgill53）的意见合并同类项后排成一张清单。每一项写明：来源、问题是什么、为什么是问题（原理）、怎么改、当前状态。

- PR：[NVIDIA/srt-slurm#630](https://github.com/NVIDIA/srt-slurm/pull/630)，审阅基于 head `5bf86633`，对照 Infera `v0.3.3` 源码
- 代码修改在本地 worktree `~/code/bench_agentx/srt-slurm-630-review`（分支 `llying/infera-frontend-review`，已合并 main #661），提交 `afefb95`，**已推送到 PR 分支（11-11 14:27 后），未回复 reviewer**
- 原理和执行过程的详细讲解见同目录 `EXPLAIN.zh-CN.md`
- 来源缩写：D = Devin，K = nv-kmcgill53，我 = 我的 review
- 状态：待改 / 已改 / 不改（只回复） / 待你决定

## 总览

| 编号 | 优先级 | 一句话 | 来源 | 处理方式 | 状态 |
|---|---|---|---|---|---|
| R1 | P0 | decode worker 打开了未分配的 KV-events 端口 | D1、D2、D3、K5a、我 | 改代码、测试、snapshot | 已改 |
| R2 | P1 | worker 启动入口的接口形状，以及 readiness 端口的 `env` 前缀 | K1、K2、我 | 改代码（新增 3 个 hook） | 已改 |
| R3 | P1 | 就绪判断太宽松 | K4、我 | 改代码，加 fixture 测试 | 已改 |
| R4 | P1 | `kv-aware` 拿不到 KV events 时悄悄退化 | K6、我 | 加 `validate()` 校验 | 已改 |
| R5 | P2 | 路由器和 worker 的 Infera 版本可能不一致 | K8 | 加 `validate()` 校验 | 已改 |
| R6 | P2 | 安装脚本 import 了私有函数 | D5、我 | 改成检查文件路径 | 已改 |
| R7 | P2 | `WORKER_LAUNCHERS` 看似通用，且可能抛 `KeyError` | K5d、K | 随 R2 显式报错，补注释和文档 | 已改 |
| R8 | P2 | etcd 地址重复构造 | D4、我 | 不改，回复说明 | 不改 |
| R9 | P3 | 就绪等于"已注册"，是否等于"能服务" | K3 | 文档写明，回复 | 已改（文档） |
| R10 | P3 | Infera 参数透传方式变了没人发现 | K5b | 文档写明，回复 | 已改（文档） |
| R11 | P3 | SGLang 细节写死在 InferaFrontend 里 | K5c | 回复说明 | 不改 |
| R12 | P3 | `page-size`、`served-model-name` 在各 role 重复写 | K7 | 文档加说明 | 已改（文档） |
| R13 | P3 | snapshot 里的 `127.0.0.1` 来自 mock | K | 回复 | 不改 |
| R14 | P3 | `get_hostname_ip` 每个 worker 调一次 | K | 回复（已有缓存） | 不改 |
| R15 | 验证 | 硬件验证早于 readiness 端口的改动 | 我 | 改完后在 AUS 跑一次冒烟 | 已完成，通过 |
| R16 | 验证 | 分支落后 main 1 个提交 | 我 | 合并 main，重新生成 snapshot | 已完成 |

---

## P0：必须修

### R1　decode worker 打开了未分配的 KV-events 端口（真实 bug）

- **来源**：D1、D2（Devin 标为阻塞）、D3（文档那句不成立）、K5a（KV events 由两个文件分别决定）、我的第 1 条。
- **问题**：`infera.py` 用 `process.kv_events_port is None` 判断要不要加 `--no-enable-kv-events`。snapshot 里 decode worker 的命令既没有 `--kv-events-config`，也没有 `--no-enable-kv-events`。
- **原理**：
  1. srtctl 的 topology 给**每个**进程都分配一个 KV-events 端口块（`core/topology.py`，`kv_events_port=allocator.next(...)` 没有条件），所以 `kv_events_port` 永远不是 `None`，上面的判断永远不成立。
  2. SGLang backend 只在 role 开了 `kv_events` 时才加 `--kv-events-config`（`get_kv_events_config_for_mode(mode)` 不为 `None`）。decode 没开，所以 backend 不加。
  3. Infera 0.3.3 的 `--enable-kv-events` 默认是开的；argv 里没有显式的 `--kv-events-config` 时，`worker.py` 会调用 `free_tcp_port_block(dp_size)` 自己找空闲端口，再补上 `--kv-events-config`。
  4. 结果：decode 上开了一个分配器不知道的 ZMQ 端口（违反"所有监听端口来自分配器"的规则），还发布了路由器根本不读的事件。
  5. 单测能通过，是因为测试里用 `_process(..., kv_events_port=None)` 人为构造了一个真实 topology 里不存在的进程。
- **怎么改**：
  - 在 `Backend` 基类上加 `get_kv_events_config_for_mode(mode)`，默认返回 `None`（SGLang、vLLM 已经各自实现）。
  - frontend 改成看 `backend.get_kv_events_config_for_mode(process.endpoint_mode) is None` 来决定是否加 `--no-enable-kv-events`。这样 backend 加 `--kv-events-config`、frontend 加 `--no-enable-kv-events` 都以同一个方法为准，K5a 说的"两处分别决定"也就解决了。
  - 单测改成用 backend 真实分配出的进程来拼命令，并断言 decode 带了 `--no-enable-kv-events`、prefill 带了 `--kv-events-config`。
  - 重新生成 snapshot，decode 那一行会出现 `--no-enable-kv-events`。
  - `docs/infera.md` 里"A role without `kv_events` registers with `--no-enable-kv-events`"这句因此成立，不用改。

---

## P1：应该改（主要是维护者的设计要求）

### R2　worker 启动入口的接口形状，以及 readiness 端口的 `env` 前缀

- **来源**：K1（`env` 前缀绕过环境变量层）、K2（hook 返回原始 argv）、我的第 4 条。
- **问题**：
  - 现在新增的 hook `Frontend.direct_worker_launcher()` 返回一整段 argv：`env INFERA_READINESS_PORT=7500 python3 -m infera.engine.sglang --discovery-backend etcd ...`，SGLang backend 直接把它拼在命令开头。
  - readiness 端口是通过命令里的 `env KEY=VAL` 前缀传进去的。
- **原理**：
  - **K1**：srtctl 的 worker 环境变量有专门的组装流程（`_worker_environment_defaults`、`Backend.get_process_environment`、recipe 里的 `environment` 等），最后会出现在每个 worker 的 `Env:` 日志和 snapshot 的 `env_to_set` 里，而且用户可以在 recipe 里覆盖。藏在命令里的 `env` 前缀绕开了这一切：日志里看不到，也没法覆盖。
  - **K2**：hook 返回任意 argv，backend 就不再掌控 `python3 -m <module>`，frontend 理论上可以往命令开头塞任何东西（`env` 前缀就是一例）。另外 `... if use_sglang else None` 又判断了一次 `worker_launch`，和 frontend 自己声明的信息重复。
- **怎么改**（按 K 给的方案，三个 hook 都有默认值，别的 frontend 不受影响）：
  - 删掉 `direct_worker_launcher`，在 `Frontend` 基类上新增：
    - `worker_module(backend_type) -> str | None`：用哪个 Python 模块代替引擎自己的 server 模块，默认 `None`；
    - `worker_launcher_args(backend, process, runtime) -> list[str]`：这个模块自己的参数，放在引擎参数前面，默认 `[]`；
    - `worker_environment(process, runtime) -> dict[str, str]`：这个 frontend 给每个 worker 进程加的环境变量，默认 `{}`。
  - SGLang backend：`module = frontend.worker_module(self.type) or python_module`，然后拼 `["python3", "-m", module, *frontend.worker_launcher_args(...)]`，`python3 -m` 仍由 backend 拼。去掉多余的 `if use_sglang`。
  - `worker_stage._worker_environment_defaults` 对**所有** frontend 合并 `worker_environment()`。它在用户环境变量之前合并，所以用户可以覆盖；两条 worker 启动路径（逐进程、按 endpoint）都会经过它。
  - Infera：`worker_module` 查 `WORKER_LAUNCHERS`；`worker_launcher_args` 返回 `--discovery-backend etcd ...`；`worker_environment` 返回 `{"INFERA_READINESS_PORT": str(process.sys_port)}`。snapshot 里这个变量会从命令里挪到 `env_to_set`。
  - 同步更新 `frontends/AGENTS.md`、`dynamic_frontend.py` 的说明和 `tests/test_frontends.py` 的契约表。
  - 不新增第三种 `worker_launch`：K 自己也说只有将来要包装第二个 backend 时才值得，因为会牵动 vLLM 和 TRT-LLM。

### R3　就绪判断太宽松

- **来源**：K4、我的第 2 条。
- **问题**：`parse_health` 里，worker 缺 `status` 字段时按 `active` 算，缺 `disagg_mode` 时按 `mixed`（也就是 decode）算。
- **原理**：srtctl 靠路由器 `/v1/workers` 的统计判断"可以开始压测了"。如果将来 Infera 改了字段名，这些默认值会把不认识的条目都算成就绪，job 会在 worker 还没真正注册时就开始 benchmark，而且不会报任何错。
- **怎么改**：
  - 只统计 `status == "active"` 且 `disagg_mode` 是 `prefill`、`decode`、`mixed` 之一的条目；缺字段就不计数。
  - 用 Infera v0.3.3 `/v1/workers` 真实的响应结构做测试 fixture。将来字段名变了，测试会失败，而不是悄悄误判为就绪。

### R4　`kv-aware` 拿不到 KV events 时悄悄退化

- **来源**：K6、我的第 3 条。
- **问题**：示例用 `router-policy: kv-aware`，但 srtctl 没有检查是否真的有 role 开了 `kv_events`。
- **原理**：
  - srtctl 用 `python3 -m infera.server` 启动路由器，这个启动器的默认策略就是 `kv-aware`（`infera/server/args.py`），所以即使 recipe 一个字都不写，用的也是它。
  - 所有 role 都没开 `kv_events` 时，路由器收不到任何缓存信息，但它不会报错，只是退化成按负载路由；直到连续多次未命中缓存，才往路由器日志里打一条 warning（`rust/router/src/policy.rs`："routing has degenerated to load-only"）。
  - 用户以为在跑 KV-aware 路由，实际不是，benchmark 结果会被误读。
- **怎么改**：
  - 在 `InferaFrontend.validate()` 里检查：生效策略是 `frontend.args` 里的 `router-policy`，没写就按 `kv-aware`。如果是 `kv-aware`，而所有 active role 都没开 `kv_events`，就报错，提示"给 prefill（或 aggregate）role 加 `kv_events: true`，或者换一个不读 KV events 的策略"。
  - 加对应的单测。

---

## P2：小问题，顺手改

### R5　路由器和 worker 的 Infera 版本可能不一致

- **来源**：K8。
- **问题**：路由器和 worker 用同一个版本，只是因为示例里两边都读同一个全局变量 `INFERA_VERSION`。
- **原理**：安装脚本在每个容器里按 `INFERA_VERSION` 安装。如果有人在 `frontend.env` 或 `roles.<role>.env` 里单独改了这个值，路由器和 worker 就会装上不同版本，etcd 里的注册格式或 KV events 格式可能对不上，而且没有任何检查。
- **怎么改**：在 `validate()` 里收集全局 `environment`、`frontend.env` 和各 role `env` 里的 `INFERA_VERSION`，出现两个以上不同的值就报错；加单测。

### R6　安装脚本 import 了私有函数

- **来源**：D5、我的第 5 条。
- **问题**：`install-infera.sh` 用 `from infera.server.launch_rust import _find_binary` 检查路由器二进制是否存在。
- **原理**：`_find_binary` 是 Infera 的私有函数，以后的版本改个名字，安装检查就会失败。
- **怎么改**：按 Devin 的写法，直接检查 wheel 里自带的 `infera/_bin/infera-router` 是否存在且可执行。这个路径同样是 Infera 的内部目录结构，所以示例里继续 pin 版本。

### R7　`WORKER_LAUNCHERS` 看似通用，且可能抛 `KeyError`

- **来源**：K5d、K 的概括意见。
- **问题**：`WORKER_LAUNCHERS[backend.type]` 按引擎名查表，看起来像支持多种引擎，实际 `required_backend = "sglang"` 挡住了其他引擎。
- **原理**：加载配置时 `SrtConfig._validate_frontend` 会检查所有 active role 都是 sglang，所以这个 `KeyError` 在正常流程里触发不到；但以后有人加一行 vLLM 时，还需要改端口和 metrics 方面的假设（见 R11），光加一行表项并不够。
- **怎么改**：随 R2，`worker_module()` 对表里没有的引擎抛出明确的报错；在注释和 `docs/infera.md` 写明"目前只支持 sglang"，以及支持 vLLM 还需要改哪些地方。

### R8　etcd 地址重复构造（不改）

- **来源**：D4、我的第 5 条。
- **问题**：`etcd_endpoint(runtime)` 自己拼了一遍 etcd 地址，而 `services/implicit.py` 里已有 `discovery_env(config, runtime)["ETCD_ENDPOINTS"]`，它还能处理 external etcd。
- **原理**：worker 命令是在 `Backend.build_worker_command(process, endpoint_processes, runtime, ...)` 里拼的，这里拿不到整份 config，而 `discovery_env` 需要 config。要改就得给 hook 多传一个 config 参数。
- **怎么改**：不改。保留 `etcd_endpoint(runtime)` 和"拒绝 external etcd"的校验，两处拼出的地址一定相同。Devin 也说保留拒绝是可以的。回复里说明原因。

---

## P3：只需写进文档或回复

### R9　就绪等于"已注册"，是否等于"能服务"

- **来源**：K3。
- **结论**：等于。Infera 的 `start()`（`infera/engine/sglang/worker.py`）先启动 SGLang，再 `_wait_ready()` 轮询 SGLang 的 `/health`，返回 200 之后才解析 page size、组装注册信息。所以 worker 出现在 etcd 里时，SGLang 已经在服务。
- **怎么改**：在 `docs/infera.md` 的 Readiness 那一行写明，回复确认。

### R10　Infera 参数透传方式变了没人发现

- **来源**：K5b。
- **原理**：`infera.engine.sglang` 用 `parse_known_args` 先拿走自己的参数，剩下的交给 SGLang 的 `ServerArgs` 解析。如果将来 Infera 新增了一个和 SGLang 同名的参数，就会悄悄吞掉它。
- **为什么不加测试**：要在 srtctl 的 CI 里测，就得安装 amd-infera，而它解析参数时要 import SGLang 的 `ServerArgs`；srtctl CI 里没有 SGLang，装它太重。
- **怎么改**：在 `docs/infera.md` 写明"依赖所 pin 的 Infera 版本的参数透传行为，升级版本时需要重新做硬件验证"，回复说明。

### R11　SGLang 细节写死在 InferaFrontend 里（不改）

- **来源**：K5c。
- **结论**（11-11 修正，详见 `QA-20261111.zh-CN.md` 第 10 题）：Infera 发现 worker 的方式是 dynamic（和 Dynamo 一样，靠 etcd 注册），但 worker 本身是 direct（SGLang 自己的 server 监听 `http_port`），端口回答取决于后者。metrics 路径专门覆盖了 `DynamicFrontend` 的 Dynamo 默认值（`/metrics`），改为问 backend 的 `prometheus_metrics_path`；endpoint、metrics、profiling 端口回答 leader 的 `http_port`，和 static router 对 direct worker 的回答相同（Infera 少了 gRPC 分支）。之前说"完全一样"不准确：ready 端口 static router 回答 `sys_port`，Infera 回答 `http_port`，但 SGLang 加 Infera 的流程用不到它。另外发现 Infera 没拦 `grpc-mode`，可选择补一条校验。
- **怎么改**：回复说明，不改代码。

### R12　`page-size`、`served-model-name` 在各 role 重复写

- **来源**：K7。
- **结论**（核实后比最初的判断更具体）：
  - **served-model-name**：路由器按"模型名 + 角色"给 worker 分池（`pool.rs` 的 `list_active(model, mode)`），请求按它指定的模型名去找 prefill 池和 decode 池，所以 prefill 和 decode 的名字必须一致才能配对。但 srtctl 本来就只传一个名字：`SGLangBackend.get_served_model_name()` 依次查 prefill、agg、decode，取第一个设置的值，`build_worker_command` 用它给每个 role 传 `--served-model-name`。所以不可能不一致，不需要加校验。
  - **page-size**：每个 worker 注册时上报 SGLang 实际使用的 page size，路由器按这个大小给该 worker 建 KV 视图，所以不需要和路由器的配置对齐；prefill 和 decode 的 page size 本来就必须一致，这是 SGLang PD 传输的要求，与 Infera 无关。
- **怎么改**：在 `docs/infera.md` 新增一节"What the roles must agree on"写明以上两点；回复里说明 srtctl 只传一个 served model name。

### R13　snapshot 里的 `127.0.0.1`（不改）

- **来源**：K。
- **结论**：说得对，这些地址来自 mock，不代表真实集群里的地址。真实地址由 `get_hostname_ip` 解析，PR 描述里的 AUS 硬件验证覆盖了这一点。回复即可。

### R14　`get_hostname_ip` 每个 worker 调一次（不改）

- **来源**：K。
- **结论**：`core/slurm.py` 里的 `get_hostname_ip` 带 `@functools.cache`，每个节点只解析一次；#651 已经在这个分支里（分支的 merge-base 就是 #651 的提交）。回复确认即可。

---

## 验证

### R15　硬件验证早于 readiness 端口的改动（待你决定）

- PR 描述里的 AUS 硬件验证是在 readiness 端口那次改动（`028898b1`）之前做的；这次 R1、R2 又改了 worker 的启动命令和环境变量。
- 建议改完后在 AUS 上用 InferenceX 的启动方式跑一次 1P1D 冒烟（起栈、注册、`/v1/workers` 就绪、GSM8K 少量题），确认：decode 不再自己开端口，`INFERA_READINESS_PORT` 出现在 worker 的 `Env:` 日志里。
- 这会占用 GPU 节点，需要你同意后再跑。

### R16　合并 main，重新生成 snapshot

- 分支落后 main 1 个提交（#661，"place a colocated decode on spread prefill nodes"），而示例里正好是 colocated decode，snapshot 可能受影响。先合并 main，再重新生成 snapshot。

---

## 测试结果（11-11 14:13–14:20 UTC）

在 holder 35609 的节点 n02-29 上用 `srun --overlap` 跑（holder 当时空闲、队列为空），Python 3.10 与 CI 一致，脚本 `/apps/liyingli/jobs/srt630_dev_tests.sh`，日志 `/apps/liyingli/jobs/llying-srt630-dev-overlap-1111T1413.out`。

| 检查 | 结果 |
|---|---|
| 重新生成 launch snapshot | 只有 `sglang__infera-disagg.txt` 变化：decode 多了 `--no-enable-kv-events`；`INFERA_READINESS_PORT` 从命令里的 `env` 前缀挪进 `env_to_set`。其他示例的 snapshot 都没变 |
| `ruff check`、`ruff format --check`、`srtctl schema-docs --check`、`ty check`、`mkdocs build --strict`、`bash -n install-infera.sh` | 全部通过 |
| 相关单测（Infera 前端、前端契约、设计规则、launch snapshot、backend 接口） | 222 passed |
| 完整 pytest | 5529 passed，3 failed。失败的是 `test_dry_run.py` 的两个用例和 `test_schema_docs.py::test_cli_check_fails_on_a_stale_file`：srtctl 的 rich 输出在节点上按 80 列折行，把长路径和 `cli-reference.md` 拆成两行，断言找不到完整字符串。**在没有本次改动的原始 PR head（`5bf86633`）上这 3 个同样失败**，与本次修改无关；GitHub 上这个 PR 的 CI 测试此前是通过的 |

---

## 真机冒烟（R15，11-11 14:41–14:46 UTC，通过）

- 方式：仓库示例 `examples/sglang/infera-disagg.yaml` 按 AUS 改写（`gpu_type: mi355x`、原样镜像 `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`、两个 role 加 Mooncake 走 8 条 ionic 的环境变量和参数），模型 `/apps/data/models/Qwen3-0.6B`，安装脚本用 PR 自己的 `install-infera.sh`（amd-infera 0.3.3）。用 PR head `afefb95` 的 srtctl 渲染，放进 holder 35609 的队列运行（holder 当时空闲）。
- 拓扑：prefill 在 n02-29，decode 在 n05-29（跨机 1P1D），etcd 和路由器在 n02-29。
- 文件：recipe 和渲染脚本在 `/apps/liyingli/jobs/srt630-smoke/`；输出在 `~/code/bench_agentx/srt-slurm-630-review/outputs/holder-35609-70a-srt630-smoke/`。

| 检查 | 结果 |
|---|---|
| 起栈和就绪 | etcd、两个 worker、路由器都起来；srtctl 轮询 `/v1/workers` 在 14:45:51 判定就绪；sweep 以 "Sweep completed successfully" 结束 |
| `/v1/workers` | prefill `active`，`kv_events_endpoint = tcp://10.235.192.61:3000`；decode `active`，`kv_events_endpoint = null` |
| decode 的 KV events（R1） | worker 命令带 `--no-enable-kv-events`；Infera 实际拉起 SGLang 的命令里 decode 没有 `--kv-events-config`，prefill 有（分配的 3000 端口）。decode 不再自己开端口 |
| readiness 端口（R2） | worker 的 `Env:` 日志里有 `INFERA_READINESS_PORT=7500 / 7501`，命令里不再有 `env` 前缀；Infera 日志 "readiness port open on 7500 / 7501" |
| 路由器订阅 | 只订阅 prefill 的 `tcp://10.235.192.61:3000` |
| 请求 | 3 个共享长 system prompt 的请求全部答对（42、72、63） |
| 无关的报错 | GPU 指标采集容器（`dcgm-exporter` 服务，用的是 AMD device-metrics-exporter）两台节点上都以退出码 127 结束，只影响监控指标，与本 PR 无关 |

---

## 真实 GLM-5.2 recipe 的渲染检查（11-11 15:10 UTC，不占 GPU，通过）

- 方式：取 InferenceX 启动器最近一次为三种形态渲染出的 GLM-5.2 配置（4+4 C80 来自 11-02，1P1D C88、2P1D C176 来自 10-30），用 PR head `afefb95` 的 srtctl 加载，再用生成 launch snapshot 的 `render_launch_plan()`（mock 编排器）生成全部 srun 调用。在 holder 35609 的 n02-29 上用 `srun --overlap` 跑，约 1 分钟。
- 文件：`/apps/liyingli/jobs/srt630-smoke/glm/`（三份配置、`render_plans.py`、`run.sh`、`summarize.py`，以及生成的 `*.plan.txt`）。

| 检查 | 结果 |
|---|---|
| 加载和校验（含新加的 `kv-aware` 与 `INFERA_VERSION` 两条） | 三份都通过：recipe 用 `router-policy: kv-aware`，prefill 开了 `kv_events`；不设 `INFERA_VERSION`（用 InferenceX 自己的 `infera-sglang.sh` 安装） |
| mock 编排 | 三份都以退出码 0 结束 |
| prefill | readiness 端口在 `env_to_set` 里（7500，2P1D 的第二个 prefill 是 7501）；`--kv-events-config` 用分配的端口（3000，2P1D 第二个是 3001），同机两个 prefill 不冲突 |
| decode | readiness 端口在 `env_to_set` 里；带 `--no-enable-kv-events`，没有 `--kv-events-config`；GLM 自己的参数原样保留（`--dp-size 8`、`--enable-dp-attention`、`--tp-size 8`、`--speculative-algorithm EAGLE`、Mooncake 与 8 条 ionic） |
| 命令里的 `env` 前缀 | 三种形态都没有了 |
| 路由器 | `--router-policy kv-aware` 等参数不变 |

对 GLM 唯一的行为变化：decode 不再发布 KV events，路由器也不再订阅 decode。GLM recipe 关掉了 decode 的 radix cache，decode 的事件本来就没有可用信息（e2e 时路由器对 decode 报过 "kv event feed is not being applied"），所以路由结果不变，这条 warning 会消失。

限制：用的是 InferenceX 10-30 / 11-02 渲染的配置，InferenceX recipe 此后如有改动未覆盖；完整的 GLM 上机测试留到 #630 合入、InferenceX 更新 srt-slurm 版本时，在 InferenceX 的 PR CI 里跑。
