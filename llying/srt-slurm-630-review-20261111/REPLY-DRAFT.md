# srt-slurm#630 回复草稿（待你审阅，未发出）

- PR：[NVIDIA/srt-slurm#630](https://github.com/NVIDIA/srt-slurm/pull/630)，当前 head `afefb95`（已推送）。
- 发法：第一部分发成 PR 的总评论（`gh pr comment`）；第二部分逐条回到各自的 inline 讨论串里（`gh api repos/NVIDIA/srt-slurm/pulls/630/comments/<id>/replies`）。
- 每条下面的"中文要点"只给你看，不发。
- 需要你重点确认的地方见文末。

---

## 一、总评论

> Thanks @nv-kmcgill53 and Devin for the reviews. All points are addressed in afefb95, on top of a merge with current `main` (#661). Replies are in the threads; in short:
>
> - **KV events (Devin's blocker):** the decision now comes from `Backend.get_kv_events_config_for_mode()`, the same method the backend uses for `--kv-events-config`, so a role without `kv_events` gets `--no-enable-kv-events` and Infera's launcher no longer opens a port of its own.
> - **Worker hooks:** `direct_worker_launcher()` is replaced by `worker_module()`, `worker_launcher_args()` and `worker_environment()`. The backend builds `python3 -m <module>` itself, and `INFERA_READINESS_PORT` moves from an `env` prefix into the worker environment.
> - **Readiness:** an entry of `/v1/workers` without `status` or `disagg_mode` no longer counts; the tests pin v0.3.3's payload.
> - **Load-time checks:** kv-aware routing (also `infera.server`'s default) with no role publishing KV events, and different `INFERA_VERSION` values for the router and the workers, are rejected.
> - **Smaller items:** the install check no longer imports a private Infera helper; docs cover readiness semantics, the release pin, what the roles must agree on, and why SGLang only.
>
> Validation on afefb95:
>
> - **Checks and tests:** `ruff check`, `ruff format --check`, `ty check`, `srtctl schema-docs --check` and `mkdocs build --strict` pass, and the regenerated launch snapshot changes only `sglang__infera-disagg.txt`. The infera, frontend, design-rule, launch-snapshot and backend tests pass (222). The full suite has 5529 passed and 3 failed, in `test_dry_run.py` (2) and `test_schema_docs.py` (1): on our nodes rich wraps output at 80 columns and splits the asserted paths. The same 3 fail on the previous head (5bf86633), so they come from the environment, not this change.
> - **Hardware smoke (AMD MI355X, Slurm + Pyxis, 2 nodes):** the example recipe adapted to the cluster (`gpu_type: mi355x`, stock `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`, Mooncake over the ionic rails), Qwen3-0.6B, amd-infera 0.3.3 installed by `install-infera.sh`, srtctl at afefb95. Prefill and decode on separate nodes, both registered, and `/v1/workers` readiness passed.
>   - The decode registered with `kv_events_endpoint: null`, and Infera spawned its SGLang without `--kv-events-config`.
>   - `INFERA_READINESS_PORT` showed in the per-worker `Env:` lines, and the launchers opened readiness on the allocated 7500 and 7501.
>   - The router subscribed only to the prefill's allocated `tcp://<prefill>:3000`.
>   - Three chat requests returned correct answers.
> - **Production recipes:** our GLM-5.2 MXFP4 recipes (4+4, 1P1D, and 2P1D with two prefills on one node) load and render through the mock orchestrator with afefb95. Decode gets `--no-enable-kv-events`, and the colocated prefills get distinct readiness ports (7500 / 7501) and KV-events ports (3000 / 3001).
>
> @nv-kmcgill53, could you take another look when you have time? The CI and Copyright workflows for afefb95 are waiting for approval again (fork PR); could a maintainer approve them? Thanks!

**中文要点：** 概括改了什么，再列三层验证：检查和单测、真机冒烟、生产 GLM recipe 的渲染；最后请 nv-kmcgill53 再看一次，并请维护者批准 CI。完整 pytest 的 3 个失败如实说明是环境问题，并给出"旧 head 上同样失败"的证据。

---

## 二、逐条回复

### Devin

**[4518887987] `infera.py:154`（阻塞：decode 没有 `--no-enable-kv-events`）**

> Fixed in afefb95, as you suggested. `Backend` now has `get_kv_events_config_for_mode(mode)` with a `None` default (SGLang and vLLM already implement it), and the frontend appends `--no-enable-kv-events` when it returns `None` for the process's mode: the same call the backend uses to emit `--kv-events-config`, so the two flags can no longer disagree. The example's decode now carries `--no-enable-kv-events` in the snapshot. On hardware the decode registered with `kv_events_endpoint: null` and Infera spawned its SGLang without `--kv-events-config`, so it opens no publisher.

中文要点：按 Devin 的建议改；snapshot 和真机都验证了。

**[4518888000] `tests/test_infera_frontend.py:183`（阻塞：测试是人为构造的）**

> Replaced by `test_only_a_role_that_publishes_kv_events_gets_an_endpoint`, which builds both processes from `SGLangBackend.allocate_endpoints()` / `endpoints_to_processes()`, asserts that the decode process does have an allocated KV-events port, and checks that the decode command gets `--no-enable-kv-events` while the prefill gets `--kv-events-config`.

中文要点：改成用真实 topology 构造进程，并断言 decode 确实分到了端口（这正是原 bug 的陷阱）。

**[4518888008] `docs/infera.md:36`（nit：文档与实际不符）**

> True now, and the row also says why the flag is needed (without it the launcher publishes on a port of its own).

中文要点：修了 bug 后这句成立，顺便补了原因。

**[4518888018] `infera.py:60`（nit：etcd 地址应复用 `discovery_env`）**

> I kept `etcd_endpoint(runtime)` and the external-etcd rejection for now. Worker commands are built in `Backend.build_worker_command(process, endpoint_processes, runtime, ...)`, which has no `config`, so `discovery_env(config, runtime)` would mean passing the config into the worker hook as well. With external etcd rejected, the router and the workers resolve the same address. Happy to plumb the config through in this PR if you'd like external etcd supported.

中文要点：没改，说明原因（worker 侧拿不到 config），并表示愿意改。

**[4518888025] `configs/install-infera.sh:18`（nit：用了私有函数 `_find_binary`）**

> Switched to your suggestion: the script now checks that `infera/_bin/infera-router` is executable, by path, and fails with a message naming the release. It still imports `infera.engine.sglang` first, and the example keeps the 0.3.3 pin.

中文要点：按 Devin 的写法改了。

### nv-kmcgill53

**[4519138224] `infera.py:137`（`env` 前缀绕过环境变量层）**

> Done as you suggested. `Frontend.worker_environment(process, runtime)` (default `{}`) is merged in `_worker_environment_defaults` for every frontend, beneath the recipe's environment so it can be overridden, and it covers both the per-process and the per-endpoint launch paths. Infera returns `{INFERA_READINESS_PORT: str(process.sys_port)}`, and the `env` prefix is gone from the launcher argv. The port now shows in `env_to_set` in the snapshot and in the per-worker `Env:` line; on hardware the launchers opened readiness on 7500 and 7501.

中文要点：按建议加了 hook；两条启动路径都覆盖；可被 recipe 覆盖；真机验证。

**[4519138242] `backends/sglang.py:372`（hook 返回原始 argv）**

> Took your first option. `direct_worker_launcher()` is replaced by `worker_module(backend_type)` and `worker_launcher_args(backend, process, runtime)`, so the backend builds `python3 -m <module>` itself and the duplicated `if use_sglang` check is gone. `worker_launcher_args` takes the backend so the KV-events decision can ask it. I did not add a third `worker_launch` value, in line with your note, since only SGLang is wrapped. vLLM and TRT-LLM don't call the hooks; `required_backend` keeps infera on SGLang (it is checked for every active role at load), and `worker_module()` now raises a clear error for any other engine instead of a `KeyError`.

中文要点：采纳他的方案；说明为什么 `worker_launcher_args` 多一个 backend 参数；顺带回答了他概括意见里的 `KeyError`。

**[4519138231] `infera.py:158`（就绪等于"已注册"，是否等于"能服务"）**

> Confirmed against amd-infera 0.3.3: `SglangEngine.start()` spawns SGLang and awaits `_wait_ready()`, which polls `/health` until it answers 200, before it resolves the page size and builds the registration. So a worker appears in etcd only once SGLang serves. The Readiness row of docs/infera.md now says so.

中文要点：查了 Infera 源码确认，并写进文档。

**[4519138236] `infera.py:163`（缺字段时默认值太宽松）**

> Changed: only entries with `status == "active"` and a known `disagg_mode` count, and a missing field no longer counts. The tests use v0.3.3's full `/v1/workers` entry (the router serializes every field of its `Worker` struct) and add entries with each field removed, so a rename fails readiness instead of passing it.

中文要点：缺字段不计数，测试用 v0.3.3 的真实结构。

**[4519138254] `infera.py:155`（Infera/SGLang 契约分散在两个文件）**

> Taking the four points in turn:
>
> - **KV events decided on both sides:** both flags now come from `get_kv_events_config_for_mode` (see Devin's thread).
> - **Flag forwarding:** I didn't add a test in srtctl's CI. `infera.engine.sglang` takes its own flags with `parse_known_args` and hands the rest to SGLang's `ServerArgs`, so checking the forwarding needs SGLang installed. docs/infera.md now says that srtctl relies on the pinned release forwarding unknown flags unchanged, and that a new release should be validated on hardware before moving the pin.
> - **SGLang facts in the frontend:** Infera discovers workers dynamically, as Dynamo does, but its workers are direct: the launcher starts SGLang's own server on the allocated `http_port` and only registers it, and the router sends requests straight to that port. The port answers follow the worker, not the discovery. Dynamo answers `sys_port` because every `dynamo.<engine>` rank runs the Dynamo system server there, whatever the engine; an Infera worker has no such server. So the endpoint, metrics and profiling ports are the leader's `http_port`, which is the `--port` the SGLang backend passes. Every direct frontend answers this way for every engine (`StaticRouterFrontend`, `trtllm_serve`), without asking the backend. The metrics path does ask the backend: `InferaFrontend.worker_metrics_path` overrides `DynamicFrontend`'s Dynamo `/metrics` and returns `Frontend.worker_metrics_path`, which reads `prometheus_metrics_path`. I left the ports as they are, but I'm happy to factor out a shared direct-worker helper if you prefer.
> - **`WORKER_LAUNCHERS`:** commented and documented as SGLang only, with what a vLLM row would also need: the vLLM backend calling the hooks, and the port answers checked against its layout.

中文要点：四点逐一回答；参数透传不加测试（需要装 SGLang），改为文档写明；说明 Infera 发现 worker 是 dynamic、worker 本身是 direct，端口跟着 worker 走，所以和 direct 类 frontend 一样回答 `http_port`，metrics 路径已覆盖 Dynamo 默认值改为问 backend，没改但愿意抽公共函数；表只支持 SGLang 写清楚了。

**[4519138261] `examples/sglang/infera-disagg.yaml:35`（`kv-aware` 和 `kv_events` 没有关联）**

> Infera degrades silently: with no events the router's KV view stays empty and it routes on load alone, logging a warning only after a streak of zero-hit picks. Since `kv-aware` is also `infera.server`'s default policy, `InferaFrontend.validate()` now rejects an effective kv-aware policy (set or defaulted) when no active role publishes KV events, and points to `kv_events: true` on the prefill (or agg) role or another `router-policy`. One exception: `decode-input-demand` (decode balancing by in-flight input tokens) runs only under kv-aware and needs no events, so when it is `on` or `shadow` (in `frontend.args`, or `INFERA_DECODE_INPUT_DEMAND` in the router's env, which the router reads when the arg is unset), loading logs a warning instead.

中文要点：回答"会静默退化"，并加了加载时校验（包括没写 policy 的默认情况）；例外是开了 `decode-input-demand` 时只打 warning，因为它只能在 kv-aware 下用，而且不需要 KV events。

**[4519138272] `examples/sglang/infera-disagg.yaml:49`（`page-size`、`served-model-name` 重复写）**

> Added a "What the roles must agree on" section to docs/infera.md:
>
> - **Served model name:** the router pools workers by the model name they register and routes a request to the pools of the model it names. srtctl passes every role one `--served-model-name` (`SGLangBackend.get_served_model_name()`, the first of prefill, agg, decode that sets it), so prefill and decode always pair.
> - **Page size:** each worker registers the page size SGLang resolved, and the router builds that worker's KV view at that size, so nothing on the router side has to match. Prefill and decode still need one `page-size` for SGLang's own KV transfer.

中文要点：两者都说明了；served-model-name 由 srtctl 统一传，不可能不一致。

**[4519138277] `examples/sglang/infera-disagg.yaml:29`（`INFERA_VERSION` 只靠一个全局变量）**

> `InferaFrontend.validate()` now rejects a recipe whose `environment`, `frontend.env` and `roles.<role>.env` set different `INFERA_VERSION` values, naming each source in the error.

中文要点：加了校验。

### nv-kmcgill53 概括意见里的另外两点（回到他的 review 下面，或并进总评论）

> - **Snapshot addresses:** right, the `127.0.0.1` etcd and advertise addresses come from the mock orchestrator. Real addresses come from `get_hostname_ip`; the hardware run above had etcd and the workers on the nodes' own IPs.
> - **`get_hostname_ip` per worker command:** it is the cached `get_hostname_ip` (`@functools.cache`), and #651 is in this branch.

中文要点：两条都是确认性回答，不需要改代码。

---

## 需要你确认的地方

1. **D4（etcd 地址）和 K5 第三点（端口逻辑）没有改代码**，回复里说明了原因，并表示愿意改。你如果希望这一轮就改掉，我再改。
2. **硬件信息写到什么程度**：总评论里提到了"AMD MI355X、2 nodes、原样镜像名、Qwen3-0.6B"，没有写集群名和节点名；回复里出现了 `tcp://<prefill>:3000` 这样的占位写法，没有写具体 IP。
3. **完整 pytest 的 3 个失败**：如实写了，并给出"旧 head 上同样失败"的证据。你也可以选择不提。
4. **at 谁**：总评论只 at 了 nv-kmcgill53；批准 CI 写的是"a maintainer"，没有点名。你上次 at 过 ishandhanani 和 alec-flowers，要不要再 at 他们请求批准 CI，你来定。
5. Devin 关于 readiness 端口的"开放问题"他已经同意，不需要单独回复。
