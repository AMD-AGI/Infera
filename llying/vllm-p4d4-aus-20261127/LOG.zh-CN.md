# 执行记录（新的在上）

## 11-27

- 18:43 **a49 在 aus 上的 InferenceX eval**：全量 GSM8K（1,319 题，5-shot）在并发 152、真实接受下，strict-match / flexible-extract 都是 **0.9689**（阈值 0.90，通过）。AL 门禁没跑成：`kit/eval.sh` 传给它的是服务根地址，少了 `/metrics`，已修复；之后 C1 / D 的 eval 会跑 AL 门禁。结果在 `results/a49-gsm8k-c152/`。
- 18:35 单元测试的结论：
  - 原版 nightly：三个回归文件 136 个全过。
  - `test_rocm_paged_mxfp4_indexer.py`：在 C0 和 C1 上单独跑都是 55 个全过；之前的崩溃只在多个文件放在同一个进程里跑时出现。
  - `test_rocm_aiter_mla_sparse_metadata_sync.py` 的 46 个失败：C0 测试 checkout 只打了 `vllm/` 下的改动，测试文件还是 main 的原版，和 #11 改过的代码对不上。已重建两份 checkout（测试改动一起打上），排了重测。
  - #15 新测试失败 1 个：测试伪造的 quant config 缺字段（测试 bug），已修复并推送。
- 18:20 问题：`kit/engine/up.sh` 里把引擎命令写成了多行，`bash -c` 只执行了第一行（prefill 成了 PP1、没有 MTP，加载时 OOM）。改为传一行，C1 fast 重排。
- 18:12 节点检查结论（两个节点相同，amdgpu 6.19.14，有 `ib_peer_mem`），150 GiB 显存注册：
  | 方式 | 注册后可分配显存 | 实际空闲 |
  |---|---|---|
  | hipMalloc + dma-buf | 0 | 137.7 GiB（双倍计数） |
  | hipMalloc + peer memory | 0 | 137.7 GiB（双倍计数） |
  | GEM + dma-buf | 137 GiB | 0.56 GiB（与实际一致） |

  6.19 上 dma-buf 和 peer memory 都有双倍计数，所以用 GEM（与用户的判断一致）；镜像自带的 libionic 能注册，不挂主机的。
- 18:10 问题：
  - C1 fast 的 prefill 启动 OOM，原因是和我重跑的节点检查（GPU 0–2 上各注册 150 GiB）撞在了一起；以后 GPU 测试只在节点空闲时排。
  - 节点检查第二次重跑时报 "Stale file handle"：我提交时 `git pull --rebase` 替换了正在被 bash 读的脚本。改成每个任务先把脚本树复制到节点本地 `/tmp`（`holder/snap.sh`），从副本运行；回放目录 R 改为每个节点一份。
  - C1 单元测试：新增测试 76 过 1 失败；回归文件 `test_rocm_aiter_mla_sparse_metadata_sync.py` 46 个失败，正在用原版 nightly 做对照，确认是不是补丁引入的。
- 17:52 37014 开始，节点 smci355-ccs-aus-n04-25 / n04-29，两台都是 amdgpu 6.19.14，都有 `ib_peer_mem`；GPU i 和 ionic_i 在同一个 PCIe 交换芯片下（与 crusoe 相同），8 块网卡都是 ACTIVE。
- 18:05 nightly 镜像构建的两个问题：(1) Dockerfile 里补丁路径少了 `vllm/` 一层；(2) 补丁都打上后，导入检查失败：#17 在模块顶层用了 `@gluon.jit`，构建时没有 GPU，`vllm.triton_utils` 把 gluon 换成没有 jit 的占位对象，而 `_aiter_ops` 导入时会连带导入它。已在 llying-001/vllm#17 上修复（gluon kernel 放进 `if HAS_TRITON:`，head 14798047），补丁文件已刷新。被连带失败的正式轮先挪到 `holder/parked/`，C1 跑通后再放回。前面各层（Python 3.14 上安装 Infera、重新构建 Mooncake）都成功了。
- 17:58 我的失误：重跑节点检查前把检查结果（含完成标记）挪走了，两个节点构建完 v0.34.0 镜像（有缓存，186 s）后领到的 a49 任务因"no probe"直接退出。已重新排进节点专属队列（a49 fast → n04-25，a49 正式 → n04-29），各节点当前任务（nightly C0 / C1 的构建与 fast）结束后先跑。
- 17:55 问题：节点检查里三个显存注册测试都没跑起来（`docker run --group-add render` 用了组名，镜像里没有 render 组），GEM=1 是失败后的默认值。改成传主机的数字 gid（`test_C1.sh` 也一样），用 `srun --overlap` 在两个节点上重跑节点检查，和正在进行的镜像构建并行。
- 15:34 静态准备进展：
  - 节点检查（`probe/probe_node.sh`）：在 nightly 原版镜像里跑 150 GiB 的 dma-buf / peer memory / GEM 三个用例，镜像自带和主机的 `libionic` 各一轮，按结果自动定 GEM 和 libionic。
  - 阶段 B 回放（`replay/`）：crusoe 脚本原样组装，路径替换用 sed，逻辑改动在 `aus.patch`（9 处）。
  - InferenceX main 版的 eval / AgentX（`kit/`）。
  - 队列：每个节点各构建一次 v0.34.0 镜像，然后由空闲节点领走阶段 B 的 fast 和 full。
  - 核对了在线 FP8 的范围：模型 78 个主干层，NextN 草稿层编号 78，FP8 只作用于 0–77，符合规则。
  - nightly 补丁集、路由侧选 rank 由子 agent 并行准备。
  - 一个小问题：一次提交带进了子 agent 还在生成中的 `patches/meta/` 草稿，等补丁集定下来后整理。
- 15:10 工作目录挪进 Infera 分支 `dev/pd_opt/vllm_glm52_agentx` 的 `llying/vllm-p4d4-aus-20261127/`；原路径 `bench_agentx/vllm-p4d4-aus-20261127` 是指向这里的软链接（37019 的批处理脚本写的是原路径）。
- 14:55 提交 37019（2 节点、24 h、`afterany:37014`），因为 QOS 单作业最长 24 h；用户在问为什么又提交了一个作业，取不取消由用户决定。
- 14:00 核对测试方法：吞吐的计法与 InferenceX 一致（a49 复算 ≈ 71,07x）；eval 不一致（应该是全量 GSM8K 1,319 题、并发等于计分并发、阈值 0.90，外加 AL 门禁），客户端 918524ff 比 main 落后 3 个月。详见 PLAN 第 9 节。
- 13:50 集群信息：aus 节点混有 amdgpu 6.16.13 / 6.19.14；网卡和 crusoe 一样是 ionic_0..7、GID index 1；模型在 `/perf_apps/data/models/GLM-5.2-MXFP4`；docker 以 root 运行，读不了 /home（NFS root-squash），所有挂载路径都放在 /perf_apps 下；节点上没有 /scratch。
