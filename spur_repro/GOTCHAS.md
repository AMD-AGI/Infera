# spur 集群使用潜规则 / 踩坑记录(what / why / how / context)

> 本次实验最有价值的产出。这些坑没写在任何官方文档里,是实测踩出来的。
> 已同步进 `~/.claude/skills/spur-cluster-usage` 和 `spur-interactive-debug` 两个 skill。

---

## ★坑 #1(最坑):`JobHoldMaxRequeue` —— 部分 idle 节点会 NODE_FAIL 掉所有 GPU 任务

**What**: 提交 GPU 任务,几秒内从 `Reason=None` 翻成 `PENDING Reason=JobHoldMaxRequeue`,
任务永远排不上;甚至会出现"RUNNING 一瞬间又掉回 PENDING"的假象。

**Why**: 某些节点会接受任务分配、但在节点侧启动(launch)时失败(NODE_FAIL),
任务被自动 requeue,几次就撞到 requeue 上限 → 挂起(held)。**这些坏节点恰恰因为会踢掉
每个任务,所以一直显示 `idle`**;而调度器又偏爱把新任务往"看起来空"的节点塞 → 反复失败。
`spur diag` 会看到 `NODE_FAIL` 计数很大且持续上涨(本次实测 972→980+)。

**How(解法)**: 抓住每次失败任务被分到的节点,累积黑名单,`--exclude` 掉再重试:
```bash
# 抓失败任务落在哪个节点:
squeue -j <jobid> -h -o "%N"
# 重试时排除:
sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 08:00:00 \
       --exclude=crsuse2-m2m-036,crsuse2-m2m-149 hold_node.sh
```
排除几轮就会落到好节点。**★必须验证真的占住**:`spur exec <job> true` 要退出 0;
任务一闪 RUNNING 又 requeue 不算数。本次排除 036、149 两个坏节点后落到 292 成功。

**Context**: 一开始被这个坑带偏,误判成"集群整体宕机""账户额度满""8 卡申请不到"。
真正定位靠**控制变量二分**:CPU-only vs GPU、idle 节点 vs mix 节点、逐一固定卡数/walltime/
账户/QOS。结论:同一条命令在 mix 节点(`Reason=Resources`,健康等待)正常、在坏 idle 节点
(`JobHoldMaxRequeue`)失败 → 锁定是节点问题,与我的提交参数无关。

---

## ★坑 #2:提交任务的正确姿势 —— `-q amd-burst-qos`,能不带 `-A` 就不带

**What**: 用哪些 flag 提交最稳。

**Why / How(实测结论)**:
- **`-p` 必须是 `amd-spur`**(全集群唯一分区;队名是**账户**不是分区,拿队名当分区会报
  `partition 'xxx' not found`)。
- **`-A`(账户)能不带就不带** —— 默认就是你自己的账户,实测不带 `-A` 的 8 卡任务能正常
  RUNNING。只有属于多个账户、要指定时才带。带了不属于的账户会被拒
  (`user 'xxx' is not associated with account`)。
- **`-q amd-burst-qos` 是推荐姿势**。QOS 决定节点额度上限:`amd-primus-qos`=node 16、
  `amd-collectives-qos`=64、**`amd-burst-qos`=256**。burst 池子大,竞争时更容易排上。
  实测 `-q amd-burst-qos` 和 `-q amd-collectives-qos` 都能用(从 amd-primus 账户)。
- **不要在 `srun` 上带 `--qos`/`--pty`** —— 会报 `unexpected argument`。QOS 只有 `sbatch`
  接受(`-q/--qos`)。GPU 活儿一律走 `sbatch` 占位 + `spur exec`,别用 srun。

最终正确写法:
```bash
sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 08:00:00 hold_node.sh
```

**Context**: 曾经误以为"加 qos 更糟"——那是因为当时正撞坑#1 的坏节点窗口,把 QOS 背了锅。
拆开测清楚后确认 burst-qos 是对的。

---

## ★坑 #3:`spur exec` 是 root@/ 的隔离 namespace —— 容器挂载别用 `$HOME`

**What**: `docker run -v $HOME:$HOME` 后,容器里看不到你 NFS home 下的脚本。

**Why**: `spur exec <job> bash` 进去的是 **root 用户、pwd=/** 的隔离 namespace,
`$HOME` 展开成 `/root`,不是 `/home/yihou`。于是挂进容器的是 `/root`,你的脚本自然看不到。

**How**: 挂载写**绝对路径**:`-v /home/yihou:/home/yihou`。`/shared_nfs` 因为本来就是
绝对路径,所以模型一开始就正常可见。

**Context**: 本次容器起来后 `docker exec ... ls .../r4_server.sh` 报 No such file,
排查发现 `spur exec` 环境外(计算节点 host)能看到脚本、容器里看不到 → 定位到挂载路径错。

---

## ★坑 #4:`docker exec ... &` 后台进程随 spur exec 会话结束被杀

**What**: `spur exec <job> bash -c "docker exec dsv4_r4 bash -c '... &'"` 起的后台压测,
会话一断就没了(日志文件都没生成)。

**Why**: spur exec 会话结束会带走它启动的子进程;裸 `&` 的后台任务不够顽强。

**How**: 用 **`docker exec -d`**(detached)起长任务,它归 dockerd 管,不随会话消失。
server 和长压测都该用 `-d`。

---

## ★坑 #5:冷启动很慢且中途"看起来卡住"

**What**: DSv4 TP8 冷启动约 23 分钟;其中 cuda-graph 捕获阶段日志静默数分钟、GPU 利用率很低。

**Why**: 806G 权重加载约 700s;之后 KV cache 分配 + cuda-graph 捕获阶段输出被缓冲不刷屏,
GPU 忙一下停一下。

**How**: 别急着判死。判活方法:看**显存是否持续上涨**(捕获会吃显存,本次 GPU0 从 130→,
GPU1 涨到 182GB)、进程是否还在(`ps`)。等 `The server is fired up and ready to roll!`
和 Uvicorn 监听 30000。legacy 也说冷启动 ~30min 属正常。

---

## 其他有用事实

- **`sbatch --output`/`--error` 被静默忽略** —— 日志一律写到 `~/spur-<jobid>.out`(合并
  stdout+stderr)。想要自定义位置就在脚本里自己重定向。
- **docker 是每节点自带的 dockerd**(29.6.1,root dir `/mnt/m2m_nobackup/docker`),
  镜像/容器**节点本地**,不在 NFS。A 节点起的容器 B 节点看不到。`docker ps` 会看到**全节点
  所有人**的容器,别误删别人的。
- **spur 原生容器路径**(`spur image import` + `--container-image`)在本部署疑似坏的,
  别用;直接在占位里跑 `docker` 就行(这是集群标准姿势)。
- **GPU 环境变量是 `ROCR_VISIBLE_DEVICES`**,不是 `HIP_VISIBLE_DEVICES`(后者常为空)。
  容器内暴露 GPU 用 `--device /dev/dri --device /dev/kfd --group-add video`(**不是 `--gpus`**)。
- **`ssh <compute-node>` 对普通用户被封**(2026-07-22 起 `AllowUsers ubuntu root` 白名单),
  报错伪装成 publickey。用 `spur exec` 代替(走 spurd,绕过 sshd)。
- **容器不随 job 自动清理** —— scancel 前先 `docker rm -f`,否则容器(占着显存/内存)会泄漏。
