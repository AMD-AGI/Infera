# User directives relevant to this packup (verbatim, with the working reading)

Source: `/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012/mission.md` ("UPDATE (user,
2026-10-13 ~05:15 UTC ...)"), full copy in `mission.full.md`; live-session items from `analysis/poll.full.md`.

## 2026-10-13 ~05:15 UTC

> 可以，那你申请2-3台oai机器开始吧，如果机器不够的话，可以在135-139之间找一个空闲的来凑（limou占了一个机器一直在干啥，好几天了，没有真实利用率给他杀了）

> 并行展开一下实验：1. 大hi-cache 大conc的我我不理解为啥没有提升，因为之之前强行8 expert开大kv cache是可以到26k, 你去并行/串行做两个实验a.单独让decode使用8 expert腾出更多的kv-cache. b. 让prefill单独开expert, 获得更大的kv-cache观察下那边有提升，看看谁是瓶颈(conc就按昨天实验最高的我记得是52？8 expert怎么玩你可以去~/dev/git/infera.glm52.pd/yihou底下的packup找)。2. 关于decode限制transfer和prealloc cache那个事情。实验验证一下直接用最初conc=40那一版的hi-cache容量压conc=56/64, 期望是比~/dev/git/infera.glm52.pd/yihou之前sweep的性能回退少一点。

Reading of item 2 (this packup): Track B at high load: runshare v2 (decode in-flight transfer/prealloc KV cap) on the
original C40 config HiCache capacity, pushed to C56 / C64; expectation: less regression than the earlier sweep in the
yihou packups. (Item 1 = the expert8 / x8d / x8p rounds; other packups.)

## 2026-10-13 ~11:55 UTC (live session; recorded in the env file header)

Asked whether to run a no-patch control of tb56 on the same node and config; user: **"要"** -> round `tb56n`
(`scripts/harness/env.tb56n-085.yihou.sh`: "NO-PATCH CONTROL for tb56").

## Also on 2026-10-13 (approval, recorded in the session brief)

Orphan containers `crusoe-sgl-q36` of another user (guanchen) on 085/274/206 were **docker-stopped (not removed)** with
the user's approval before the rounds used those GPUs.

## Original Track B rule (2026-10-12, context for the patch)

> (要申请的量 + 已经占有的量) − running req 占用的量 < 50 % 才可以 … 为 running 和预备 running 的请求留足空间
