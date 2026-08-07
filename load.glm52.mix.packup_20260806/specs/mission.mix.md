## Brief

bench得到glm5.2 agentic bench在AMD + sglang 上mix部署场景下的最优性能。

## Goal

### Task breakdown

1. 测试fixlen（ISL= 74000*10*/OSL=320, ISL= 155k*10*/OSL=3300, ISL = 235k*10/OSL=17K）性能，和inferenceX对齐。conc=(1,8,16,24)
2. 测试单conc: agentic性能，通过简单基于原有case fix yaml copy修改并设置session = 1, max session = 1, max in flight = 1，turn = 1测量以下情况（在刷入cache保证cache hit rate的情况下, 另外conc=1不需要think time delay）：
    1. P50的ISL/OSL 反复测10遍
    2. P90的ISL/OSL 反复测10遍
    3. P99的ISL/OSL 反复测10遍
3. 压测agentic场景，init session = 8, max in flight = 16, max session =24.

每步测试完成，起独立agent对实验使用packup技能packup, 放在git根目录。

## reference material

1. examples/sglang_1p1d_glm5.2: 正式入库的glm5.2启动方法。
2. ../infera/glm52.kvd.kvaware.mtp.pd.dp.kv.event.all.commited.finial: 之前成功运行的packup
3. ../infera/par8.glm52.dpaoff.packup_20260803: 之前成功运行的packup
4. ../verify_example.packup_20260804: 之前成功运行的packup
5. ../Optimus-AgenticBench: agentbench法。

## rules

1. 严格遵循用户级别CLAUDE.md的工作规范，如有调整，请交互式询问用户。
2. 先research、gather information、analysis. 再plan， 再创建子workspace(保证所有的临时实验活动都在其中进行)，再创建CLAUDE.md(备份原先的)，再开始工作。
3. 如有实验/debug部分，可以并行展开（e2e实验比较慢，此举可以提升速度。），并且尽量进行多个猜想, 实验进行多个独立互不干扰猜想的同时验证（比如多个print的代码注入等）。
4. 本目录下有很多之前实验的packup，如非用户允许，不允许参考用户在Materials中指定的以外的packup.
5. 不要用客户的(MAD) bench。
6. Use English when work, Chinese only for report to user.
