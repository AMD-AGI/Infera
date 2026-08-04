# glm5.2 example create

## Brief

infera已经提供了glm5.2生产及部署姿势example。

## Goal

使用严格使用example中glm5.2脚本（examples/sglang_1p1d_glm5.2），结合/home/yihou/dev/git/infera.merge.liying.kv.mtp/par8.glm52.dpaoff.packup_20260803中获取的本地集群信息。成功启动glm5.2(sglang + pd[mooncake] + mtp + dpa(decode) + nodpa[prefill] + kvd + kvaware[实际无用，prefill都关了dpa了]等所有特性)部署，验证各功能正确。并在之后参考gentx.caseA.customer.packup_20260803运行客户agent bench

要求：

1. 运行成功后，若example中的脚本需要fix, 则fix, 但是不允许有git操作

## rules

1. 严格遵循用户级别CLAUDE.md的工作规范，如有调整，请交互式询问用户。
2. 先research、gather information、analysis. 再plan， 再创建子workspace(保证所有的临时实验活动都在其中进行)，再创建CLAUDE.md(备份原先的)，再开始工作。
3. 如有实验/debug部分，可以并行展开（e2e实验比较慢，此举可以提升速度。），并且尽量进行多个猜想, 实验进行多个独立互不干扰猜想的同时验证（比如多个print的代码注入等）。
4. 本目录下有很多之前实验的packup，如非用户允许，不允许参考用户在Materials中指定的以外的packup.
5. Use English when work, Chinese only for report to user.
