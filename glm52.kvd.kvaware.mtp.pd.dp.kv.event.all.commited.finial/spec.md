# Merge kvd+kvaware result with mtp+dpa+pd result

## brief

对于infera库中sglang engine + mooncake对glm5.2的支持包括多个并行工作，

1. kvd+kvaware适配、
2. mtp+dpa+pd分别是两个独立的实验系列和修复系列
3. pd + mooncake 的乱码现象。
4. 现在根据现有资料对三者进行合并，以其交付一个最终的面对生产环境完整可用的工具体系和正式的git repo代码。

## Goal

1. 新建worktree和统一experiment分支。
2. 交付最终dockerfile.
3. 交付最终docker build image (放在集群里就行了)，可以通过临时build手段在kvd + kvaware的成果上patch.
4. 最终实验通过，包括：
    1. 验证kvd + kvaware的各项功能生效
    2. mtp+dpa+pd/pd + mooncake工作正确：多次单请求正确性通过，conc=16的正确性基本通过。conc=128的正确性通过。
    3. 最终实验通过后的packup skill创建的pack up。

## Materials

1. : kvware + kvd的正式交付内容: ~/dev/git.16-19/infera.glm5.2.mxfp4.offical
2. : mtp + dpa + pd的正式交付内容：/home/yihou/dev/git.16-19/infera.dp-row.fix、pr58
3. : pd + mooncake的乱码现象交付内容: https://github.com/AMD-AGI/Infera/pull/56/changes#diff-c72fb30b9f9f89d8d29e8727a86227697b91da79f19f99acc5b0f60b36c2ca67
4. : kvware + kvd最终实验packup: kvaware_kvd_pr.packup_20260731.pr.final
5. : mtp + dpa + pd最终实验packup： glm52.mxfp4.spur.mooncake.packup_20260731_main_converged

## rule

1. 严格遵循用户级别CLAUDE.md的工作规范，如有调整，请交互式询问用户。
2. 如有实验/debug部分，可以并行展开（e2e实验比较慢，此举可以提升速度。），并且尽量进行多个猜想, 实验进行多个独立互不干扰猜想的同时验证（比如多个print的代码注入等）。
3. 本目录下有很多之前实验的packup，如非用户允许，不允许参考用户在Materials中指定的以外的packup.
