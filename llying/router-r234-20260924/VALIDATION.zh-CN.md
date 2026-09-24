# 本地验证记录

2026-09-24。Rust Router 本地编译及 CPU/mock 测试完成；未启动 GPU 模型服务或性能实验。

- `cargo test --manifest-path rust/Cargo.toml -p infera-router`：267 个库单元测试、25 个 HTTP 功能测试、4 个 ZMQ 集成测试、14 个模板探测测试通过，合计 **310 passed**。
- **4 个 NATS broker 集成测试 ignored，未执行**；不将其记为通过。
- 最后调整启动日志格式后，`cargo build --manifest-path rust/Cargo.toml -p infera-router` 再次通过。这是 debug 构建，不用于性能结论；release 构建留作正式部署准备。
- 改动文件的 `rustfmt`、`git diff --check`、profile 脚本语法检查通过；验证了 combined→legacy 会清除新开关，r3-gpu 的 host 权重为 0。
- Rust 工具链和缺失的 libclang 临时安装在 `/tmp/infera-r234-*`，未修改引擎镜像和 benchmark 客户端依赖。

[测试输出](validation/cargo-test.txt)、[构建输出](validation/cargo-build.txt)、[源码与 debug 二进制哈希](validation/manifest.json)。源码哈希对应最终本地受测代码；仓库提交号由 Git 记录，不在提交内递归写自身哈希。

后续仍需实际 SGLang 输入长度与生命周期对账、Unified Radix 分层事件验证、使用路径的 NATS 验收、Router 开销评估、GPU 跑通及重启 P/D 的可信 A/B。本次检查不能替代这些验收，也没有产生性能收益声明。
