# 本轮观测边界

运行前/客户端准备期间的源码与smoke复核结果：

- P/D生成独立trace ID，客户端传入的traceparent没有成为统一根。
  本轮依赖客户端注入的完整RID及bootstrap room关联；8个smoke请求均通过。
- 服务启动预热与health check也输出request_summary，不能纳入客户端请求分母。
- 当前运行日志明确使用Unified Radix Cache。镜像补丁的host_load_submitted /
  host_load_ack插桩位于旧hiradix_cache.py；unified_radix_cache.py具有独立
  load_back / loading_check实现，未插桩。因此本轮不能用缺失host事件断言无回迁，
  也不能据此计算逐请求Unified host恢复时长。用实际导出的HiCache回迁计数器
  确认路径是否被使用；若没有足够证据区分host恢复与其他P端等待，结论必须保留。
- 未执行匹配负载与缓存状态的tracing on/off对照，历史aligned C80只能作背景参照，
  不能量化观测开销，也不能把两个并发点的差异视为无观测开销条件下的原始性能曲线。
- 节点间时钟初测有非零偏差与SSH往返不确定度，记录在snapshot/clock-offset.json。
  请求阶段duration优先取同一进程monotonic时钟；跨节点wall-clock顺序需附误差界。
- Prefill forward envelope包含调度与chunk间隙，host ack若存在也是CPU观测完成时间。
  Decode transfer等待覆盖上游排队/计算等阶段，不能和Prefill等待直接相加。

这些限制不影响对客户端性能、P/D关联、allocation阻塞原因、分rank负载、
队列与资源压力的测量，但可能限制“第一个退化的内部子阶段”的精度。
