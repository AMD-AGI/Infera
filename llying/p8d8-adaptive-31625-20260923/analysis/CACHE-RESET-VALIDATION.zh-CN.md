# 复用缓存清空的验证口径

诊断镜像 4f125ff... 中，Scheduler.flush_cache 在 fully_idle 后执行 tree_cache.reset、request pool.clear、GPU allocator.clear 和 reset_metrics。Unified._reset_full 调用 cache_controller.reset 与 mem_pool_host.clear。后者重建全量 free_slots、清空 release_slots、将 slot_used 归零。

实际观察：8 个 P rank 均记录 Cache flushed successfully，GPU used/evictable 和队列归零；但 host-used gauge 仍保留 smoke 的 4,416/43,968 tokens。源码 _maybe_log_idle_metrics 只更新 GPU/队列统计，不调用 _log_hicache_stats；后者只在 batch 统计路径刷新。因此不能用这个 idle gauge 的旧值否定清空，也不能宣称观测到了实时 host-used=0。

本轮验证改为：限定已核验镜像、同一时段 8 rank 清空成功确认、GPU used/evictable 与队列为空、host容量完整、进程/调度器身份保持一致；host-used 陈旧值作为观测限制保留。A0 首次失败及源值保存于 a0-startup/stale-host-gauge.json。所有 A/B/A 点使用相同流程。第一次清空后的等待与重复清空不计入正式测量。

仍需用 A/B/A 检验复用协议下的漂移，包括 JIT/RNG 等非 KV 状态不能完全等同于重启。若回退控制不能复现，不能把处理组变化全部归为路由开关。
