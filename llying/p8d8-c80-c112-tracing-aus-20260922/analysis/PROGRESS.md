# 实验现场

阶段：适配诊断镜像与 runner，旧服务显存释放中。
用户已授权在两节点执行实验、保存结果、阶段性提交；不需要再次询问。
Prefill stop 命令出现未收到 exit event；禁止GPU reset，需检查实际退出与显存释放。
Decode旧服务已停止，其他用户的 xiaoming-dev 保留。

计划：镜像构建→双节点idle gate→launch+smoke→C80→C112→phase/request关联与瓶颈/均衡分析。
优先获得正式数据，节点剩余时间有限；不可把未完成实验标成成功。
