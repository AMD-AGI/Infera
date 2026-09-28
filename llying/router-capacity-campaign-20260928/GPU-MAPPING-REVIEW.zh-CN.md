# HIP与ROCm-SMI编号复核

实机HIP ordinal到ROCm-SMI card映射为：0→card3、1→card0、2→card2、3→card1、4→card7、5→card4、6→card6、7→card5。两者通过PCI bus地址对应，不能把相同数字视为同一GPU。

当前所用连续0–3/4–7两组在这台机器上恰好仍对应相同card集合，既有8卡结果不受影响。后续仍显式通过PCI映射检查空闲显存，避免依赖编号或分组巧合。映射按节点boot_id缓存，每次SMI读取重新按PCI匹配；缺失或重复PCI地址会拒绝启动。

3项检查覆盖跨半节点的非平凡编号映射、缺失PCI地址和重复PCI地址，均通过。新逻辑已用于B4的D重启，真实映射记录见gpu-map-example.json。

RDMA设备选择保持原配置语义：分组worker的本地GPU编号映射到原配置中相应全局HIP编号的NIC，不改成按SMI编号选择，也不在本轮改动NIC亲和策略。固定镜像的mooncake_transfer_engine.py中get_ib_devices_for_gpu直接按传入GPU编号查配置。
