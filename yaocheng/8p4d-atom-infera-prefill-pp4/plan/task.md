
## 测试任务

- 参考 ../8p4d-atom-infera 中的环境和测试 plan，build 镜像，然后运行 yaocheng/8p4d-atom-infera-prefill-pp4 prefill pp4 的基线测试。
- 测试当前环境 c48, c80, c120 三个点的数据。


## 测试机器
<!-- 1. ssh [cyao1002@dccs-1334-slurm.prov.aus.ccs.cpe.ice.amd.com](mailto:cyao1002@dccs-1334-slurm.prov.aus.ccs.cpe.ice.amd.com)
2. 用  31626 Compute-D cyao1002 cyao1002  R       0:25      2 smci355-ccs-aus-n02-33,smci355-ccs-aus-n10-29 这个 slurm 的两台机器测试。 -->

1. 本地机器 ssh xiaobche@crsuse2-m2m-136 或 crsuse2-m2m-138
2. 直接本地启动 docker 容器进行验证
3. 本地模型地址：/shared_nfs/models/GLM-5.2-MXFP4