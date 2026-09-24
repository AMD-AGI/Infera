## task

1. 用 rocm/atom-dev:nightly 最新的 atom nightly 镜像，参考 deploy/docker/Dockerfile.atom 帮我 build 一份最新的 Infera SGLang 镜像，dockerfile 放到 yaocheng/8p4d-atom-infera/docker。
2. 尝试跑通 GLM-5.2 8P4D 的配置，Prefill TP8 DPA on，Decode TP4 DCP4 EP1, DPA off。必要时可参考 [https://inferencex.semianalysis.com/inference/glm-5-3](https://inferencex.semianalysis.com/inference/glm-5-3) 中 MI355 Atom 的单机配置。
3. 参考 yaocheng/2p1d-sweep-triton-dsa-20260922 生成 agentx 的基础脚本并测试性能。

## 要求：

1. yaocheng/8p4d-atom-infera/.record 实验遇到问题以及相关进展记录到这个目录。
2. yaocheng/8p4d-atom-infera/.tmp 中间脚本都放到这个文件夹，yaocheng/8p4d-atom-infera 中其他文件夹尽量保持干净的环境。
3. 代码尽量精简，人类可读性高，减少防御性编程。

## 实验环境：

1. ssh [cyao1002@dccs-1334-slurm.prov.aus.ccs.cpe.ice.amd.com](mailto:cyao1002@dccs-1334-slurm.prov.aus.ccs.cpe.ice.amd.com)
2. 用  31626 Compute-D cyao1002 cyao1002  R       0:25      2 smci355-ccs-aus-n02-33,smci355-ccs-aus-n10-29 这个 slurm 的两台机器测试。

