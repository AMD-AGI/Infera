# 镜像迁移记录

- 时间：2026-09-21 12:48:02Z—12:52:41Z
- 来源：`crsuse2-m2m-136`
- 目标：`crsuse2-m2m-138`
- 方法：`docker save | ssh docker load`，未在 NFS 落 66 GB tar
- 标签：`infera-sglang:v0519-yihou-0917-nextnfix-hicache`
- 加载后 image ID：
  `sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35`
- 大小：`66252559778` bytes

138 原同名标签指向
`sha256:972d8fd952e97b3915d742072874b57a1796c10de1dc0cdcc85e92123167f5d8`。
`docker load` 将旧 image 解除标签但未删除，然后把目标标签指向 136 的精确镜像。

迁移后复核：138 八张 GPU VRAM 均为 0%，无 KFD workload；136 同样空闲。
