# Run data

每次实验必须写入新的 `runs/<run-id>/`，不得覆盖或复用已有目录。

运行目录由采样器写入 `snapshot/`、`sampling/` 和 `logs/`；后续实验 wrapper
还会加入 `launch/`、`bench/`、退出码及开始/结束时间。
