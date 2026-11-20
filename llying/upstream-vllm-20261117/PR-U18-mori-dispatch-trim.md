# U18：MoRI 接入（接收缓冲区切片、IntraNode 零拷贝 combine）——切片与上游 #63412 重复，不开 PR

- 查重结论（11-20）：
  - **接收缓冲区按实际行数切片**：上游 open PR [#63412](https://github.com/vllm-project/vllm/pull/63412)（@basemam，10-23 起，11-11 更新）
    "Trim MoRI dispatch output to rows that can hold a token" 是同一个修法——MoRI `dispatch()` 返回整块预分配的接收缓冲区（`num_dispatchers ×
    max_num_tokens` 行），它用主机侧已有的 `DPMetadata.num_tokens_across_dp_cpu` 算上界切片，不引入 GPU 同步（CUDA graph 捕获下也一样）。
    [#53108](https://github.com/vllm-project/vllm/pull/53108)（open）为 AITER 的正确性问题用 `.item()` 切同一个输出。按查重规定不再开。
  - **IntraNode 以 `use_external_inp_buf=False` 创建 op 走 `_p2p` combine**：上游没找到对应 PR。但 A 线测的是 MoRI EPv1（mori 1.2.3.post1，低于 EPv2
    要求的 1.2.4），main 已合入 MoRI EPv2（[#61361](https://github.com/vllm-project/vllm/pull/61361)）；在我们的配置下这一项只多省约 0.4 ms/步
    （55.26 → 54.85 ms），MoRI 又不支持融合共享专家，单节点整体不如默认的非 EP 配置（49.77–52.9 ms）。价值低，先不做；如需要，应在 EPv2 上重测后再定。
- A 线数据（v0.34.0，聚合 GLM-5.2 MXFP4，DP4 EP4，MTP3，关融合共享专家，104 请求 × 64K / 4K，纯 decode 每步 p50，单次）：
  EP + MoRI 63.56 ms → 切片后 55.26 ms → 再加 `_p2p` combine 54.85 ms；EP + RCCL AG/RS（关融合）54.49 ms（A 线 README 3.3、`overlay-mori/`）。

## 给上游 #63412 的留言稿（英文，待用户 review，**未发**）

A data point from a different setup: GLM-5.2-MXFP4 on one MI355X node, DP4 + EP4 with MoRI EPv1 (mori 1.2.3), MTP3,
104 requests x 64K in / 4K out, decode step p50 in the all-decoding window (vLLM v0.34.0, single runs). Slicing the
dispatch output to `num_dispatchers x` the step's tokens, the same idea as this PR, took the decode step from 63.56 ms to
55.26 ms; the expert GEMMs had been picking kernels for the full `max_num_batched_tokens x 4` rows and quantizing all
of them. That was with our own patch, not this branch; happy to rerun this branch on MI355X if useful.
