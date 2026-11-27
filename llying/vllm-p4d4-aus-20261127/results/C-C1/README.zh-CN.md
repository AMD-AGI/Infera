# 阶段 C：vLLM nightly de7cf7f8 + 12 个上游 / fork PR（C1），n04-25，11-27

镜像 `infera-vllm:nightly-de7cf7f8-C1`（`docker/Dockerfile.nightly` + `patches/series-C1.txt`，每个补丁是对应 PR 的原样 diff），
kit 变体 `C1`（`kit/variants/C1.env`），C152，草稿 4（合成接受 3.33），3600 s，客户端 InferenceX main d490919c（agentx-harness 1.0.14）。

| | C1（n04-25） | a49 replay（n04-29，v0.34 + 覆盖层，客户端 918524ff） |
|---|---|---|
| tok/s/GPU | **70,672** | 69,682 |
| req/s | 4.52 | 4.47 |
| 交互 p50 / p90 | 70.1 / 59.6 | 71.5 / 59.4 |
| ITL p50 / p90（ms） | 14.3 / 16.8 | 14.0 / 16.8 |
| TTFT p50 / p90 / 均值（s） | 4.78 / 17.5 / 8.36 | 5.18 / 17.4 / 8.40 |
| 失败请求 | 17 / 18,105（InvalidInferenceResultError） | 11 |
| GSM8K（InferenceX，C152，真实接受） | 0.9682 | 0.9689 |
| AL 门禁（golden 3.33） | 3.254（0.977，pass） | — |

节点和客户端版本都不同，这一对只说明 C1 与 crusoe 最好配置处在同一水平；同节点对比见 C1 在 n04-29 上的计分（排队中）。
