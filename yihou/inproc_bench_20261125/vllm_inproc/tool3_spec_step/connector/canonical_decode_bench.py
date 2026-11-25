"""vLLM's DecodeBenchConnector, declared as restoring only canonical KV (equivalence-reference server only).

Why: in the pinned image, DeepSeek-V4.1 on ROCm keeps the aiter sparse MLA path (required by
--kv-cache-dtype fp8_e4m3) only when `kv_connector_may_load_foreign_kv` is False, i.e. when the connector's
`only_restores_canonical_kv` is True (vllm/models/deepseek_v41/amd/rocm.py:597-629,
distributed/kv_transfer/kv_connector/utils.py:103). llying's decode leg passes that check with Mooncake
`"canonical_kv": true`; the stock DecodeBenchConnector returns False and the engine refuses fp8_e4m3
(measured: v3_srv_shapeA_u173k, ValueError at start-up). Its fill values are finite (constant 0.015 /
zeros), which is what the check protects (no NaNs in unwritten slots). Behaviour is otherwise unchanged.
Load with: --kv-transfer-config '{"kv_connector":"CanonicalDecodeBenchConnector",
"kv_connector_module_path":"canonical_decode_bench","kv_role":"kv_both"}' and this directory (connector/) on PYTHONPATH.
"""
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import DecodeBenchConnector


class CanonicalDecodeBenchConnector(DecodeBenchConnector):
    @classmethod
    def only_restores_canonical_kv(cls, kv_transfer_config) -> bool:
        return True
