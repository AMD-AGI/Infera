# SPDX-License-Identifier: Apache-2.0
"""Keep AITER sparse MLA enabled for the finite synthetic benchmark KV fill."""

from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
)


class CanonicalDecodeBenchConnector(DecodeBenchConnector):
    """Reuse the native grouped cache allocator and filler for benchmark requests."""

    @classmethod
    def only_restores_canonical_kv(cls, kv_transfer_config) -> bool:
        # This is the fill verified by the DSV4.1 tool-3 serving reference.
        # The native connector encodes finite FP8 constants and zeroes packed
        # and circular state, preserving the sparse MLA finite-cache assumption.
        extra = kv_transfer_config.get_from_extra_config
        return (
            extra("fill_mean", 0.015) == 0.015
            and extra("fill_std", 0.0) == 0.0
            and extra("startup_fill", False) is False
        )
