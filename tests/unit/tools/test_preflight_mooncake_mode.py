###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

from infera.tools.preflight.mooncake_mode import _eval_mode_b


def test_mode_b_pins_direct_transfer_engine() -> None:
    nic = {
        "device": "mlx5_0",
        "odp": True,
        "vendor": "Mellanox",
        "link_gbps": 400,
        "gid_index": 3,
    }

    mode = _eval_mode_b([nic], {}, [nic], {"compiled_in": True})

    assert mode["env"]["MC_TE_FILTERS"] == "mlx5_0"
    assert all(not key.startswith("MC_MS_") for key in mode["env"])
    assert mode["launch_flags"] == ["--disaggregation-ib-device mlx5_0"]
