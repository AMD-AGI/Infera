# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""OS-level confinement: the chain, its selection, and the two bindings.

Spec §4. The boundary is the kernel's, not ours — a hook is a first gate and a
diagnostic. Above the decoupling wall (design §2.1).
"""
