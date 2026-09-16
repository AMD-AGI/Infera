# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Remote access, and its tool-call surface. Design §10.

**The far side is less confined than the near side, and that is written down.**
§4 confines a *local* process; a command sent over ``ssh`` runs in whatever the
far side provides and nothing here confines it. `Confinement` is reported per
side so a run's record says so rather than implying otherwise.
"""
