###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Runtime hooks injected into the ATOM engine process.

Unlike vLLM / SGLang (which publish KV cache events natively via
``--kv-events-config``), ATOM has no native event stream. The modules in this
package are **runtime monkey-patches** loaded *into the ATOM subprocess* (via a
site ``.pth`` → :mod:`infera.engine.atom.hooks.kv_event_bootstrap`) that bolt
KV-aware-routing support onto ATOM's existing internal prefix-cache index
**without modifying ATOM's source**.

They are deliberately kept out of the normal launcher code
(``infera/engine/atom/{__main__,args,worker}.py``) because they are engine
internals patching, not Infera product logic — and they run in a different
process (the spawned ``EngineCore``) than the launcher.
"""
