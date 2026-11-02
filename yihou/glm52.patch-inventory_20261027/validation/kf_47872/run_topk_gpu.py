#!/usr/bin/env python3
"""Execute an exact PR kernel module and its GPU tests in an older ROCm image.

Only the SGLang namespace and CI-registration hook are shimmed. Torch, Triton,
Gluon and every tested kernel execute normally on the assigned physical GPU.
This is isolated kernel validation, not a whole-server integration test.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import sys
import time
import types
import unittest
from pathlib import Path


def install_namespace(module_path):
    for name in (
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.attention",
        "sglang.kernels.ops.attention.dsa",
        "sglang.test",
        "sglang.test.ci",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, package)
    register = types.ModuleType("sglang.test.ci.ci_register")
    register.register_amd_ci = lambda **kwargs: None
    sys.modules[register.__name__] = register
    name = "sglang.kernels.ops.attention.dsa.triton_topk_transform"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the supplied Python kernel module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.modules["sglang.kernels.ops.attention.dsa"].triton_topk_transform = module
    spec.loader.exec_module(module)
    return module


def biased_sample_case(module):
    import torch

    width, topk, high_count = 131072, 2048, 256
    span = width // module._SMP_NCH
    sample_positions = (
        torch.arange(module._SMP_NCH, device="cuda")[:, None] * span
        + torch.arange(module._SMP_CH, device="cuda")[None, :]
    ).flatten()
    logits = torch.zeros((1, width), dtype=torch.float32, device="cuda")
    logits[0, sample_positions[:high_count]] = 100.0
    lengths = torch.tensor([width], dtype=torch.int32, device="cuda")
    page_table = torch.arange(width // 64, dtype=torch.int32, device="cuda").unsqueeze(0)
    out = module.topk_transform_paged_triton(logits, lengths, page_table, 64, topk)
    torch.cuda.synchronize()
    selected = out[0][out[0] >= 0].long()
    selected_count = selected.numel()
    return {
        "width": width,
        "topk": topk,
        "high_count": high_count,
        "selected_count": selected_count,
        "padding_count": int((out[0] == -1).sum()),
        "unique_count": selected.unique().numel(),
        "all_in_range": bool((selected < width).all()),
        "selected_high_count": int((logits[0, selected] == 100.0).sum()),
        "correct": (
            selected_count == topk
            and selected.unique().numel() == topk
            and bool((selected < width).all())
            and int((logits[0, selected] == 100.0).sum()) == high_count
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-underflow", action="store_true")
    args = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") == "1":
        raise RuntimeError("Unset TRITON_INTERPRET for actual GPU validation")
    if args.output.resolve() in (args.module.resolve(), args.tests.resolve()):
        raise ValueError("Output must not overwrite kernel or test source")
    import torch
    import triton

    started = time.time()
    assert torch.cuda.is_available() and torch.version.hip
    props = torch.cuda.get_device_properties(0)
    assert str(props.gcnArchName).startswith("gfx950"), props.gcnArchName
    torch.manual_seed(101)
    module = install_namespace(args.module)
    counterexample = biased_sample_case(module)
    source = runpy.run_path(str(args.tests))
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(source["TestTritonTopkTransform"])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    expected_result = (
        counterexample["selected_count"] == 256 and not counterexample["correct"]
        if args.expect_underflow
        else counterexample["correct"]
    )
    report = {
        "scope": "Exact-module gfx950 GPU tests; SGLang package namespace and CI registration shim only",
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "triton": triton.__version__,
        "gpu": props.name,
        "arch": props.gcnArchName,
        "visible_devices": {
            k: os.environ.get(k)
            for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        },
        "module_sha256": hashlib.sha256(args.module.read_bytes()).hexdigest(),
        "tests_sha256": hashlib.sha256(args.tests.read_bytes()).hexdigest(),
        "counterexample": counterexample,
        "expected_underflow": args.expect_underflow,
        "expected_counterexample_outcome_observed": expected_result,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skips": len(result.skipped),
        "elapsed_seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if expected_result and result.wasSuccessful() and not result.skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
