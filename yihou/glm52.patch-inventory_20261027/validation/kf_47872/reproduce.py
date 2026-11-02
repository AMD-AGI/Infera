#!/usr/bin/env python3
"""Run the actual PR #47872 estimator/selector with Triton's CPU interpreter.

The Gluon collector is not launched. CPU candidate packing uses the threshold
from the actual estimator. The supplied kernel source is never modified.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

SOURCE_PATH = "python/sglang/kernels/ops/attention/dsa/triton_topk_transform.py"
REVISIONS = {
    "29f9a820cb2ff2cd14963fceb4385222651703a6e94af59721d267c3b042e12b": "a899828aaf4fa1ee2013c072a19be9115b773c09",
    "9b29d2a628eaaa8df4d77bac17db507475cafd866b676029730d9bb76c9bbe13": "1061c298389facbdeaafced9891a5c7c9b399331",
}


def source_provenance(module_path):
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    commit = REVISIONS.get(digest)
    return {
        "repository": "https://github.com/sgl-project/sglang",
        "commit": commit,
        "source_path": SOURCE_PATH,
        "source_sha256": digest,
        "source_url": (
            f"https://github.com/sgl-project/sglang/blob/{commit}/{SOURCE_PATH}" if commit else None
        ),
    }


def run_case(module, high_count):
    import torch

    length, topk, capacity, page_size = 131072, 2048, 8192, 64
    positions = (torch.arange(8192) // 128) * (length // 64) + torch.arange(8192) % 128
    logits = torch.zeros((1, length), dtype=torch.float32)
    logits[0, positions[:high_count]] = 100.0
    lengths = torch.tensor([length], dtype=torch.int32)
    counters = torch.zeros((1, 64), dtype=torch.int32)
    prefix = torch.empty((2,), dtype=torch.int32)
    keys = logits.view(torch.int32)
    module._topk_est[(1,)](
        keys,
        lengths,
        prefix,
        counters,
        length,
        1,
        CH=128,
        NCH=64,
        TOPK=topk,
        OVER=2,
        T=8192,
        RMAX=128,
        CSTR=64,
        num_warps=8,
    )
    threshold = int(counters[0, 0])
    # These nonnegative inputs have identity order-preserving integer keys.
    selected = torch.nonzero(keys[0] >= threshold, as_tuple=False).flatten()
    candidates = int(selected.numel())
    candidate_keys = torch.full((1, capacity), -(1 << 31), dtype=torch.int32)
    candidate_indices = torch.zeros((1, capacity), dtype=torch.int32)
    stored = min(candidates, capacity)
    candidate_keys[0, :stored] = keys[0, selected[:stored]]
    candidate_indices[0, :stored] = selected[:stored].to(torch.int32)
    counters[0, 1] = candidates
    pages = torch.arange(length // page_size, dtype=torch.int32).unsqueeze(0)
    output = torch.full((1, topk), -777, dtype=torch.int32)
    module._topk_select[(1,)](
        candidate_keys,
        candidate_indices,
        counters,
        lengths,
        keys,
        length,
        pages,
        pages.stride(0),
        output,
        CAND=capacity,
        TOPK=topk,
        CSTR=64,
        PAGE_SHIFT=6,
        BLOCK=4096,
        num_warps=8,
    )
    slots = output[0][output[0] >= 0].long()
    valid_count = int(slots.numel())
    unique_count = int(slots.unique().numel())
    selected_sum = float(logits[0, slots].sum())
    reference_sum = float(torch.topk(logits[0], topk).values.sum())
    passed = valid_count == topk and unique_count == topk and selected_sum == reference_sum
    return {
        "high_sample_positions": high_count,
        "row_length": length,
        "topk": topk,
        "actual_estimator_threshold_key": threshold,
        "candidate_count": candidates,
        "actual_selector_valid_count": valid_count,
        "actual_selector_unique_count": unique_count,
        "actual_selector_padding_count": int((output[0] == -1).sum()),
        "unwritten_count": int((output[0] == -777).sum()),
        "selected_score_sum": selected_sum,
        "torch_topk_score_sum": reference_sum,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, help="Unmodified kernel source file")
    parser.add_argument("--output", required=True, help="Result JSON; parent must exist")
    args = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("Set TRITON_INTERPRET=1 before launching Python")
    module_path = Path(args.module).resolve()
    output_path = Path(args.output).resolve()
    if module_path == output_path:
        raise ValueError("Output must not overwrite the kernel source")
    spec = importlib.util.spec_from_file_location("topk_candidate", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the supplied Python kernel module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {
        "schema_version": 1,
        "source": source_provenance(module_path),
        "execution": "Actual _topk_est and _topk_select under TRITON_INTERPRET=1, CPU tensors",
        "candidate_packing": "CPU construction from the actual estimator threshold",
        "limitation": "Gluon collector and GPU compilation/execution are not covered",
        "cases": [run_case(module, count) for count in (256, 2048)],
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if all(case["passed"] for case in result["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
