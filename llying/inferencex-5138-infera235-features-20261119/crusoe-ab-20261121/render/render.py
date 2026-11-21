#!/usr/bin/env python3
"""Render every configuration in variants.yaml as the InferenceX CI would hand it to srtctl: srtctl's own base/override
merge of #5138's disagg-variants.yaml, the caller --set overrides, then InferenceX's golden AgentX acceptance
(infx.srt_slurm.synthetic_acceptance.build_overrides). Writes out/<name>.yaml and out/<name>.json (for the stdlib-only
harness on the nodes) and prints the flags that differ.

Run with src/venv/bin/python (srtctl at the InferenceX pin 2d9079c plus runners/srt-slurm/patches).
"""
import copy
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
E2E = HERE.parent / "src/InferenceX/inferencex-e2e"
sys.path.insert(0, str(E2E))

from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides  # noqa: E402

from infx import yamlio  # noqa: E402
from infx.srt_slurm.synthetic_acceptance import build_overrides, selected_recipes  # noqa: E402

RECIPE = E2E / "benchmarks/multi_node/srt-slurm-recipes/glm5.2/infera-sglang/mi355x-fp4/agentx/disagg-variants.yaml"
FRAMEWORK = "infera-sglang"
ENVIRONMENT = {"MODEL_PREFIX": "glm5.2", "THINKING_MODE": "thinking_on", "EVAL_ONLY": "false",
               "IS_AGENTIC": "1", "SPEC_DECODING": "mtp"}
SHOWN = ("tp-size", "dp-size", "pp-size", "speculative-num-steps", "speculative-num-draft-tokens",
         "mem-fraction-static", "stream-interval", "cuda-graph-max-bs-decode", "max-running-requests")
SHOWN_ENV = ("SGLANG_SIMULATE_ACC_LEN", "SGLANG_SIMULATE_ACC_METHOD", "SGLANG_SIMULATE_ACC_TOKEN_MODE",
             "SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION")


def render(shape: str, sets: list[str]) -> dict:
    raw = yamlio.safe_load(RECIPE.read_text())
    apply_overrides_to_recipe(raw, parse_overrides(sets, None))
    ((_, recipe),) = selected_recipes(copy.deepcopy(raw), shape)
    generated = build_overrides(recipe, FRAMEWORK, ENVIRONMENT)
    sets_ = [generated[i + 1] for i in range(0, len(generated), 2) if generated[i] == "--set"]
    unsets = [generated[i + 1] for i in range(0, len(generated), 2) if generated[i] == "--unset"]
    apply_overrides_to_recipe(raw, parse_overrides(sets_, unsets))
    ((_, final),) = selected_recipes(raw, shape)
    return final


def summary(cfg: dict) -> dict:
    out = {"frontend": dict(cfg.get("frontend", {}).get("args", {}))}
    for name, role in cfg.get("roles", {}).items():
        args, env = role.get("args", {}), role.get("env", {})
        out[name] = {"gpus": role.get("gpus"), "workers": role.get("workers"),
                     **{k: args[k] for k in SHOWN if k in args}, **{k: env[k] for k in SHOWN_ENV if k in env}}
    return out


def main() -> None:
    spec = yaml.safe_load((HERE / "variants.yaml").read_text())["variants"]
    (HERE / "out").mkdir(exist_ok=True)
    for name, v in spec.items():
        sets = [s for group in v["set"] for s in (group if isinstance(group, list) else [group])]
        cfg = render(v["shape"], sets)
        (HERE / "out" / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, width=120))
        (HERE / "out" / f"{name}.json").write_text(json.dumps(cfg, indent=1) + "\n")
        print(f"## {name} ({v['shape']})")
        for part, values in summary(cfg).items():
            print(f"  {part}: {values}")


if __name__ == "__main__":
    main()
