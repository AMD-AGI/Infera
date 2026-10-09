"""Opt-in synthetic GLM experiment: retain eight real routed experts, not model quality."""
import logging
import re

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)
_EXPERT = re.compile(r"\.mlp\.experts\.(\d+)\.")
_LAYER = re.compile(r"^model\.layers\.(\d+)\.")


def enabled(config):
    active = bool(getattr(config, "yihou_synthetic_expert8", False))
    if active:
        assert config.n_routed_experts == config.num_experts_per_tok == 8
        assert config.n_group == config.topk_group == 1
        assert config.scoring_func == "sigmoid"
    return active


def keep_weight(name, num_hidden_layers):
    expert = _EXPERT.search(name)
    layer = _LAYER.match(name)
    return not ((expert and int(expert.group(1)) >= 8) or
                (layer and int(layer.group(1)) >= num_hidden_layers))


def slice_gate(name, tensor):
    if name.endswith((".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")):
        if tensor.shape[0] not in (8, 256):
            raise ValueError(f"unexpected gate shape: {name}: {tensor.shape}")
        return tensor[:8].contiguous()
    return tensor


def weights_iterator(files, num_hidden_layers):
    """Filter names before get_tensor; never materialize discarded experts on GPU."""
    loaded = skipped = tensor_bytes = 0
    for filename in sorted(files):
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if not keep_weight(name, num_hidden_layers):
                    skipped += 1
                    continue
                tensor = slice_gate(name, handle.get_tensor(name))
                loaded += 1
                tensor_bytes += tensor.numel() * tensor.element_size()
                yield name, tensor
    logger.info("YIHOU_EXPERT8 checkpoint loaded=%d skipped=%d retained_bytes=%d", loaded, skipped, tensor_bytes)


def route_all_eight(hidden_states, gating_output, topk, renormalize):
    assert topk == gating_output.shape[-1] == 8
    weights = gating_output.float().sigmoid()
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    ids = torch.arange(8, device=gating_output.device, dtype=torch.int32)
    ids = ids.unsqueeze(0).expand(gating_output.shape[0], -1).contiguous()
    return weights.contiguous(), ids


route_all_eight.yihou_expert8 = True
