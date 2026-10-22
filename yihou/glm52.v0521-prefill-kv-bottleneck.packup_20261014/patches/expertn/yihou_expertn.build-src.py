"""Opt-in synthetic GLM experiment (task v0521cap2): keep only the first N routed experts of a 256-expert checkpoint.
Unlike yihou_expert8 (all 8 routed, custom routing), routing is the model's own (sigmoid top-k over N experts); only the
checkpoint is truncated: experts >= N are never read, the router gate and its correction bias are sliced to N rows.
Enabled by the config key yihou_synthetic_keep_experts = N (set together with n_routed_experts = N). Not model quality."""
import logging
import re

from safetensors import safe_open

logger = logging.getLogger(__name__)
_EXPERT = re.compile(r"\.mlp\.experts\.(\d+)\.")


def keep_n(config):
    n = getattr(config, "yihou_synthetic_keep_experts", None)
    if n:
        assert config.n_routed_experts == n, (config.n_routed_experts, n)
        assert n >= config.num_experts_per_tok
    return n


def weights_iterator(files, n):
    loaded = skipped = tensor_bytes = 0
    for filename in sorted(files):
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                m = _EXPERT.search(name)
                if m and int(m.group(1)) >= n:
                    skipped += 1
                    continue
                tensor = handle.get_tensor(name)
                if name.endswith((".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")):
                    if tensor.shape[0] not in (n, 256):
                        raise ValueError(f"unexpected gate shape: {name}: {tuple(tensor.shape)}")
                    tensor = tensor[:n].contiguous()
                loaded += 1
                tensor_bytes += tensor.numel() * tensor.element_size()
                yield name, tensor
    logger.info("YIHOU_EXPERTN keep=%d checkpoint loaded=%d skipped=%d retained_bytes=%d",
                n, loaded, skipped, tensor_bytes)
