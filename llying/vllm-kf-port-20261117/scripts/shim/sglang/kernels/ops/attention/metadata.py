import torch


def q8kv8_topk_length_from_indices(indices: torch.Tensor) -> torch.Tensor:
    """Per row, one past the last non-negative index (rows are left-packed)."""
    return (indices >= 0).sum(dim=-1, dtype=torch.int32)
