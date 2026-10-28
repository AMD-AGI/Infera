"""CPU semantic controls; these do not certify the GPU candidates."""

import math

import naive
import torch


def cache_with_keys(keys):
    """Encode only the fixture's nonzero values at explicit physical offsets."""
    pages, tokens, dimension = keys.shape
    raw = torch.zeros(pages, tokens * (dimension + 4), dtype=torch.uint8)
    for page in range(pages):
        for token in range(tokens):
            for dim in range(dimension):
                offset = (
                    (token // 16) * 16 * dimension
                    + (dim // 16) * 256
                    + (token % 16) * 16
                    + dim % 16
                )
                raw[page, offset] = keys[page, token, dim].to(torch.float8_e4m3fn).view(torch.uint8)
        raw[page, tokens * dimension :] = torch.ones(tokens).view(torch.uint8)
    return raw


def test_sparse_attention_and_dcp_empty_row():
    query = torch.zeros(2, 1, 1)
    cache = torch.tensor([[2.0, 9.0], [4.0, 7.0]])
    slots = torch.tensor([[[0, 1]], [[-1, -1]]])
    output, lse = naive.sparse_mla(
        query, query, cache, slots, 1.0, d_v=1, lengths=torch.tensor([2, 0])
    )
    torch.testing.assert_close(output, torch.tensor([[[3.0]], [[0.0]]]))
    torch.testing.assert_close(lse, torch.tensor([[1.0], [-float(2**30)]]))


def test_indexer_selects_physical_slots_and_pads_short_rows():
    keys = torch.zeros(2, 64, 128)
    keys[1, 0, 0], keys[1, 1, 0] = 1, 3
    cache = cache_with_keys(keys)
    query = torch.zeros(1, 1, 128)
    query[0, 0, 0] = 1
    result = naive.dsa_indexer(
        query, cache, torch.ones(1, 1), torch.tensor([2]), torch.tensor([[1]]), topk=3
    )
    assert result.tolist() == [[65, 64, -1]]
    torch.testing.assert_close(naive.unpack_index_cache(cache), keys)


def test_dcp_indexer_merges_remote_candidates_before_slot_mapping():
    keys = torch.zeros(2, 64, 128)
    keys[0, 0, 0] = 2
    cache = cache_with_keys(keys)
    query = torch.zeros(1, 1, 128)
    query[0, 0, 0] = 1
    remote = torch.stack(
        (
            torch.tensor([[5.0, -float("inf")]]).view(torch.int32),
            torch.tensor([[64, -1]], dtype=torch.int32),
        )
    )[None]
    result = naive.dcp_indexer(
        query,
        cache,
        torch.ones(1, 1),
        torch.tensor([[0]]),
        torch.tensor([1]),
        torch.tensor([[0, 1]]),
        remote,
        topk=2,
        world_size=2,
    )
    assert result.tolist() == [[64, 0]]


def test_mxfp4_rounding_and_routed_moe():
    values = torch.zeros(1, 32)
    values[0, :4] = torch.tensor([6.0, 2.5, 3.5, -3.5])
    rounded = naive.quant_dequant_mxfp4(values)
    assert rounded[0, :4].tolist() == [6.0, 2.0, 4.0, -4.0]
    w13 = torch.full((1, 64, 16), 0x22, dtype=torch.uint8)
    w2 = torch.full((1, 32, 16), 0x22, dtype=torch.uint8)
    result = naive.moe_mxfp4(
        torch.ones(1, 32),
        torch.tensor([[0]]),
        torch.tensor([[0.25]]),
        w13,
        torch.full((1, 64, 1), 127, dtype=torch.uint8),
        w2,
        torch.full((1, 32, 1), 127, dtype=torch.uint8),
    )
    torch.testing.assert_close(result, torch.full((1, 32), 8192.0))


def test_hadamard_and_rope_known_basis():
    basis = torch.zeros(1, 128)
    basis[0, 0] = 1
    torch.testing.assert_close(naive._hadamard(basis), torch.full_like(basis, 1 / math.sqrt(128)))
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    result = naive._rope(value, torch.tensor([0]), torch.zeros(1, 1), torch.ones(1, 1))
    assert result.tolist() == [[-2.0, 1.0, 3.0, 4.0]]


def test_fused_indexer_writes_only_selected_cache_slot():
    cache = cache_with_keys(torch.zeros(1, 64, 128))
    before = cache.clone()
    result = naive.fused_indexer(
        q_lora=torch.zeros(1, 4, dtype=torch.bfloat16),
        x_bf16=torch.zeros(1, 4, dtype=torch.bfloat16),
        w_q_b=torch.zeros(4096, 4, dtype=torch.bfloat16),
        w_kw=torch.zeros(160, 4, dtype=torch.bfloat16),
        norm_w=torch.ones(128),
        norm_b=torch.zeros(128),
        cos2d=torch.ones(2, 32, dtype=torch.bfloat16),
        sin2d=torch.zeros(2, 32, dtype=torch.bfloat16),
        positions=torch.tensor([0]),
        out_cache_loc=torch.tensor([3]),
        kv_cache=cache,
        seqlens=torch.tensor([1]),
        page_table_64=torch.tensor([[0]]),
        eps=1e-6,
        weights_scale=1,
        topk=2,
    )
    assert result.tolist() == [[0, -1]]
    begin = 64 * 128 + 3 * 4
    scale = cache[0, begin : begin + 4].contiguous().view(torch.float32).item()
    assert scale == 2 ** math.ceil(math.log2(1e-4 / 448))
    restored = cache.clone()
    restored[0, begin : begin + 4] = before[0, begin : begin + 4]
    assert torch.equal(restored, before)
