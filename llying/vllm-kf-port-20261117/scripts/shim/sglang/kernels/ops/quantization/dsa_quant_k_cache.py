def fp8_ds_mla_row_width(d_v: int, d_tail: int) -> int:
    """Bytes per fp8_ds_mla row: d_v fp8 values, one f32 scale per 128, a bf16 tail."""
    return d_v + (d_v // 128) * 4 + d_tail * 2
