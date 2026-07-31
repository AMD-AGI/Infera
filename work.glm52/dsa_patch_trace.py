"""Patch dsa_backend.py IN PLACE (inside the container, before launch) to print the
exact aiter call args. Unlike a sitecustomize hook this runs at the right time and
cannot deadlock the interpreter's aiter import.

Adds a print at the top of _forward_aiter and _prepare_aiter_dsa_decode_metadata.
"""
import pathlib, sys

D = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py")
s = D.read_text()

if "__DSA_TRACE__" in s:
    print("already patched"); sys.exit(0)

# 1) trace the metadata prepare
old_prep = """        self._ensure_aiter_dsa_decode_metadata_buffer(
            max_seqlen_q=max_seqlen_q,
            batch_size=bs,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
        )"""
new_prep = """        print(f"__DSA_TRACE__ prep bs={bs} maxq={max_seqlen_q} "
              f"cap={self.aiter_dsa_metadata_capacity} "
              f"qo={tuple(qo_indptr.shape)} kvindptr={tuple(kv_indptr.shape)} "
              f"nhead_pad={self.num_head_padded} topk={self.dsa_index_topk}", flush=True)
        self._ensure_aiter_dsa_decode_metadata_buffer(
            max_seqlen_q=max_seqlen_q,
            batch_size=bs,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
        )"""
assert old_prep in s, "prep anchor missing"
s = s.replace(old_prep, new_prep, 1)

# 2) trace the decode kernel launch (the _forward_aiter one, bs is in scope there)
old_dec = """        mla_decode_fwd(
            q_kernel,
            kv_cache.view(-1, 1, 1, layer.head_dim),
            o_kernel,
            metadata.cu_seqlens_q,"""
new_dec = """        print(f"__DSA_TRACE__ dec bs={bs} q={tuple(q_kernel.shape)} "
              f"o={tuple(o_kernel.shape)} kv={tuple(kv_cache.shape)} "
              f"head_dim={layer.head_dim} v={layer.v_head_dim} "
              f"cu_q={tuple(metadata.cu_seqlens_q.shape)} "
              f"kvindptr={tuple(kv_indptr.shape)} kvidx={tuple(kv_indices.shape)} "
              f"maxq={metadata.max_seq_len_q} pt={tuple(page_table_1.shape)}", flush=True)
        mla_decode_fwd(
            q_kernel,
            kv_cache.view(-1, 1, 1, layer.head_dim),
            o_kernel,
            metadata.cu_seqlens_q,"""
assert old_dec in s, "dec anchor missing"
s = s.replace(old_dec, new_dec, 1)

D.write_text(s)
print("__DSA_TRACE__ patched OK")
