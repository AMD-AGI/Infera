import json, glob, os, re, sys
rows = {}
for t in ("triton-rocm720-1113", "triton-rocm10-1112"):
    for j in glob.glob(f"{t}/**/_sparse_mla_fwd_split_dim_kernel.json", recursive=True):
        d = os.path.dirname(j); m = json.load(open(j))
        ttir = open(f"{d}/_sparse_mla_fwd_split_dim_kernel.ttir").read()
        st = re.search(r"scf\.for .* step %c(\d+)_i32", ttir)
        bn = int(st.group(1)) if st else None
        asm = open(f"{d}/_sparse_mla_fwd_split_dim_kernel.amdgcn").read()
        g = lambda k: int(re.search(rf"\.{k}:\s+(\d+)", asm).group(1))
        asyncc = m.get("TRITON_HIP_USE_ASYNC_COPY", m.get("use_async_copy", "?"))
        key = (bn, m["num_warps"], m["num_stages"], str(asyncc))
        rows.setdefault(key, {})[t] = (g("vgpr_count"), g("agpr_count"), g("vgpr_spill_count"), g("private_segment_fixed_size"))
print(f"{'BLOCK_N,warps,stages,async':28s} {'3.7: vgpr/agpr/spill/scratch':32s} {'3.8: vgpr/agpr/spill/scratch':32s}")
for k in sorted(rows, key=lambda k: (k[3], k[0] or 0, k[1], k[2])):
    a = rows[k].get("triton-rocm720-1113"); b = rows[k].get("triton-rocm10-1112")
    f = lambda x: "/".join(map(str, x)) if x else "-"
    print(f"{str(k):28s} {f(a):32s} {f(b):32s}")
