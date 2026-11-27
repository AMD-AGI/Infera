#!/usr/bin/env python3
"""Tests for the CPU-tier copy-coalescing overlay (make_kvcoalesce_overlay.py).
  test_kvcoalesce.py OVERLAY_GPU_WORKER.py          numpy only: the merge on synthetic descriptors
  test_kvcoalesce.py OVERLAY_GPU_WORKER.py import   also import the module (overlay mounted over the image's file)
The merge must copy exactly the same bytes to the same places as the unmerged descriptors: replay both against a
small source / destination byte array (descriptors are offsets into them) and compare the destinations.
"""
import ast
import sys

import numpy as np


def load_helpers(path):
    """vpd_coalesce / vpd_count_runs from the overlay source, without importing vllm."""
    tree = ast.parse(open(path).read())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("vpd_coalesce", "vpd_count_runs")]
    assert len(funcs) == 2, "helpers not found in the overlay"
    ns = {"np": np}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), ns)
    return ns["vpd_coalesce"], ns["vpd_count_runs"]


def replay(src_buf, n_dst, s, d, z):
    out = np.zeros(n_dst, dtype=np.uint8)
    for a, b, k in zip(s, d, z):
        out[b:b + k] = src_buf[a:a + k]
    return out


def case(rng, kind, n, page):
    """Descriptors for n blocks of `page` bytes, laid out like (block, layer) copies."""
    if kind == "runs":        # contiguous on both sides, in runs of random length
        s, d = [], []
        sa = da = 0
        i = 0
        while i < n:
            r = int(rng.integers(1, 9))
            for j in range(min(r, n - i)):
                s.append(sa + j * page); d.append(da + j * page)
            sa += (r + int(rng.integers(1, 4))) * page
            da += (r + int(rng.integers(0, 3))) * page
            i += r
        s, d = s[:n], d[:n]
    elif kind == "reversed":  # GPU side descending (free-queue order), CPU side ascending: nothing merges
        s = [(n - 1 - i) * page for i in range(n)]
        d = [i * page for i in range(n)]
    elif kind == "random":    # a random permutation on both sides
        s = list(rng.permutation(n) * page)
        d = list(rng.permutation(n) * page)
    elif kind == "all":       # one long run
        s = [i * page for i in range(n)]
        d = [7 * page + i * page for i in range(n)]
    elif kind == "src-only":  # adjacent on the source side only: must not merge
        s = [i * page for i in range(n)]
        d = [2 * i * page for i in range(n)]
    else:
        raise ValueError(kind)
    z = [page] * n
    return np.array(s, dtype=np.int64), np.array(d, dtype=np.int64), np.array(z, dtype=np.int64)


def main():
    coalesce, count_runs = load_helpers(sys.argv[1])
    rng = np.random.default_rng(0)
    expect_merge = {"runs": True, "reversed": False, "random": None, "all": True, "src-only": False}
    for kind, merges in expect_merge.items():
        for n in (0, 1, 2, 3, 17, 500):
            for page in (8, 64):
                s, d, z = case(rng, kind, n, page)
                n_src = int((s + z).max()) if n else 0
                n_dst = int((d + z).max()) if n else 0
                src_buf = rng.integers(0, 256, size=max(n_src, 1), dtype=np.uint8)
                ref = replay(src_buf, max(n_dst, 1), s, d, z)
                runs = count_runs(s, d, z, n)
                # pad the buffers like the pinned descriptor buffers (capacity > count)
                S = np.concatenate([s, np.full(5, -1, np.int64)])
                D = np.concatenate([d, np.full(5, -1, np.int64)])
                Z = np.concatenate([z, np.full(5, -1, np.int64)])
                m = coalesce(S, D, Z, n)
                assert m == runs, (kind, n, page, m, runs)
                assert (S[n:] == -1).all() and (D[n:] == -1).all() and (Z[n:] == -1).all(), "wrote past n"
                got = replay(src_buf, max(n_dst, 1), S[:m], D[:m], Z[:m])
                assert (got == ref).all(), (kind, n, page)
                assert int(Z[:m].sum()) == int(z.sum()), "bytes changed"
                if n >= 3 and merges is True:
                    assert m < n, (kind, n, m)
                if merges is False:
                    assert m == n, (kind, n, m)
    print("coalesce: same bytes as the unmerged descriptors on runs / reversed / random / single-run / src-only cases")
    if len(sys.argv) > 2 and sys.argv[2] == "import":
        import vllm.v1.kv_offload.cpu.gpu_worker as gw
        assert hasattr(gw, "vpd_coalesce") and hasattr(gw, "_VPD_COALESCE"), "overlay not the module imported"
        print("import: overlay gpu_worker imported; coalesce", gw._VPD_COALESCE, "stats", gw._VPD_STATS)


if __name__ == "__main__":
    main()
