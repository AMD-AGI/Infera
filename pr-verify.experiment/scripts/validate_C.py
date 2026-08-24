#!/usr/bin/env python3
"""Equivalence + scope check for the DSA decode DP host-sync fix (2a/2a2).

The deadlock itself needs gfx950 + PD + DP-attention + MTP, which this machine
does not have (see working_process.md). What IS checkable here, on real GPU
tensors, is the two claims the fix rests on -- each of which, if wrong, is a
correctness bug rather than a missed optimisation:

  A. `self.req_to_token.shape[1]` is a sync-free substitute for
     `int(seq_lens.max().item())`, and over-allocating page-table columns is
     SAFE because the extra columns are masked by cache_seqlens through top-k.
  B. The two host mirrors removed by 2a2 are genuinely dead on the
     DRAFT_EXTEND_V2 path -- i.e. DRAFT_EXTEND_V2 is not is_extend(), which is
     what gates every consumer of them in this file.

Plus the thing that makes A worth doing at all:

  C. `.max().item()` really is a device-to-host sync (so a branch taken by only
     some DP ranks really does desynchronize them), while `.shape[1]` is not.

Run inside a ROCm/CUDA sglang container with upstream on PYTHONPATH.
"""

import sys
import time

import torch

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


# ------------------------------------------------- C. is .item() really a sync?


def test_item_is_a_sync():
    print("\n[C] .max().item() is a D2H sync; .shape[1] is not")
    stream = torch.cuda.Stream()
    n = 4096
    # Warm up so the timings below measure the sync, not first-call overhead.
    with torch.cuda.stream(stream):
        a = torch.randn(n, n, device="cuda")
        b = torch.randn(n, n, device="cuda")
        acc = torch.zeros(n, n, device="cuda")
        for _ in range(30):
            acc = acc + (a @ b)
        _ = acc[0, :64].to(torch.int32).max().item()
    torch.cuda.synchronize()

    # Re-queue the same work, then time each candidate.
    with torch.cuda.stream(stream):
        acc2 = torch.zeros(n, n, device="cuda")
        for _ in range(30):
            acc2 = acc2 + (a @ b)
        seq_lens2 = acc2[0, :64].to(torch.int32)
    t0 = time.perf_counter()
    _ = int(seq_lens2.max().item())
    dt_item = time.perf_counter() - t0

    req_to_token = torch.zeros(8, 4096, dtype=torch.int32, device="cuda")
    with torch.cuda.stream(stream):
        acc3 = torch.zeros(n, n, device="cuda")
        for _ in range(30):
            acc3 = acc3 + (a @ b)
    t0 = time.perf_counter()
    _ = req_to_token.shape[1]
    dt_shape = time.perf_counter() - t0
    torch.cuda.synchronize()

    check(
        ".max().item() blocks on queued device work",
        dt_item > 1e-4,
        f"-- took {dt_item * 1e3:.2f} ms",
    )
    check(
        ".shape[1] does not block",
        dt_shape < dt_item / 10,
        f"-- took {dt_shape * 1e6:.1f} us vs {dt_item * 1e3:.2f} ms",
    )
    print(f"        (.item()={dt_item * 1e3:.2f} ms, .shape[1]={dt_shape * 1e6:.1f} us)")


# ------------------------------- A. over-allocating page-table columns is safe


def test_wide_page_table_is_masked():
    print("\n[A] extra page-table columns are never selected")
    bs, real_max, pool_width, topk = 4, 37, 4096, 16
    req_to_token = torch.arange(bs * pool_width, dtype=torch.int32, device="cuda").reshape(
        bs, pool_width
    )
    seq_lens = torch.tensor([37, 12, 5, 30], dtype=torch.int32, device="cuda")

    narrow = req_to_token[:, : int(seq_lens.max().item())]  # what main computes
    wide = req_to_token[:, : req_to_token.shape[1]]  # what the fix computes

    check("the fix yields a wider table", wide.shape[1] > narrow.shape[1])
    check(
        "the wide table's leading columns are identical to the narrow one",
        torch.equal(wide[:, : narrow.shape[1]], narrow),
    )

    # Model the masking the kernel does: per row, positions >= cache_seqlens are
    # set to -inf, so top-k can only draw REAL selections from the first
    # seq_lens[i] columns.
    #
    # NB: top-k must still return `topk` indices per row, so a row with
    # seq_len < topk gets surplus slots whose VALUE is -inf. Those exist
    # identically at both widths -- they are a consequence of seq_len < topk,
    # not of widening -- so the property to assert is about the real
    # selections, not about every returned index.
    torch.manual_seed(0)
    base = torch.randn(bs, wide.shape[1], device="cuda")

    def run(width):
        lg = base[:, :width]
        pos = torch.arange(width, device="cuda").unsqueeze(0)
        masked = lg.masked_fill(pos >= seq_lens.unsqueeze(1), float("-inf"))
        return masked.topk(topk, dim=1)

    vals_n, sel_n = run(narrow.shape[1])
    vals_w, sel_w = run(wide.shape[1])

    n_real = [int(min(int(seq_lens[i]), topk)) for i in range(bs)]

    check(
        "every REAL selection is identical narrow vs wide",
        all(torch.equal(sel_n[i, : n_real[i]], sel_w[i, : n_real[i]]) for i in range(bs)),
        "-- widening must not change which tokens are attended to",
    )
    check(
        "their scores are identical too",
        all(torch.equal(vals_n[i, : n_real[i]], vals_w[i, : n_real[i]]) for i in range(bs)),
    )
    check(
        "no REAL selection exceeds its row's seq_len",
        all(bool((sel_w[i, : n_real[i]] < int(seq_lens[i])).all()) for i in range(bs)),
        f"-- so none reaches the widened columns (real_max was {real_max})",
    )
    check(
        "surplus (-inf) slot count is the same at both widths",
        all(
            int(torch.isinf(vals_n[i]).sum()) == int(torch.isinf(vals_w[i]).sum())
            for i in range(bs)
        ),
        "-- widening neither creates nor removes padding slots",
    )


# ------------------ B. the removed mirrors are dead on the DRAFT_EXTEND_V2 path


def test_removed_mirrors_are_dead():
    print("\n[B] DRAFT_EXTEND_V2 does not reach any consumer of the removed mirrors")
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    m = ForwardMode.DRAFT_EXTEND_V2
    check(
        "DRAFT_EXTEND_V2 is NOT is_extend() by default",
        not m.is_extend(),
        "-- this is what makes every consumer unreachable",
    )
    check(
        "DRAFT_EXTEND_V2 IS is_extend(include_draft_extend_v2=True)",
        m.is_extend(include_draft_extend_v2=True),
        "-- sanity: the flag means what we think",
    )
    check("DRAFT_EXTEND_V2 is is_draft_extend_v2()", m.is_draft_extend_v2())

    # DSAMetadata.seq_lens_sum is stored but never read, so passing None for it
    # (the consequence of not materializing the mirror) changes nothing.
    import inspect

    from sglang.srt.layers.attention import dsa_backend

    body = inspect.getsource(dsa_backend)
    reads = [
        ln.strip()
        for ln in body.splitlines()
        if ".seq_lens_sum" in ln
        and "forward_batch.seq_lens_sum" not in ln
        and "seq_lens_sum:" not in ln
        and "seq_lens_sum=" not in ln
        and not ln.strip().startswith("#")
    ]
    check(
        "DSAMetadata.seq_lens_sum has no reader in dsa_backend",
        not reads,
        f"-- found {reads}",
    )

    # And the backend already declares it does not want the host mirror.
    check(
        "the backend declares needs_cpu_seq_lens = False",
        dsa_backend.DeepseekSparseAttnBackend.needs_cpu_seq_lens is False,
        "-- so a D2H sync in its eager fallback contradicts its own contract",
    )


def main():
    print(f"device: {torch.cuda.get_device_properties(0).name}")
    print(f"torch:  {torch.__version__} hip={torch.version.hip}")
    test_item_is_a_sync()
    test_wide_page_table_is_masked()
    test_removed_mirrors_are_dead()
    print()
    if FAILURES:
        print("RESULT: FAIL —", ", ".join(FAILURES))
        return 1
    print("RESULT: PASS")
    print()
    print("NOT covered here (needs gfx950 + PD + DP-attention + MTP): that the")
    print("DP group actually stops deadlocking. Deferred -- see working_process.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
