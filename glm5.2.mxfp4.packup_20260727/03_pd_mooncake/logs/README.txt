prefill.log — the no-MTP mooncake RDMA prefill leg (chi2878). Grep "rdma_context.cpp .* HIP dmabuf
  disabled" to confirm the RDMA path; "Disaggregation warmup request completed with status 200" for
  transfer working; no "not alive"/KVTransferError anywhere.
decode.log — NOTE: this file reflects the LATER MTP run (the no-MTP decode leg was relaunched with
  MTP in-place for experiment 06, overwriting its log). The no-MTP decode evidence is the transfer
  success visible from the prefill side + the bench result (results/bench_conc64.txt). For a clean
  no-MTP decode log, re-run scripts/up.sh without the MTP leg.
