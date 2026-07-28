# TCP-only failure appendix (superseded)

This is the EARLIER, FAILED mooncake attempt on the **rc6** image (not pd-unified). It is kept for
the record — per "同配置只留成功者" the main 03 experiment is now the mooncake **RDMA** success on
the pd-unified image; this appendix documents why the rc6/TCP path failed.

- `bench_conc64_FAIL.txt` — 50/256 successful, median TTFT 51 s (TCP KV-transfer sessions drop under
  conc load).
- `logs/` — prefill.log floods with `KVTransferError: remote mooncake session not alive`;
  decode.log shows `TcpTransport: listen ...` (TCP was the only working transport on rc6).
- `scripts/` — the rc6 mooncake engine (MODE=tcp/rdma) + orchestrator.

Root cause and the fix are in the parent `../notes.md`.
