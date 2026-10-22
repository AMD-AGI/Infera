# Server logs included (2026-10-14, user-approved): gzip if < 4 MB, else excerpt = head 2 MB + key lines + tail 1 MB
| round | file | original bytes | packed as | packed bytes |
|---|---|---|---|---|
| tb56 | server-logs/decode-0.a1.log | 89163045 | excerpt (head 1 MB, Decode batch 1/200, RUNSHARE block 1/20, tail 500 KB) | 415672 |
| tb56 | server-logs/prefill-0.log | 23395199 | full gzip | 2079021 |
| tb56n | server-logs/decode-0.a1.log | 37552437 | full gzip | 2664822 |
| tb56n | server-logs/prefill-0.log | 22935634 | full gzip | 2049170 |
