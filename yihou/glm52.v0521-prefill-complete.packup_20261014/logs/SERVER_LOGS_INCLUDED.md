# Server logs included (2026-10-14, user-approved): gzip if < 4 MB, else excerpt = head 2 MB + key lines + tail 1 MB
| round | file | original bytes | packed as | packed bytes |
|---|---|---|---|---|
| pc48 | server-logs/prefill-0.log | 7508 | full gzip | 2341 |
| pc48b | server-logs/decode-0.a1.log | 32221614 | full gzip | 2108384 |
| pc48b | server-logs/prefill-0.log | 15318339 | full gzip | 1277051 |
| pc48e | server-logs/decode-0.a1.log | 32353232 | full gzip | 2113694 |
| pc48e | server-logs/prefill-0.log | 15424151 | full gzip | 1316570 |
