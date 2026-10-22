# Server logs included (2026-10-14, user-approved): gzip if < 4 MB, else excerpt = head 2 MB + key lines + tail 1 MB
| round | file | original bytes | packed as | packed bytes |
|---|---|---|---|---|
| hc50 | server-logs/decode-0.a1.log | 17258600 | full gzip | 839157 |
| hc50 | server-logs/prefill-0.log | 278854 | full gzip | 34412 |
