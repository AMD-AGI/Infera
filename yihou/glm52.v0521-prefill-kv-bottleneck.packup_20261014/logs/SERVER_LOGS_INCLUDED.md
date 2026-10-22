# Server logs included (2026-10-14, user-approved): gzip if < 4 MB, else excerpt = head 2 MB + key lines + tail 1 MB
| round | file | original bytes | packed as | packed bytes |
|---|---|---|---|---|
| x8d | server-logs/decode-0.a1.log | 28091013 | full gzip | 1711906 |
| x8d | server-logs/prefill-0.log | 8920213 | full gzip | 752010 |
| x8p | server-logs/decode-0.a1.log | 32731758 | full gzip | 2150048 |
| x8p | server-logs/prefill-0.log | 24909986 | full gzip | 1656119 |
| x8d2 | server-logs/decode-0.a1.log | 38404692 | full gzip | 2727998 |
| x8d2 | server-logs/prefill-0.log | 24866199 | full gzip | 2027045 |
| x8pc | server-logs/decode-0.a1.log | 32732443 | full gzip | 2137499 |
| x8pc | server-logs/prefill-0.log | 21777533 | full gzip | 1560719 |
| x128p | server-logs/decode-0.a1.log | 32635045 | full gzip | 2130159 |
| x128p | server-logs/prefill-0.log | 17863921 | full gzip | 1389210 |
