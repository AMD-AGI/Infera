# Server logs (policy: gzip if < 4 MB, else excerpt = head 2 MB + key lines + tail 1 MB)

| dir | source | state |
|---|---|---|
| ofg4c | NFS | complete (correctness round, 252) |
| ofg4, ofg2 | NFS | up to the oai release 05:00 |
| ofg3b | NFS | snapshot (A3 running) |
| pf90b | NFS | prefill only, stopped in bring-up |
| ofg2h.fail1, ofg2h2 | NFS | the two warm-up failures |
| ofg2h3 | NFS | snapshot (A1 C56 running) |
| ofg4b | 136 node-local, streamed over ssh | complete |
| ofg4h | 136 node-local | prefill only; the decode never launched (empty excerpt) |
