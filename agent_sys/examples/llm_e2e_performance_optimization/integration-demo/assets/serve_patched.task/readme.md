# serve_patched

The same bring-up with the overlay applied, and the evidence that the overlay is
live.

## The evidence is gathered here, not by the validator

By the time `check_patch_live` runs, this deployment may already have been torn
down. A validator can only read the handoff, so everything it needs is captured
while the container is still up:

- `env/docker_mounts.json` — what `docker inspect` reports.
- `env/container_hashes.tsv` — what each mounted path hashes **inside the running
  container**. This is the only static proof that works. A bind mount does not
  change the path, so `__file__` reads identically on this arm and the stock one;
  people expect it to be enough and it is not.
- `env/marker_hits.tsv` — how often each marker the patch declared appears in the
  engine log. The import marker says the bytes were compiled; the first-call
  marker says a request entered the patched code.

`mix_up.sh` also compares the host and in-container hashes at start-up and
refuses to continue if they differ, because docker will happily create an empty
file at a mount destination whose source vanished, and the engine would import it
minutes later as an unattributable ImportError.

## Why its edge points at measure_stock

Its first act is an idempotent teardown of whatever holds the GPUs, which
includes the stock deployment. The edge has to mean "the stock arm has been
measured".
