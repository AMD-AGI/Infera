# Scripts

Every script that ran, verbatim. Ordered as `../REPRODUCE.md` uses them.

| script | step | what |
|---|---|---|
| `build_image.sh` | 1 | one-stage build on a node; runs the docker build in the **foreground** of `spur exec` deliberately |
| `start_ctr.sh` | 2 | container + GPU/RDMA gate + the 9-check **bytecode gate** |
| `start_services.sh` | 4 | etcd (prefill only) + the infera-kvd daemon (both) |
| `ab_boot.sh` | 5, 6 | boot one PD leg; writes an env file to shared storage, runs it with `docker exec -d` |
| `glm52_leg_spur_mtp.sh` | 5, 6 | the leg itself — all engine flags and env live here |
| `ab_wait.sh` | 6 | poll both legs to readiness over HTTP (never by log grep) |
| `ab_router.sh` | 7 | the router, with a selectable `POLICY` |
| `ab_bench.sh` | 10 | drive Optimus-AgenticBench from the **login node** |
| `collect_env.sh` | 11 | environment snapshot — **shipped but never run**; see `../env/README.md` |

The `ab_*.sh` scripts are new in this kit: the predecessor `boot.sh` / `router.sh`
hardcoded a since-dead allocation's job ids and IPs, and had no way to pass
`DPA` / `CHUNK` / `GMU` or a router policy through — all four of which the A/B pair
needs.

`glm52_leg_spur_mtp.sh` carries two edits from its Case A ancestor; both are in
`../patches/`. They are not tuning — each removes a way that flipping `DPA` would
silently have changed a second variable.
