# Patches — what differs from the tracked repo, and why

The harness in `../scripts/bench-harness/` is a **vendored copy** of
`bench/glm5p2_pd/` from this repo. `diff -rq` against the tracked tree shows
**exactly three** files differ; each diff here is one same-node defect. The
tracked repo was never modified.

Every change is a **no-op for a cross-node deployment** — that was a design
constraint, so the same harness still serves the existing 1P1D/2P1D/P8D8 work.

| diff | file | what it fixes |
|---|---|---|
| `tools_topology.py.diff` | `tools/topology.py` | **Defect 1.** `load()` raised on `node in nodes or data_ip in ips`, which a single host cannot satisfy — one node, one IP. Relaxed to reject only an exactly-duplicated `(role, node, ip)` row. Role whitelist, IPv4 check, at-least-one-of-each, and the `index`/`instance` numbering are all preserved — every port derives from the row index, so that numbering is load-bearing. |
| `launch.sh.diff` | `launch.sh` | **Defects 2 and 3.** (a) `ENGINE_PORT_STRIDE` on the **engine port only**: SGLang derives an internal port block from `--port` (`server_args.py:836,846-861`) and reserves `port_base+0..NUM_DERIVED_PORTS-1`; with `BASE+index` the two legs got 29001/29002 and overlapped 9 of 10 ports. Default 1. (b) the **same-node start gate**: before starting a row whose node already has a started leg, wait for that leg's `/health` (reusing `tools/wait_healthy.py`), because two concurrent 4-GPU RCCL inits on one host race and one dies. Only fires when a node repeats in the topology. Knobs: `SAME_NODE_START_GATE=health\|off`, `ENGINE_START_STAGGER_S`. |
| `tools_agentx_env.py.diff` | `tools/agentx_env.py` | Mirrors `ENGINE_PORT_STRIDE` in `load_topology`, which **independently** re-derives each worker URL from `ENGINE_PORT_BASE + index`. Without this the benchmark client would address the decode leg at the wrong port. Easy to miss — the stride has to be applied in both places. |

`SHA256SUMS.txt` covers every file in the kit.

**Two diffs are present twice**, under both `<name>.diff` and
`tools_<name>.diff`, for `topology.py` and `agentx_env.py`. The bodies are the
same change; they differ only in the `---`/`+++` header paths, because one pair
was generated against the live experiment workspace and the other against this
kit. Use the `tools_*` pair — its paths are self-contained. The other pair is
kept rather than deleted so nothing that was produced during the experiment is
silently dropped.

## Not here

The **image** carries two patches that are baked in, not applied at run time:
the GLM NextN shared-experts fusion fix, and sglang PR #37152 (HiCache ROCm
copy-round widening — open upstream, never merged, its AMD ROCm CI red). Both
were verified present in the image by import, not by grep; see
`../analysis/image_transfer.yihou.md`. PR #37152 is **inert for this run**:
it only does anything when HiCache is on, and Phase A runs with HiCache off.
