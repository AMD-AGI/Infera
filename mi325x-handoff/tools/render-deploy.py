#!/usr/bin/env python3
"""Render any glm5.2-fp8-gfx942 combo for the tw041/tw044 cluster.

The recipe ships placeholders. On this cluster two of the substitutions cannot be
expressed as a plain `sed`, so they live here instead:

  1. GID index 1, not 3. The eight 400G rails carry no IPv4, so their GID tables
     hold only fe80:: entries (0 = RoCEv1, 1 = RoCEv2). Index 3 exists only on the
     200G management NIC, which is the one the preflight tool happened to read.
  2. Mooncake pinned to a single rail. Cross-rail RDMA does not connect here (each
     rail is its own L2 segment), and with link-local-only GIDs every rail looks
     like the same fe80::/64 subnet, so subnet-based rail pairing can pair prefill
     rail A with decode rail B and the KV transfer just times out.

Both apply to the disaggregated combos only. The aggregated ones move no KV off
the node, so they carry no rail, no GID index and no Mooncake env at all — which
is also why they are the combo to reach for when you do not yet know whether a
problem is in the engine or in the fabric.

Usage:  render-deploy.py --combo aggregated -o out.yaml
"""

import argparse
import re
import socket
import subprocess
import sys

COMBOS = ("aggregated", "aggregated-kvd", "disaggregated", "disaggregated-kvd")

# Mooncake KV-registration Mode A (bare ibv_reg_mr + peer-mem), as reported viable
# and best-ranked by `python -m infera.tools.preflight.mooncake_mode` on both nodes.
# ib_peer_mem is loaded, so nothing is pinned and no dma-buf image rebuild is needed.
MODE_A_ENV = [
    ("MOONCAKE_DISABLE_HIP_DMABUF", "1"),
    ("MC_DISABLE_HIP_TRANSPORT", "1"),
    ("NCCL_IB_DISABLE", "1"),
]


def drop_mtp(text: str) -> tuple[str, int]:
    """Strip the four EAGLE/MTP flags from every engine command, and rename the CR.

    An experiment, not a shape: the point is to test whether MTP is what makes
    hicache deadlock a mixed worker, which is the one lever the aggregated-kvd
    banner lists as untried. It renames the deployment so the result cannot be
    confused with the shipped combo's, and it deliberately leaves
    --hicache-io-backend direct in place — that flag exists BECAUSE of MTP's draft
    pool, so removing both at once would confound two variables.
    """
    # The flags sit two-per-line in the manifests, so drop whole lines rather
    # than trying to excise argv entries.
    lines, kept, n = text.split("\n"), [], 0
    for line in lines:
        if '"--speculative-' in line:
            n += 1
            continue
        kept.append(line)
    out = "\n".join(kept)
    out = out.replace("glm52-fp8-mixed-kvd", "glm52-fp8-mixed-kvd-nomtp")
    return out, n


def set_router_backend(text: str, backend: str) -> tuple[str, int]:
    """Flip `--router-backend` on the server command.

    The recipe pins `python` explicitly, so this is a value swap rather than an
    insertion. Kept as an experiment flag because the four combos' §7 numbers were
    all measured on `python` — switching the router is a separate before/after,
    not a free substitution.
    """
    return re.subn(
        r'("--router-backend",")(python|rust)(")',
        lambda m: m.group(1) + backend + m.group(3),
        text,
    )


def scale_worker(text: str, replicas: int) -> str:
    """Set worker replicas and let the scheduler spread them.

    Only the WORKER's nodeSelector is dropped — the server stays pinned. No
    anti-affinity rule is needed: each replica asks for `amd.com/gpu: 8` and every
    node here has exactly 8, so two replicas physically cannot co-locate.

    This exists for the kv-aware check. At `replicas: 1` a routing policy has
    nothing to choose between, so "kv-aware is working" and "kv-aware silently
    degraded to load balancing" are indistinguishable — which is the exact failure
    mode the rust ZMQ decoder once had.
    """
    text, n = re.subn(
        r"(\n      role: mixed\n      replicas: )\d+",
        lambda m: m.group(1) + str(replicas),
        text,
    )
    if n != 1:
        sys.exit(f"--worker-replicas: expected 1 replicas line, rewrote {n}")

    # Drop the WORKER's nodeSelector, leaving the server's intact. Anchored on the
    # comment that follows only the worker's — the node name is already
    # substituted by this point, so it cannot be matched on.
    text, n = re.subn(
        r"^[ \t]*nodeSelector: \{kubernetes\.io/hostname: [^}]*\}\n(?=[ \t]*# No hostNetwork here)",
        "        # nodeSelector dropped: replicas > 1 must land on different nodes\n",
        text,
        flags=re.MULTILINE,
    )
    if n != 1:
        sys.exit(f"--worker-replicas: expected 1 worker nodeSelector to drop, dropped {n}")
    return text


def render(src: str, combo: str, args: argparse.Namespace) -> str:
    text = src
    for ph, val in [
        ("<NODE>", args.node),
        ("<PREFILL_NODE>", args.prefill_node),
        ("<DECODE_NODE>", args.decode_node),
        ("<MODEL_DIR>", args.model_dir),
        ("<KVD_L3_DIR>", args.kvd_l3_dir),
        ("<RDMA_IB_DEVICES>", args.rail),
        ("<PREFILL_GID_INDEX>", args.gid_index),
        ("<DECODE_GID_INDEX>", args.gid_index),
    ]:
        text = text.replace(ph, val)

    text = text.replace("image: infera:sglang-gfx942-glm52", f"image: {args.image}")

    # Append the single-rail pin + Mode A env next to each MC_GID_INDEX entry the
    # recipe already has, preserving its indentation. Disaggregated combos only:
    # the aggregated ones have no MC_GID_INDEX because they have no RDMA.
    want = 2 if combo.startswith("disaggregated") else 0

    def inject(m: re.Match) -> str:
        indent, line = m.group(1), m.group(0)
        extra = [f'{indent}- {{name: MC_MS_AUTO_DISC, value: "0"}}',
                 f'{indent}- {{name: MC_MS_FILTERS, value: "{args.rail}"}}']
        extra += [f'{indent}- {{name: {k}, value: "{v}"}}' for k, v in MODE_A_ENV]
        return line + "\n" + "\n".join(extra)

    text, n = re.subn(
        r"^([ \t]*)- \{name: MC_GID_INDEX[^}]*\}$", inject, text, flags=re.MULTILINE
    )
    if n != want:
        sys.exit(f"{combo}: expected {want} MC_GID_INDEX entries, patched {n}")

    if args.drop_mtp:
        text, n = drop_mtp(text)
        if n == 0:
            sys.exit(f"{combo}: --drop-mtp found no speculative flags to remove")

    if args.router_backend:
        text, n = set_router_backend(text, args.router_backend)
        if n != 1:
            sys.exit(f"{combo}: expected 1 --router-backend to rewrite, found {n}")

    if args.worker_replicas != 1:
        if combo != "aggregated":
            sys.exit("--worker-replicas is only wired for the aggregated combo")
        before = text
        text = scale_worker(text, args.worker_replicas)
        if text == before:
            sys.exit(f"{combo}: --worker-replicas matched nothing to rewrite")

    if args.name_suffix:
        text = text.replace(
            "\n  name: glm52-fp8-mixed\n", f"\n  name: glm52-fp8-mixed-{args.name_suffix}\n"
        )

    left = sorted(set(re.findall(r"<[A-Z_]+>", text)))
    if left:
        sys.exit(f"{combo}: unsubstituted placeholders remain: {left}")
    return text


def check_rail(args) -> None:
    """Refuse to render a manifest pinned to a rail that is not up on every node.

    Both legs are pinned to one rail because cross-rail RDMA does not connect on
    these hosts, which means a rail that is down on one node takes the whole
    deployment with it -- and does so in a way that looks like a GPU bug rather
    than a fabric one. Two seconds of sysfs here replaces eleven minutes of weight
    loading followed by a memory access fault.
    """
    nodes = ([args.node] if args.combo.startswith("aggregated")
             else [args.prefill_node, args.decode_node])
    local = socket.gethostname()
    dead = []
    for node in sorted(set(nodes)):
        path = f"/sys/class/infiniband/{args.rail}/ports/1/state"
        cmd = (["cat", path] if node == local
               else ["ssh", "-o", "ConnectTimeout=8", node, f"cat {path}"])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except subprocess.SubprocessError as e:
            dead.append(f"{node}: could not read state ({e})")
            continue
        state = r.stdout.strip()
        if "ACTIVE" not in state:
            dead.append(f"{node}: {args.rail} state={state or r.stderr.strip() or '?'}")
    if dead:
        sys.exit(
            f"rail {args.rail} is not usable:\n  " + "\n  ".join(dead)
            + "\n\nPick a rail that is ACTIVE on every node above and pass --rail. "
              "To list candidates:\n"
              "  for d in /sys/class/infiniband/*; do "
              "echo \"$(basename $d) $(cat $d/ports/1/state)\"; done\n"
              "Then confirm it carries traffic between the two nodes:\n"
              f"  ib_write_bw -d <rail> -x {args.gid_index}          # on one node\n"
              f"  ib_write_bw -d <rail> -x {args.gid_index} <peer>   # on the other\n"
              "Use --skip-rail-check to render anyway."
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--combo", required=True, choices=COMBOS)
    p.add_argument("--src-root", default="examples/recipes/glm5.2-fp8-gfx942")
    # Single-node combos land on tw041: it is the control-plane node but carries no
    # taint, and putting them here leaves tw044 free.
    p.add_argument("--node", default="tw041")
    p.add_argument("--prefill-node", default="tw044")
    p.add_argument("--decode-node", default="tw041")
    p.add_argument("--model-dir", default="/tmp/infera-models")
    p.add_argument("--kvd-l3-dir", default="/tmp/infera-kvd-l3")
    # rocep246s0 was the default until 2026-08-24, when it went down on tw041
    # (phys_state "3: Disabled") while staying ACTIVE on tw044. rocep147s0 is up on
    # both and measures 28 GB/s cross-node. See --skip-rail-check for why this is
    # worth verifying on every render rather than trusting.
    p.add_argument("--rail", default="rocep147s0")
    p.add_argument("--gid-index", default="1")
    p.add_argument(
        "--skip-rail-check",
        action="store_true",
        help="do not verify the rail is ACTIVE on every node this render targets. "
             "The check exists because a rail that is down does not produce a "
             "transport error: Mooncake logs three info-level lines about skipping "
             "the device, comes up with no usable rail, and the engine then dies "
             "~11 minutes later in PD warmup with `Memory access fault by GPU "
             "node-N`. That is a very expensive way to learn a cable is out.",
    )
    p.add_argument(
        "--image",
        default="localhost:5010/infera:sglang-gfx942-glm52-p01-rocm700",
        help="rocm700 base: these hosts run amdgpu 6.3.x, which does not support "
             "the rocm720 default (Dockerfile PRECONDITION)",
    )
    p.add_argument(
        "--drop-mtp",
        action="store_true",
        help="experiment: strip the EAGLE/MTP flags and rename the CR to "
             "*-nomtp. Tests the one lever the aggregated-kvd banner lists as "
             "untried; not a shape anyone should ship.",
    )
    p.add_argument(
        "--router-backend",
        choices=("python", "rust"),
        help="experiment: flip the router data plane. The recipe ships python; "
             "rust is inside the supported subset here (kubernetes discovery, "
             "kv-aware, http, zmq) and the operator injects the label selector "
             "it needs, but the shipped numbers were taken on python.",
    )
    p.add_argument(
        "--worker-replicas",
        type=int,
        default=1,
        help="experiment: scale the mixed worker out and let the scheduler spread "
             "it. Needed to test kv-aware at all — at 1 replica the policy has "
             "nothing to choose between.",
    )
    p.add_argument(
        "--name-suffix",
        default="",
        help="append to the CR name so an experiment cannot be confused with the "
             "shipped combo's results.",
    )
    p.add_argument("-o", "--out", default="-")
    args = p.parse_args()

    if not args.skip_rail_check:
        check_rail(args)

    path = f"{args.src_root}/{args.combo}/deploy.yaml"
    with open(path, encoding="utf-8") as f:
        out = render(f.read(), args.combo, args)

    if args.out == "-":
        sys.stdout.write(out)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"wrote {args.out} from {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
