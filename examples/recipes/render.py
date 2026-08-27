#!/usr/bin/env python3
"""Render a recipe manifest for your cluster.

Recipes ship `<PLACEHOLDER>` strings instead of defaults, because every value they
stand for is a property of your cluster that a default would get wrong. `sed` does
substitute them, and for the aggregated combos that is all this script does. It
exists for the three things `sed` cannot do:

  1. Refuse to emit a manifest that still has a placeholder in it. `kubectl`
     rejects a literal `<NODE>` outright, but it accepts
     `--disaggregation-ib-device <RDMA_IB_DEVICES>`, and that one only fails once
     the engine opens the device -- minutes later, after the weights have loaded,
     in a message about a device rather than about a placeholder.
  2. Refuse an unknown placeholder name. `sed -e "s|<MODELDIR>|...|"` matches
     nothing and exits 0, so a typo reads as success.
  3. Pin Mooncake to a single rail (`--pin-rail`), which is an insertion into the
     env list rather than a substitution. Needed only on fabrics whose rails carry
     no IPv4 — see the recipe README, or `--help` on that flag.

Usage:
  render.py glm5.2-fp8-gfx942/aggregated --set NODE=node-a --set MODEL_DIR=/mnt/models
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

RECIPES_DIR = Path(__file__).resolve().parent

# Digits are part of a placeholder name: <KVD_L3_DIR> has one, and a character
# class of A-Z and _ alone silently skips it -- which would defeat the whole
# point of the leftover check on exactly the combos that need it most.
PLACEHOLDER_RE = r"<([A-Z][A-Z0-9_]*)>"

# Every placeholder the recipes use. Rendering fails on a name outside this set so
# a misspelled --set cannot pass silently, and fails on a member left unset so a
# forgotten one cannot reach the cluster.
PLACEHOLDERS = (
    "NODE",
    "PREFILL_NODE",
    "DECODE_NODE",
    "MODEL_DIR",
    "PREFILL_MODEL_DIR",
    "DECODE_MODEL_DIR",
    "KVD_L3_DIR",
    "RDMA_IB_DEVICES",
    "PREFILL_GID_INDEX",
    "DECODE_GID_INDEX",
)


def parse_set(pairs: list[str]) -> dict[str, str]:
    values = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            sys.exit(f"--set wants NAME=VALUE, got {pair!r}")
        if name not in PLACEHOLDERS:
            sys.exit(
                f"--set {name}: not a placeholder any recipe uses. Known names:\n  "
                + "\n  ".join(PLACEHOLDERS)
            )
        values[name] = value
    return values


def pin_rail(text: str, rail: str) -> str:
    """Add the single-rail pin next to every MC_GID_INDEX entry.

    Both legs get it, at the indentation of the entry it follows. The aggregated
    combos have no MC_GID_INDEX because they move no KV off the node, so pinning
    them is a no-op and is reported as an error rather than done quietly.
    """
    if "," in rail:
        sys.exit(
            f"--pin-rail wants one rail, got {rail!r}. Pinning exists because "
            "Mooncake cannot tell link-local rails apart; naming several defeats it."
        )

    def inject(m: re.Match) -> str:
        indent = m.group(1)
        return (
            m.group(0)
            + f'\n{indent}- {{name: MC_MS_AUTO_DISC, value: "0"}}'
            + f'\n{indent}- {{name: MC_MS_FILTERS, value: "{rail}"}}'
        )

    text, n = re.subn(r"^([ \t]*)- \{name: MC_GID_INDEX[^}]*\}$", inject, text, flags=re.MULTILINE)
    if n == 0:
        sys.exit(
            "--pin-rail: this combo has no MC_GID_INDEX to pin. The aggregated "
            "combos carry no rail, no GID index and no Mooncake config at all."
        )
    return text


def check_rail(rail: str, nodes: list[str]) -> None:
    """Refuse to render against a rail that is not ACTIVE on every target node.

    A rail that is down does not produce a transport error. Mooncake logs three
    info-level lines about skipping the device, comes up with no usable rail, and
    the engine dies in PD warmup with `Memory access fault by GPU node-N` -- an
    expensive way to learn a cable is out. Reads sysfs over ssh, so it needs ssh
    to each node; skip the flag if that is not available and check by hand.
    """
    path = f"/sys/class/infiniband/{rail}/ports/1/state"
    dead = []
    for node in sorted(set(nodes)):
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", node, f"cat {path}"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except subprocess.SubprocessError as e:
            dead.append(f"{node}: could not read {path} ({e})")
            continue
        if "ACTIVE" not in r.stdout:
            dead.append(f"{node}: {rail} state={r.stdout.strip() or r.stderr.strip() or '?'}")
    if dead:
        sys.exit(
            f"rail {rail} is not usable:\n  "
            + "\n  ".join(dead)
            + "\n\nList the candidates on each node:\n"
            "  for d in /sys/class/infiniband/*; do "
            'echo "$(basename $d) $(cat $d/ports/1/state)"; done\n'
            "Then confirm the one you pick carries traffic between the nodes:\n"
            "  ib_write_bw -d <rail> -x <gid>          # on one node\n"
            "  ib_write_bw -d <rail> -x <gid> <peer>   # on the other"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render a recipe manifest for your cluster.",
        epilog="example: render.py glm5.2-fp8-gfx942/disaggregated "
        "--set PREFILL_NODE=node-a --set DECODE_NODE=node-b "
        "--set MODEL_DIR=/mnt/models --set RDMA_IB_DEVICES=rdma0 "
        "--set PREFILL_GID_INDEX=3 --set DECODE_GID_INDEX=3 -o pd.yaml",
    )
    p.add_argument(
        "combo",
        help="recipe and combo, e.g. glm5.2-fp8-gfx942/disaggregated; or a path to a deploy.yaml",
    )
    p.add_argument(
        "--set",
        metavar="NAME=VALUE",
        action="append",
        default=[],
        dest="sets",
        help="substitute one placeholder; repeatable. Read these off the nodes "
        "rather than copying them from any document -- the GID index in "
        "particular is per node, not per cluster.",
    )
    p.add_argument(
        "--image",
        help="override the engine image, for a registry the cluster pulls from "
        "rather than the recipe's local tag",
    )
    p.add_argument(
        "--pin-rail",
        action="store_true",
        help="pin Mooncake to the single rail named by RDMA_IB_DEVICES. Needed "
        "when your rails carry no IPv4: every GID is then link-local fe80::, so "
        "every rail looks like the same /64 subnet, and Mooncake can pair the "
        "prefill node's rail A with the decode node's rail B, where the transfer "
        "times out. Harmless to omit on fabrics whose rails carry IPv4.",
    )
    p.add_argument(
        "--check-rail",
        action="store_true",
        help="ssh to each target node and refuse to render unless the rail is "
        "ACTIVE there. Two seconds of sysfs against ~11 minutes of weight "
        "loading followed by a GPU memory fault.",
    )
    p.add_argument("-o", "--out", default="-", help="output file (default: stdout)")
    args = p.parse_args()

    src = Path(args.combo)
    if not src.is_file():
        src = RECIPES_DIR / args.combo / "deploy.yaml"
    if not src.is_file():
        combos = sorted(
            str(q.parent.relative_to(RECIPES_DIR)) for q in RECIPES_DIR.glob("*/*/deploy.yaml")
        )
        sys.exit(f"no manifest at {src}. Available:\n  " + "\n  ".join(combos))

    text = src.read_text(encoding="utf-8")
    values = parse_set(args.sets)

    needed = sorted(set(re.findall(PLACEHOLDER_RE, text)))
    missing = [ph for ph in needed if ph not in values]
    if missing:
        sys.exit(f"{src}: no value for " + ", ".join(missing) + "\nPass each as --set NAME=VALUE.")
    unused = sorted(set(values) - set(needed))
    if unused:
        sys.exit(f"{src}: does not use " + ", ".join(unused) + " -- wrong combo?")

    if args.pin_rail and "RDMA_IB_DEVICES" not in values:
        sys.exit(
            f"--pin-rail: {src} takes no RDMA_IB_DEVICES. The aggregated combos "
            "move no KV off the node, so they carry no rail to pin."
        )

    if args.check_rail:
        if not args.pin_rail:
            sys.exit("--check-rail checks the rail that --pin-rail pins; pass both")
        nodes = [values[k] for k in ("NODE", "PREFILL_NODE", "DECODE_NODE") if k in values]
        check_rail(values["RDMA_IB_DEVICES"], nodes)

    if args.pin_rail:
        text = pin_rail(text, values["RDMA_IB_DEVICES"])

    for name, value in values.items():
        text = text.replace(f"<{name}>", value)

    if args.image:
        text, n = re.subn(r"^(\s*image:\s*)\S+$", rf"\g<1>{args.image}", text, flags=re.MULTILINE)
        if n == 0:
            sys.exit(f"--image: no image field found in {src}")

    left = sorted(set(re.findall(PLACEHOLDER_RE, text)))
    if left:
        sys.exit(
            f"{src}: unsubstituted placeholders remain: " + ", ".join(f"<{name}>" for name in left)
        )

    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out} from {src}", file=sys.stderr)


if __name__ == "__main__":
    main()
