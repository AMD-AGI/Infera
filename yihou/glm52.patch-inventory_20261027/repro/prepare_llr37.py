#!/usr/bin/env python3
"""Assemble archived llr37 inputs in a new directory; never start a service."""

import argparse
import hashlib
import ipaddress
import json
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "yihou/glm52.patch-inventory_20261027"
BASE = ROOT / "yihou/glm52.llrepro-pp4-dpa4.packup_20261024"
C104 = ROOT / "yihou/glm52.llrepro-c104-conc.packup_20261027"
GEMM = ROOT / "yihou/glm52.llrepro-k3-fp8-gemm-mem.packup_20261027"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-ip", required=True)
    parser.add_argument("--decode-ip", required=True)
    parser.add_argument("--decode-ssh", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", default="infera-sglang:v0522-fdbafb60")
    parser.add_argument("--prefill-gpus", default="0,1,2,3")
    parser.add_argument("--decode-gpus", default="0,1,2,3")
    parser.add_argument("--rdma-devices", required=True)
    parser.add_argument("--gid-index", type=int, default=1)
    parser.add_argument("--container-prefix", default="yihou-public-llr37")
    args = parser.parse_args()
    output = args.output.resolve()
    model = args.model.resolve()
    if output.exists():
        parser.error("output already exists; choose a new directory (nothing is overwritten)")
    for value in (args.prefill_ip, args.decode_ip):
        if ipaddress.ip_address(value).version != 4:
            parser.error("the archived launcher requires IPv4")
    for value in (args.prefill_gpus, args.decode_gpus):
        ids = value.split(",")
        if len(ids) != 4 or len(set(ids)) != 4 or not all(x.isdigit() for x in ids):
            parser.error("each role requires four distinct GPU IDs")
    # Archived launchers split mount lists and forward SSH commands as shell text.
    # Require simple paths/identifiers instead of silently misquoting them.
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-,:@")
    values = (
        str(output),
        str(model),
        args.decode_ssh,
        args.image,
        args.rdma_devices,
        args.container_prefix,
    )
    if any(not value or any(c not in safe for c in value) for value in values):
        parser.error("paths/identifiers must contain only letters, digits, / . _ - , : @")
    if not args.container_prefix.startswith("yihou-"):
        parser.error("use a dedicated container prefix starting with yihou-")

    kit = output / "kit/sglang_2p1d_glm5.2"
    sources = {}

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[str(target.relative_to(output))] = {
            "source": str(source.relative_to(ROOT)),
            "sha256": digest(source),
        }

    for source in sorted((BASE / "kit/sglang_2p1d_glm5.2").rglob("*")):
        if source.is_file():
            copy(source, kit / source.relative_to(BASE / "kit/sglang_2p1d_glm5.2"))
    for name in ("worker.sh", "common.sh"):
        copy(C104 / "scripts" / name, kit / "engine" / name)
    csvs = (
        "merged/yihou_glm52_dpa_bf16_tuned_gemm.csv",
        "ptpc/yihou_glm52_ptpc_bf16_tuned_gemm.csv",
    )
    for name in csvs:
        copy(GEMM / "aiter_retune" / name, output / "aiter_retune" / name)

    overlay = output / "overlay"
    overlay.mkdir()
    patch = INVENTORY / "patches/K01-kf-decode-kernel-overlay.NEWFILES.diff"
    with patch.open("rb") as stream:
        subprocess.run(
            ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-d", str(overlay)],
            stdin=stream,
            check=True,
        )
    site = (C104 / "sites/llr37.site.env").read_text()
    replacements = {
        "10.245.148.209": args.prefill_ip,
        "10.245.154.168": args.decode_ip,
        "crsuse2-m2m-136": args.decode_ssh,
        "/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4": str(model),
        "infera-sglang:v0522-fdbafb60": args.image,
        "/mnt/m2m_nobackup/yihou_llrepro/work": str(output / "work"),
        "ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7": args.rdma_devices,
        "MC_GID_INDEX=1": f"MC_GID_INDEX={args.gid_index}",
        "PREFILL0_GPUS=4,5,6,7": f"PREFILL0_GPUS={args.prefill_gpus}",
        "PREFILL1_GPUS=4,5,6,7": f"PREFILL1_GPUS={args.prefill_gpus}",
        "DECODE_GPUS=4,5,6,7": f"DECODE_GPUS={args.decode_gpus}",
        "CTR=yihou-llr": f"CTR={args.container_prefix}",
        "/shared_nfs/yihou_glm52/kf_e2e_20261023/kf/e2e/overlay": str(overlay),
        "/shared_nfs/yihou_glm52/llrepro_20261023/prep_next.yihou/aiter_retune": str(
            output / "aiter_retune"
        ),
    }
    # Keep executable values from the archive, replacing only site-specific ones.
    site = "\n".join(line for line in site.splitlines() if not line.startswith("#")) + "\n"
    for old, new in replacements.items():
        if old not in site:
            raise ValueError(f"archived site anchor changed: {old}")
        site = site.replace(old, new)
    site = (
        "# Historical llr37 performance settings: simulated acceptance; known correctness failures.\n"
        "# Read PUBLIC_REPRODUCE.md before launching. This file does not start services.\n" + site
    )
    (output / "site.env").write_text(site)
    # These are an experimental control, not a certified correctness recipe.
    (output / "real-acceptance.env").write_text(
        f"source {shlex.quote(str(output / 'site.env'))}\n"
        "export SIM_ACCEPT_LEN=''\n"
        "export DECODE_CUSTOM_AR=0\n"
        "export PREFILL_ENV_EXTRA='SGLANG_PP_COMM_OVERLAP=0'\n"
        "# Additional fused-kernel/long-context correctness remains unqualified.\n"
    )
    manifest = {
        "purpose": "Historical llr37 source assembly; no image build or GPU validation",
        "archive_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_files": sources,
        "kf_patch": {"source": str(patch.relative_to(ROOT)), "sha256": digest(patch)},
        "assembled_files": {
            str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()
        },
    }
    (output / "ASSEMBLY.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {output}; no service started, no GPU accessed.")
    print("Use the same absolute paths on both nodes. Review site.env before launching.")


if __name__ == "__main__":
    main()
