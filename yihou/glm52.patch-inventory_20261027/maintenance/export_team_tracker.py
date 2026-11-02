#!/usr/bin/env python3
"""Export the public, main-based team tracker from the daily GitHub audit."""

import argparse
import json
import posixpath
import re
from pathlib import Path
from urllib.parse import urlsplit

import update_pr_ledger as ledger

ARCHIVE = "b563462fa4a20b4a6f563d279de14ddee455b343"
INVENTORY = "yihou/glm52.patch-inventory_20261027"
ARCHIVE_ROOT = f"https://github.com/AMD-AGI/Infera/blob/{ARCHIVE}/"
FILES = ("README.md", "ACTIVE_PRS.md", "HISTORY.md", "PACKUP_MAP.md", "REFERENCES.md")


def pinned(text):
    """Resolve archive-relative links without relying on files being on main."""

    def replace(match):
        label, target = match.groups()
        if target.startswith("https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/"):
            target = target.replace("/blob/dev/pd_opt/glm_5.2_agentx/", f"/blob/{ARCHIVE}/", 1)
        elif not urlsplit(target).scheme and not target.startswith("#"):
            target = ARCHIVE_ROOT + posixpath.normpath(posixpath.join(INVENTORY, target))
        return f"[{label}]({target})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, text)


def archive(path, label):
    return f"[{label}]({ARCHIVE_ROOT}{path})"


def generate(inventory, runtime, destination):
    raw = json.loads((runtime / "latest_raw.json").read_text())
    notes = json.loads((inventory / "maintenance/handoff_notes.json").read_text())
    public = ledger.public_only(raw["prs"], raw["visibility"])
    owned = [p for p in public if p["owned"]]
    external = [p for p in public if not p["owned"]]
    active = [p for p in owned if p["state"] == "OPEN"]
    ready = [p for p in active if not p["isDraft"]]
    drafts = [p for p in active if p["isDraft"]]
    checked = raw["checked_at_utc"]
    by_key = {ledger.key(p): p for p in public}
    priority_roles = {
        "sgl-project/sglang#33059": "P0: historical Aiter query-row repair is baked into the image; current-main port needs HaiShaw's model matrix.",
        "sgl-project/sglang#46830": "P0: baked dsa_dp_sync fix; replaces closed #33973. Prioritize CI and review.",
        "sgl-project/sglang#46820": "P0: baked nonblocking metrics fix; CPU/socket regression already exists. Prioritize CI and review.",
        "sgl-project/sglang#45524": "P0, external: related PP producer-event fix. Preserve the distinct historical image patch; coordinate with its author.",
        "sgl-project/sglang#47276": "P1: reworked fused indexer/top-k port; not the exact historical overlay. Coordinate #45307 overlap.",
        "sgl-project/sglang#47278": "P1: reworked sparse MLA port with BF16 split partials; qualify the new combined stack separately.",
        "AMD-AGI/Infera#221": "P1, evidence tooling: publish kernel provenance; not required at runtime. Retarget after #220 and qualify redacted derivatives.",
    }
    priority_rows = []
    for key, role in priority_roles.items():
        p = by_key[key]
        priority_rows.append(f"| {ledger.link(p)} | {ledger.state(p)} | {ledger.cell(role)} |")
    overview = [
        "# GLM-5.2 PD: team PR tracker",
        "",
        "## 1. Merge priorities for the 48k deployment",
        "",
        "Prioritize SGLang **#33059, #46830 and #46820**, and coordinate the existing external **#45524**. They cover fix families already carried by the historical image. Then progress the narrower kernel ports **#47276/#47278** and the evidence catalog **Infera #221**. This is the integration priority for reducing downstream patches; the already-patched frozen image does not need to wait for these merges.",
        "",
        "| Reference | Total tokens/s/GPU | Change / interpretation |",
        "|---|---:|---|",
        "| llr35 / llr35r, C96 | 47,843 / 47,950; mean 47,897 | Earlier baseline |",
        "| llr37 / llr37r / llr37r2, C104 | 48,713 / 48,103 / 47,795; mean **48,204** | Same patch stack, concurrency 96 to 104 |",
        "| llr51, C104 | **48,806**, one run | Decode memory fraction 0.95 to 0.96; not replicated |",
        "",
        "The 47.9k-to-48k changes are existing configuration settings, with **no additional required PR**. The historical topology is eight MI355X GPUs, PP4/TP1 prefill plus TP4/attention-DP4 decode, DCP1. The score counts input plus output tokens, including cache hits, not decode-only generation. Raw measurements and site files: [references R1–R4](REFERENCES.md#performance-and-patch-provenance).",
        "",
        "**Evidence boundary:** serving throughput used simulated MTP acceptance 2.99. Real-acceptance investigation found garbling associated with PP overlap and decode custom all-reduce. These are historical performance results, not a correctness-qualified deployment or individual-patch speedup. Later kernel PRs have separate numerical/real-acceptance evidence; their serving timings also use simulated acceptance.",
        "",
        f"PR state checked: **{checked}**. Ready means non-draft, not approval or passed functional CI.",
        "",
        "| Priority PR | Current state | Deployment role / next action |",
        "|---|---|---|",
        *priority_rows,
        "",
        "### Required inputs without a merge-ready PR",
        "",
        "Preserve **S13** (PP hidden size; llying attribution agreement needed before filing), the **complete K01 overlay**, **both A01 Aiter CSVs**, **I07 launch hooks/site files**, the **S02 workaround** and the inherited image/RDMA patches. The two kernel PRs do not replace the complete overlay. Exact sources and ownership are in [PACKUP_MAP.md](PACKUP_MAP.md).",
        "",
        "DCP transport/cache PRs #46819/#46824/#46826, OffloaderV2 #45484, transfer-queue #42163, prefill_complete #45337 and HIP-transport #5005/#4340 are useful separate work, not extra requirements for this DPA4/RDMA deployment. Late llr42 fused-indexer activation and llr52 blocking-poll experiments were not kept.",
        "",
        f"## 2. Ready for review — {len(ready)} owned public PRs",
        "",
        "These can receive review now. Resolve the listed CI/review/merge-order work; do not add a blanket new GPU gate to an already justified port. Detailed heads, evidence, commands and reviewer requests: [ACTIVE_PRS.md](ACTIVE_PRS.md#ready-for-review).",
        "",
        *ledger.active_table(ready, notes),
        "",
        f"## 3. Draft — {len(drafts)} owned public PRs",
        "",
        "Each row separates author work needed for promotion from maintainer-controlled merge requirements. Reuse matching archived hardware evidence; request a pool node only for concrete uncovered behavior. Human review and attribution cannot be supplied by the updater. Details: [ACTIVE_PRS.md](ACTIVE_PRS.md#draft).",
        "",
        *ledger.active_table(drafts, notes),
        "",
        "## 4. Scope, ownership and references",
        "",
        f"**{ledger.count_text(owned)}** across all public repositories visible to the account-wide author query (`dorado269`). This includes historical and general tooling work. External authors' PRs are separate dependencies and are not counted as ours. Restricted-repository records remain in the internal handoff and are not published here.",
        "",
        "- [Active details and external dependencies](ACTIVE_PRS.md): exact revisions, checks, evidence, last actions, next actors and commands.",
        "- [Complete public PR history](HISTORY.md): merged and closed entries, replacement/duplicate dispositions.",
        "- [Patch-to-pack-up map](PACKUP_MAP.md): required historical inputs, later ports, excluded trials and unfiled work.",
        "- [References and maintenance](REFERENCES.md): immutable archive links, source pins, daily refresh and handoff protocol.",
        "",
        "Task owner is **yihou until a named delegate accepts the row**. Record the delegate, tested SHA, run IDs, review responses and next action in the curated notes; do not rely on a requested reviewer having accepted ownership. A changed head or new review activity requires triage. No GPU testing, PR promotion or merging is performed by the daily status collector.",
        "",
    ]
    full = ledger.render(
        owned, external, ledger.public_only(raw["issues"], raw["visibility"]), notes, checked
    )
    details = full.split("## Handoff details\n", 1)[1].split("## External dependencies", 1)[0]
    blocks = re.split(r"(?=^### \[)", details, flags=re.M)
    blocks = [b for b in blocks if b.startswith("### [")]
    pr_blocks = {}
    for block in blocks:
        url = re.search(r"^### \[[^\]]+\]\(([^)]+)\)", block).group(1)
        pr_blocks[url] = block
    tail = (
        "## External dependencies"
        + full.split("## External dependencies", 1)[1].split("## Complete authored history", 1)[0]
    )
    detail_doc = [
        "# Active PR handoff",
        "",
        f"GitHub refresh: **{checked}**. [Overview and priorities](README.md).",
        "",
        "## Ready for review",
        "",
    ]
    detail_doc += [
        pinned(pr_blocks[p["url"]]) for p in sorted(ready, key=lambda p: ledger.order(p, notes))
    ]
    detail_doc += ["## Draft", ""]
    detail_doc += [
        pinned(pr_blocks[p["url"]]) for p in sorted(drafts, key=lambda p: ledger.order(p, notes))
    ]
    detail_doc += [pinned(tail)]
    history = (
        "# Complete public authored PR history\n\n"
        + f"Checked: **{checked}**. **{ledger.count_text(owned)}**. [Active handoff](README.md).\n\n"
        + full.split("## Complete authored history\n", 1)[1]
    )
    deployment = (inventory / "LLR37_DEPLOYMENT_PRIORITY.md").read_text()
    required = deployment.split("## Required inputs that are not open PRs", 1)[1].split(
        "## Other open work", 1
    )[0]
    late = deployment.split("## Changes after the 47.9k baseline: no additional required PR", 1)[
        1
    ].split("## Required inputs", 1)[0]
    map_doc = [
        "# PR, patch and experiment correspondence",
        "",
        f"PR states checked: **{checked}**. [Live work queues](README.md). The original archive is pinned at `{ARCHIVE}`; a run's source is not automatically identical to its later main-based PR.",
        "",
        "## Owned GLM-5.2 core patches",
        "",
        "| PR / state | Purpose | Role in historical deployment | Evidence / pack-up |",
        "|---|---|---|---|",
    ]
    core = {
        "sgl-project/sglang#" + str(n)
        for n in [33059, 45484, 46819, 46820, 46822, 46824, 46826, 46828, 46830, 47276, 47278]
    } | {"AMD-AGI/Infera#218", "kvcache-ai/Mooncake#5005"}
    for p in sorted(
        [p for p in owned if ledger.key(p) in core], key=lambda p: ledger.order(p, notes)
    ):
        n = ledger.note_for(p, notes)
        map_doc.append(
            "| "
            + " | ".join(
                map(
                    ledger.cell,
                    [
                        ledger.link(p) + " / " + ledger.state(p),
                        n["purpose"],
                        n["deployment"],
                        pinned(n["evidence"]),
                    ],
                )
            )
            + " |"
        )
    map_doc += [
        "",
        "## Required artifacts without a merge-ready PR",
        pinned(required),
        "## Late 48k trials and configuration-only changes",
        pinned(late),
        "## Other dispositions",
        "",
        "S02/S12 remain issue-only; S04/S08/S09/S14 are excluded; S13 needs attribution agreement; A01 and the remaining original K01 overlay are not qualified for wholesale filing. S03 layer parallelism had no demonstrated gain; M03 was unbuilt. Other authors' equivalent PRs receive evidence instead of duplicates. See the "
        + archive(INVENTORY + "/PR_PACKUP_MAP.md", "complete archived disposition map")
        + " for source diffs and original pack-ups.",
        "",
    ]
    refs = [
        "# References and maintenance",
        "",
        "## Performance and patch provenance",
        "",
        f"All archive links below pin commit `{ARCHIVE}` in the public Infera repository. They remain valid when this documentation branch is rebased onto main, which does not contain the experiment pack-ups. PR URLs intentionally point to live upstream discussions.",
        "",
        "| Reference | Source | What it establishes |",
        "|---|---|---|",
    ]
    records = [
        (
            "R1",
            "glm52.llrepro-c104-conc.packup_20261027/README.md",
            "C104 archive",
            "llr37 48,713; concurrency-only change and correctness caveat",
        ),
        (
            "R2",
            "glm52.llrepro-c104-tail.packup_20261028/README.md",
            "Final C104 archive",
            "Three-run mean 48,204; llr51 48,806 unreplicated; llr50/52 not carried",
        ),
        (
            "R3",
            "glm52.llrepro-c104-conc.packup_20261027/results/llr/llr37-c104/agentx_conc104.json",
            "Raw llr37 result",
            "Input/output accounting and reported performance metrics",
        ),
        (
            "R4",
            "glm52.llrepro-c104-tail.packup_20261028/sites/llr51.site.env",
            "llr51 site",
            "Only decode memory fraction changes from the llr37 site",
        ),
        (
            "R5",
            "glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/README.md",
            "K3, FP8 draft and GEMM pack-up",
            "47.9k baseline, both Aiter CSVs and per-trial outcomes",
        ),
        (
            "R6",
            "glm52.llrepro-pp4-dpa4.packup_20261024/README.md",
            "Base PP4/DPA4 kit",
            "Frozen image and S13 model source",
        ),
        (
            "R7",
            "glm52.llrepro-kf-ar-topk.packup_20261024/README.md",
            "KF kernel investigation",
            "Historical overlay family; not proof of final #47276/#47278 identity",
        ),
        (
            "R8",
            "glm52.llrepro-garblefree-sweep.packup_20261028/README.md",
            "Overlap/custom-AR sweep",
            "Real-acceptance failure context, llr42 alias and rejected trials",
        ),
        (
            "R9",
            "glm52.dcp4-pd.packup_20261022/README.md",
            "DCP PD experiments",
            "pdBe-r1, R16–R25 and separate DCP patch evidence",
        ),
        (
            "R10",
            "glm52.dcpx-pack-fix-best.packup_20261023/README.md",
            "DCP gather experiments",
            "dcpx8c/9/10 and the metadata/producer-wait evidence boundary",
        ),
        (
            "R11",
            "glm52.v0521-offloaderv2.packup_20261015/README.md",
            "OffloaderV2 experiments",
            "ofs0c/ofg3c/ofg2c; distinct from HiCache and the 48k stack",
        ),
        (
            "R12",
            "glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md",
            "c44 transfer experiments",
            "Metrics, HIP IPC/priority and rejected transfer trials",
        ),
    ]
    refs += [
        f"| {rid} | {archive('yihou/' + path, label)} | {why} |"
        for rid, path, label, why in records
    ]
    refs += [
        "",
        "## Reproduction inputs and unresolved identities",
        "",
        "- "
        + archive(INVENTORY + "/PUBLIC_REPRODUCE.md", "Public reproduction procedure")
        + ": image build, artifact reconstruction, topology and benchmark commands.",
        "- "
        + archive(INVENTORY + "/repro/SOURCE_PINS.json", "Source and image pins")
        + "; "
        + archive(INVENTORY + "/repro/SHA256SUMS", "artifact checksums")
        + "; "
        + archive(INVENTORY + "/repro/prepare_llr37.py", "source-only kit reconstruction")
        + ".",
        "- Frozen Infera source: [fdbafb6065](https://github.com/AMD-AGI/Infera/tree/fdbafb6065dae6d1c53936c7d2cf1f5022a514d0); Mooncake [faae8dd4a6](https://github.com/kvcache-ai/Mooncake/tree/faae8dd4a6309c3ecd47e0721a83b0250d686fa2) plus inherited RDMA patches.",
        "- Official SGLang base: `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`; exact registry digest is in SOURCE_PINS.json. The historical integrated image name is a local tag, not a public pull URL. Rebuild from the pinned public sources.",
        "- Public model/trace revisions and retrieval references are in the reproduction guide. Their byte identity to historical local weights/trace files is unverified; record exact hashes in a new run. Current-main kernel ports have separate evidence and do not certify this frozen overlay.",
        "",
        "## Daily refresh and team handoff",
        "",
        "Owner: yihou until a named delegate accepts a row. The configured maintenance host runs daily at **09:00 UTC / 17:00 Asia/Shanghai**. It discovers new authored PRs, checks repository visibility, refreshes CI/review/heads and republishes these Markdown files to the open tracking PR branch. A timestamp older than 36 hours needs operator investigation. The host and authenticated GitHub/SSH access must remain available.",
        "",
        "The public account ledger includes historical work outside the GLM-5.2 core set. Internal repository records are kept in the restricted handoff. Requested reviewers are not assumed assignees. Ready means available for review, not approved; skipped/draft/permission gates do not establish functional-test coverage.",
        "",
        "For each substantive change, record the named owner, last action, next actor, exact tested head, commands, run IDs, unresolved review and dependency order. Update the curated notes before refreshing; otherwise generated text would overwrite an unrecorded manual table edit. New PRs and changed heads/activity are flagged for triage. Keep evidence qualification separate from GitHub state.",
        "",
        "The maintained source is "
        + f"[handoff_notes.json](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/{INVENTORY}/maintenance/handoff_notes.json)"
        + "; "
        + f"[maintenance tools](https://github.com/AMD-AGI/Infera/tree/dev/pd_opt/glm_5.2_agentx/{INVENTORY}/maintenance)"
        + " regenerate the public export. Machine-specific cron/config/logs stay outside this PR. The experiment workspace and uncommitted pack-ups are not staged by the publisher.",
        "",
        "Daily publication updates only the open documentation branch; it does not promote, merge or edit runtime PRs. When the tracking PR merges or closes, the publisher preserves its artifacts and flags that a new main-based update PR is needed; create that follow-up before resuming publication to the team tracker. The underlying account status collector continues its daily refresh. Never push generated changes directly to main.",
        "",
        "Validation work uses isolated checkouts. Keep `git commit -s` under the configured identity; no assistant/co-author trailers. Allocate GPU checks through the machine pool; do not use the live experiment nodes. Matching archived hardware evidence is reusable, while changed implementations and missing model matrices need specific additional coverage.",
        "",
    ]
    contents = {
        "README.md": "\n".join(overview),
        "ACTIVE_PRS.md": "\n".join(detail_doc),
        "HISTORY.md": history,
        "PACKUP_MAP.md": "\n".join(map_doc),
        "REFERENCES.md": "\n".join(refs),
    }
    for filename, content in contents.items():
        # Keep active navigation local; archive links retain immutable provenance.
        content = content.replace(
            ARCHIVE_ROOT + INVENTORY + "/LLR37_DEPLOYMENT_PRIORITY.md", "README.md"
        )
        content = (
            content.replace(ARCHIVE_ROOT + INVENTORY + "/PR_PACKUP_MAP.md", "PACKUP_MAP.md")
            if filename == "ACTIVE_PRS.md"
            else content
        )
        for repo, visibility in raw["visibility"].items():
            if visibility != "public" and repo.lower() in content.lower():
                raise RuntimeError(f"Restricted repository identifier found in {filename}")
        ledger.write_text(destination / filename, content.rstrip() + "\n")
    return {
        "checked_at_utc": checked,
        "public_prs": len(owned),
        "ready": len(ready),
        "draft": len(drafts),
        "files": list(FILES),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if "yihou" not in str(args.destination.resolve()):
        raise SystemExit("Destination must be under the yihou task worktree")
    print(json.dumps(generate(args.inventory, args.runtime, args.destination), indent=2))


if __name__ == "__main__":
    main()
