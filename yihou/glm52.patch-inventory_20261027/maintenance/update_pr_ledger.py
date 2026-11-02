#!/usr/bin/env python3
"""Discover authored PRs, refresh GitHub facts, and render the handoff ledgers.

Only explicitly public repositories enter tracked outputs. Raw responses and the
account-wide ledger belong in the ignored runtime directory. No PR is mutated.
"""

import argparse
import json
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

TARGETS = {"AMD-AGI/Infera", "sgl-project/sglang", "kvcache-ai/Mooncake", "ROCm/aiter"}
FIELDS = (
    "author,number,url,title,state,isDraft,body,updatedAt,mergedAt,closedAt,"
    "headRefOid,baseRefOid,headRefName,baseRefName,mergeable,mergeStateStatus,"
    "reviewDecision,labels,reviewRequests,assignees,reviews,comments,statusCheckRollup"
)
BEGIN = "<!-- DAILY-PR-STATUS:BEGIN -->"
END = "<!-- DAILY-PR-STATUS:END -->"


def gh(*args):
    for attempt in range(4):
        result = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=180)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if isinstance(data, dict) and data.get("errors"):
                raise RuntimeError("GitHub GraphQL returned errors")
            return data
        if attempt < 3:
            time.sleep(2**attempt)
    raise RuntimeError(f"GitHub query failed: {result.stderr.strip()}")


def write_json(path, data):
    write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_text(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # All destinations are under the checked task/runtime directory.
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(data)
    temporary.replace(path)


def key(pr):
    return f"{pr['repo']}#{pr['number']}"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def state(pr):
    if pr["state"].upper() == "OPEN":
        return "DRAFT" if pr["isDraft"] else "READY"
    return pr["state"].upper()


def public_only(prs, visibility):
    return [p for p in prs if visibility.get(p["repo"]) == "public"]


def check_counts(pr):
    counts = Counter()
    for c in pr.get("statusCheckRollup") or []:
        outcome = c.get("conclusion") or c.get("state") or c.get("status") or "UNKNOWN"
        counts[outcome] += 1
    return dict(sorted(counts.items()))


def checks_text(pr):
    counts = check_counts(pr)
    return ", ".join(f"{name} {count}" for name, count in counts.items()) or "No checks reported"


def review_threads(pr):
    owner, name = pr["repo"].split("/")
    query = """query($owner:String!, $name:String!, $n:Int!, $cursor:String) {
      repository(owner:$owner,name:$name) { pullRequest(number:$n) {
        reviewThreads(first:100,after:$cursor) {
          nodes { isResolved isOutdated path comments(first:1) {
            nodes { url author { login } body createdAt }
          }} pageInfo { hasNextPage endCursor }
        }
      }}
    }"""
    threads, cursor = [], None
    while True:
        args = [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-f",
            f"owner={owner}",
            "-f",
            f"name={name}",
            "-F",
            f"n={pr['number']}",
        ]
        if cursor:
            args += ["-f", f"cursor={cursor}"]
        data = gh(*args)["data"]["repository"]["pullRequest"]["reviewThreads"]
        threads.extend(data["nodes"])
        if not data["pageInfo"]["hasNextPage"]:
            return threads
        cursor = data["pageInfo"]["endCursor"]


def discover(author):
    # Search must be complete; never silently publish a truncated account list.
    meta = gh(
        "api",
        "--method",
        "GET",
        "search/issues",
        "-f",
        f"q=is:pr author:{author}",
        "-f",
        "per_page=1",
    )
    if meta.get("incomplete_results") or meta["total_count"] >= 1000:
        raise RuntimeError("Author search is incomplete; partition the query before publishing")
    found = gh(
        "search",
        "prs",
        "--author",
        author,
        "--limit",
        "1000",
        "--json",
        "number,url,title,state,isDraft,repository,updatedAt",
    )
    if len(found) != meta["total_count"]:
        raise RuntimeError(
            "Search count changed during discovery; rerun instead of publishing a partial list"
        )
    return found


def collect(author, notes, runtime):
    found = discover(author)
    seeds = {
        p["url"]: {"repo": p["repository"]["nameWithOwner"], "number": p["number"], "url": p["url"]}
        for p in found
    }
    for url in notes.get("external_prs", []):
        parts = url.split("/")
        seeds.setdefault(url, {"repo": "/".join(parts[3:5]), "number": int(parts[-1]), "url": url})
    repositories = sorted({p["repo"] for p in seeds.values()})

    def visibility_of(repo):
        data = gh("api", f"repos/{repo}")
        return repo, "public" if data.get("visibility") == "public" and not data[
            "private"
        ] else "restricted"

    with ThreadPoolExecutor(max_workers=4) as executor:
        visibility = dict(executor.map(visibility_of, repositories))

    def detail(seed):
        pr = gh("pr", "view", seed["url"], "--json", FIELDS)
        pr["repo"] = seed["repo"]
        pr["owned"] = pr["author"]["login"].lower() == author.lower()
        if pr["state"] == "OPEN":
            pr["reviewThreads"] = review_threads(pr)
        else:
            pr["reviewThreads"] = []
        return pr

    with ThreadPoolExecutor(max_workers=4) as executor:
        prs = list(executor.map(detail, seeds.values()))
    issues = []
    for url in notes.get("issues", []):
        d = gh("issue", "view", url, "--json", "url,title,state,updatedAt,comments")
        d["repo"] = "/".join(url.split("/")[3:5])
        issues.append(d)
    snapshot = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "author": author,
        "visibility": visibility,
        "prs": prs,
        "issues": issues,
    }
    write_json(runtime / "latest_raw.json", snapshot)
    return snapshot


def note_for(pr, notes):
    return notes.get("prs", {}).get(
        key(pr),
        {
            "priority": "TRIAGE",
            "deployment": "Unclassified: inspect before adding to a deployment",
            "purpose": pr["title"],
            "owner": "yihou; delegate unassigned",
            "ready_action": "New/unclassified PR: review diff, purpose, ownership, evidence and dependencies; write curated handoff notes.",
            "merge_action": "Resolve current review and required CI; no automatic merge certification.",
            "evidence": "No curated evidence yet; inspect the PR.",
            "tests": [],
            "last_action": "Discovered by the daily author-wide query.",
        },
    )


def evidence_status(pr, note):
    audited = note.get("reviewed_head")
    if not audited:
        return "NOT AUDITED"
    if not pr["headRefOid"].startswith(audited):
        return f"HEAD CHANGED: re-audit evidence from {audited[:10]}"
    return "Recorded head unchanged; evidence retains its stated limits"


def activity_status(pr, note):
    if note.get("reviewed_activity_at") != pr["updatedAt"]:
        return (
            "NEW/UNAUDITED ACTIVITY: inspect new comments, reviews, checks and dependency changes"
        )
    return "GitHub activity timestamp matches the curated audit"


def last_activity(pr):
    events = [(c["createdAt"], c["url"]) for c in pr.get("comments", [])]
    if events:
        timestamp, url = max(events)
        return f"[Latest comment {timestamp}]({url}); PR updated {pr['updatedAt']}"
    return f"PR updated {pr['updatedAt']}; no issue comments"


def count_text(prs):
    counts = Counter(state(p) for p in prs)
    return (
        f"{len(prs)} PRs: {counts['READY'] + counts['DRAFT']} open "
        f"({counts['READY']} ready / {counts['DRAFT']} draft), "
        f"{counts['MERGED']} merged, {counts['CLOSED']} closed unmerged"
    )


def link(pr):
    return f"[{pr['repo']} #{pr['number']}]({pr['url']})"


def order(pr, notes):
    note = note_for(pr, notes)
    return note.get("priority", "TRIAGE"), pr["repo"], pr["number"]


def active_table(prs, notes):
    lines = [
        "| Priority | PR / purpose | State / head | Historical 47k-class deployment | Ready action | Merge action / waiting on |",
        "|---|---|---|---|---|---|",
    ]
    for pr in sorted(prs, key=lambda p: order(p, notes)):
        n = note_for(pr, notes)
        lines.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    [
                        n["priority"],
                        link(pr) + " — " + n["purpose"],
                        f"{state(pr)} / `{pr['headRefOid'][:10]}`",
                        n["deployment"],
                        ("Already Ready. " if state(pr) == "READY" else "") + n["ready_action"],
                        n["merge_action"],
                    ],
                )
            )
            + " |"
        )
    return lines


def render(prs, external, issues, notes, checked, internal=False):
    active = [p for p in prs if p["state"] == "OPEN"]
    lines = [
        "# "
        + (
            "Account-wide PR handoff — internal"
            if internal
            else "Public PR handoff and daily status"
        ),
        "",
        f"Last successful GitHub refresh: **{checked}**. Author: `dorado269`.",
        "",
        f"**{count_text(prs)}.** Includes historical work outside GLM-5.2; external authors are listed separately.",
        "",
        "Ready means non-draft, not maintainer approval or passed functional CI. GitHub mergeability is a snapshot, not a merge recommendation.",
        "",
        "Daily refresh: 09:00 UTC (17:00 Asia/Shanghai), on the configured maintenance host. A timestamp older than 36 hours requires operator investigation. Raw CI counts can include multiple workflows; skipped/draft/permission gates do not establish executed test coverage.",
        "",
        "Task owner is yihou until an explicit delegate is recorded. Requested reviewers are not assumed to have accepted work. Edit curated notes to record a named delegate, evidence and the next action; the updater preserves those notes and flags changed heads.",
        "",
        "Priority P0 = historical frozen-image behavior / required reproduction inputs; P1 = related current-main kernel ports and evidence packaging; P2 = other GLM fixes; P3 = unrelated or comparison work. See [deployment requirements](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/LLR37_DEPLOYMENT_PRIORITY.md).",
        "",
    ]
    if internal:
        lines += [
            "**INTERNAL: contains restricted-repository links. Share only with colleagues authorized for those repositories; do not publish this file to the public repository.**",
            "",
        ]
    lines += ["## Active work", "", *active_table(active, notes), "", "## Handoff details", ""]
    for pr in sorted(active, key=lambda p: order(p, notes)):
        n = note_for(pr, notes)
        requests = (
            ", ".join(r.get("login", r.get("name", "team")) for r in pr.get("reviewRequests", []))
            or "none requested"
        )
        assigned = ", ".join(a["login"] for a in pr.get("assignees", [])) or "none"
        unresolved = [t for t in pr["reviewThreads"] if not t["isResolved"]]
        reviews = (
            ", ".join(
                f"{r['author']['login'] if r.get('author') else 'deleted'}: {r['state']} ({r.get('submittedAt') or 'pending'})"
                for r in pr.get("reviews", [])
            )
            or "none"
        )
        lines += [
            f"### {link(pr)} — {n['purpose']}",
            "",
            f"- **State / responsibility:** {state(pr)}; {n.get('owner', 'yihou; delegate unassigned')}. GitHub assignees: {assigned}. Requested reviewers: {requests}.",
            f"- **Base / head:** `{pr['baseRefName']}` @ `{pr['baseRefOid']}` / `{pr['headRefName']}` @ `{pr['headRefOid']}`.",
            f"- **Review / CI:** {pr.get('reviewDecision') or 'no decision'}; mergeability {pr['mergeable']} / {pr['mergeStateStatus']}; {len(unresolved)} unresolved inline threads. Checks: {checks_text(pr)}. Reviews: {reviews}.",
            f"- **Last curated action:** {n['last_action']}",
            f"- **Latest GitHub activity:** {last_activity(pr)}.",
            f"- **Evidence freshness:** {evidence_status(pr, n)}. Notes last assessed {n.get('assessed_at', 'unrecorded')}; new comments/reviews still require human triage.",
            f"- **Activity triage:** {activity_status(pr, n)}.",
            f"- **Evidence / pack-up:** {n['evidence']}",
            f"- **To Ready:** {n['ready_action']}",
            f"- **To merge / dependency:** {n['merge_action']}",
            "",
        ]
        for thread in unresolved:
            comment = thread["comments"]["nodes"][0]
            lines += [
                f"- Unresolved [review thread]({comment['url']}) in `{thread['path']}` (outdated={thread['isOutdated']}); inspect the current diff before resolving."
            ]
        failed = [
            c
            for c in pr.get("statusCheckRollup") or []
            if (c.get("conclusion") or c.get("state"))
            in {"FAILURE", "ERROR", "TIMED_OUT", "ACTION_REQUIRED"}
        ]
        if failed:
            lines += [
                "- Red-check entry points: "
                + "; ".join(
                    f"[{cell(c.get('name') or c.get('context') or 'check')}]({c.get('detailsUrl') or c.get('targetUrl') or pr['url'] + '/checks'})"
                    for c in failed[:5]
                )
                + (
                    "; remaining checks are in the linked PR and JSON snapshot."
                    if len(failed) > 5
                    else "."
                ),
                "",
            ]
        if n.get("tests"):
            lines += [
                "Validation entry points (run in an isolated checkout at the recorded head; allocate a pool node for GPU commands):",
                "",
                "```sh",
                *n["tests"],
                "```",
                "",
            ]
    lines += [
        "## External dependencies and equivalent work",
        "",
        "These PRs belong to their original authors. Contribute evidence there; do not open duplicates or edit their branches.",
        "",
        "| PR | Purpose | State / head | Dependency / next action |",
        "|---|---|---|---|",
    ]
    for pr in sorted(external, key=lambda p: order(p, notes)):
        n = note_for(pr, notes)
        lines.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    [
                        link(pr),
                        n.get("purpose", pr["title"]),
                        f"{state(pr)} / `{pr['headRefOid'][:10]}`",
                        n["merge_action"],
                    ],
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Issue-only work and patches without a PR",
        "",
        "| Issue | State | Updated |",
        "|---|---|---|",
    ]
    for issue in issues:
        lines.append(
            f"| [{cell(issue['title'])}]({issue['url']}) | {issue['state']} | {issue['updatedAt']} |"
        )
    lines += [
        "",
        "Required unpublished S13, A01, the full historical K01 overlay, I07 and inherited image patches are recorded in [deployment requirements](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/LLR37_DEPLOYMENT_PRIORITY.md). Remaining exclusions and reasons are in [PR_PACKUP_MAP.md](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/PR_PACKUP_MAP.md).",
        "",
        "## Complete authored history",
        "",
        "| PR | Purpose / title | State | Last GitHub update | Next action |",
        "|---|---|---|---|---|",
    ]
    for pr in sorted(prs, key=lambda p: (p["repo"], p["number"])):
        n = note_for(pr, notes)
        action = (
            "See active handoff above"
            if pr["state"] == "OPEN"
            else n.get(
                "disposition",
                "Merged: retain provenance; no active merge work"
                if pr["state"] == "MERGED"
                else "Closed unmerged: retained for history; do not infer a fix landed or reopen without rechecking replacements",
            )
        )
        lines.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    [link(pr), n.get("purpose", pr["title"]), state(pr), pr["updatedAt"], action],
                )
            )
            + " |"
        )
    lines += [
        "",
        "The historical llr37-c104 result is 48,713 total tokens/s/GPU (input + output, eight GPUs), using simulated MTP acceptance. PP overlap and decode custom all-reduce have known real-acceptance correctness failures. This table makes no individual-patch speedup or production-correctness claim.",
        "",
    ]
    return "\n".join(lines)


def compact(pr):
    fields = (
        "repo",
        "number",
        "url",
        "title",
        "state",
        "isDraft",
        "owned",
        "author",
        "updatedAt",
        "mergedAt",
        "closedAt",
        "headRefOid",
        "baseRefOid",
        "headRefName",
        "baseRefName",
        "mergeable",
        "mergeStateStatus",
        "reviewDecision",
        "labels",
        "assignees",
        "reviewRequests",
    )
    result = {k: pr[k] for k in fields}
    result["unresolved_review_threads"] = sum(not t["isResolved"] for t in pr["reviewThreads"])
    result["latest_check_counts"] = check_counts(pr)
    result["checks_url"] = pr["url"] + "/checks"
    return result


def generate(snapshot, inventory, runtime, public_notes, private_notes):
    prs, visibility = snapshot["prs"], snapshot["visibility"]
    checked = snapshot["checked_at_utc"]
    public = public_only(prs, visibility)
    public_owned = [p for p in public if p["owned"]]
    external = [p for p in public if not p["owned"]]
    issues = public_only(snapshot["issues"], visibility)
    account_notes = {**public_notes, "prs": {**public_notes["prs"], **private_notes.get("prs", {})}}
    public_doc = render(public_owned, external, issues, public_notes, checked)
    # Fail closed on restricted repository identifiers in tracked output/notes.
    restricted = [repo for repo, value in visibility.items() if value != "public"]
    serialized = public_doc + json.dumps(public_notes)
    if any(repo.lower() in serialized.lower() for repo in restricted):
        raise RuntimeError("Restricted repository identifier found in public output/notes")
    write_text(inventory / "PR_HANDOFF.md", public_doc)
    write_text(
        runtime / "PR_HANDOFF_ALL.md",
        render(
            [p for p in prs if p["owned"]],
            external,
            snapshot["issues"],
            account_notes,
            checked,
            internal=True,
        ),
    )
    write_json(
        inventory / "repro/PR_MAINTENANCE_SNAPSHOT.json",
        {
            "checked_at_utc": checked,
            "author": snapshot["author"],
            "prs": [compact(p) for p in public],
            "issues": [{k: i[k] for k in ("url", "title", "state", "updatedAt")} for i in issues],
        },
    )
    target = [p for p in public_owned if p["repo"] in TARGETS]
    index = [
        "# Owned PR index for the upstreaming target repositories",
        "",
        f"Checked: {checked}. Author: `dorado269`.",
        "",
        f"**{count_text(target)}.** Full actionable public handoff: [PR_HANDOFF.md](PR_HANDOFF.md). External authors are excluded from these counts.",
        "",
        "| PR | Purpose / title | State | Next action |",
        "|---|---|---|---|",
    ]
    for p in sorted(target, key=lambda x: (x["repo"], x["number"])):
        n = note_for(p, public_notes)
        index.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    [
                        link(p),
                        n.get("purpose", p["title"]),
                        state(p),
                        "See handoff"
                        if p["state"] == "OPEN"
                        else n.get("disposition", "Historical; no active merge work"),
                    ],
                )
            )
            + " |"
        )
    write_text(inventory / "OWNED_PR_INDEX.md", "\n".join(index) + "\n")
    status_path = inventory / "PR_STATUS.md"
    status = status_path.read_text()
    section = "\n".join(
        [
            BEGIN,
            f"Last automatic GitHub refresh: **{checked}**. **{count_text(target)}** in the four target repositories.",
            "",
            "Current actionable source: [PR_HANDOFF.md](PR_HANDOFF.md); [deployment priorities](LLR37_DEPLOYMENT_PRIORITY.md); [daily-maintenance instructions](maintenance/README.md). The dated records below retain earlier decisions and may be superseded by this live section.",
            "",
            *active_table([p for p in target if p["state"] == "OPEN"], public_notes),
            END,
        ]
    )
    if BEGIN in status:
        start, rest = status.split(BEGIN, 1)
        _, end = rest.split(END, 1)
        status = start + section + end
    else:
        title, rest = status.split("\n", 1)
        status = (
            title
            + "\n\n"
            + section
            + "\n\n## Dated maintenance records (snapshot before daily automation)\n"
            + rest
        )
    write_text(status_path, status)
    return {
        "checked_at_utc": checked,
        "account": count_text([p for p in prs if p["owned"]]),
        "public": count_text(public_owned),
        "target_repositories": count_text(target),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--author", default="dorado269")
    parser.add_argument("--from-snapshot", type=Path)
    args = parser.parse_args()
    for path in (args.inventory, args.runtime):
        if "yihou" not in str(path.resolve()):
            raise SystemExit("All output directories must contain yihou in their absolute path")
    notes = json.loads((args.inventory / "maintenance/handoff_notes.json").read_text())
    private_path = args.runtime / "private_notes.json"
    private = json.loads(private_path.read_text()) if private_path.exists() else {}
    snapshot = (
        json.loads(args.from_snapshot.read_text())
        if args.from_snapshot
        else collect(args.author, notes, args.runtime)
    )
    result = generate(snapshot, args.inventory, args.runtime, notes, private)
    write_json(args.runtime / "last_refresh.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
