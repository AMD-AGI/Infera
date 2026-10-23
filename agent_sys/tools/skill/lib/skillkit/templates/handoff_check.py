#!/usr/bin/env python3
"""A declarative handoff validator body, configured entirely by `args`.

Copied into a generated package as `assets/lib/handoff_check.py` by
`astk scaffold`; each validator's `entry.sh` runs it. Standard library only
(YAML files are parsed when PyYAML happens to be importable).

Validator contract (agent_sys `validator/phase.py`): cwd is a fresh zone
holding `inputs.json` (handoff ids), `args.json` (this validator's `args`) and
`materials.json` (id -> content dir with README.md and items/). The body MUST
write `verdict.json` `{id: bool}` — this one always does, even when it crashes,
because a missing verdict is "nothing was decided", which helps nobody. Reasons
go to `validator_report.txt` (stdout is kept only as the zone's body_output.txt).

Calibration — what this checks and what it refuses to check:
  * presence, non-emptiness, structure, parseability, internal consistency,
    and stated numeric bounds — things a correct result always satisfies;
  * never wording, style, length beyond a floor, or an exact number. A floor is
    a floor: set it well under what a good result has, and say why in `why`.
Only PROBLEM lines fail a handoff; `note:` lines never do.

args (all optional):
  require_items:   [name, ...]          items/<name> exists and is non-empty
  require_files:   [glob, ...]          each glob (relative to content/) matches >= 1 non-empty file
  min_lines:       {glob: n}            every matching file (or file under a matching dir) has >= n non-blank lines
  min_matches:     {glob: {pattern, n, why}}  >= n regex matches across the matching files (e.g. citations)
  json_files:      {glob: [key, ...]}   every match parses (.json/.yaml/.yml) and has the dotted keys
  executable:      [glob, ...]          every match has the executable bit
  shell_parses:    [glob, ...]          every match passes `sh -n` (bash -n when its shebang says bash)
  readme_sections: [name, ...]          extra document-root README headings (agent_sys does not enforce a kind's)
  numeric:         [{file, key, min, max, why}]  a value in a JSON/YAML file within bounds
  forbid_placeholders: true             template text ([More Information Needed], <fill ...>, TODO: fill)
                                        in prose; quoted in `code` or a fence it is a mention, not a gap
  forbid_secrets:  true                 literal secret values (names and $VAR references are fine)
  max_file_mb:     n                    no single file above n MB (unset = only a note above 4 MB)
"""

import fnmatch
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

PLACEHOLDER = re.compile(r"\[More Information Needed\]|<fill[^>]*>|TODO: fill", re.I)
# -- copied from skillkit/secretscan.py (a test keeps the two equal) --
#: A value: not a flag; has a letter; and either a digit (8+ chars) or 16+ chars —
#: random-looking and not an UPPER_CASE constant name, so `initial-cluster-token:
#: etcd-cluster` and `token = "M4_FRAGMENT_FIRST_CALL"` are not ones.
_VALUE = r"(?!-)(?![A-Z0-9_]+(?:[\s\"',;}{]|$))(?=[^\s\"',;}{]*[A-Za-z])(?:(?=[^\s\"',;}{]*\d)[^\s\"',;}{]{8,}|[^\s\"',;}{]{16,})"
#: A credential-ish name at the END of an identifier (so `max_tokens` is not one).
_NAME = (
    r"(?i:[A-Za-z0-9_.-]*(?:api[_-]?key|access[_-]?key|secret(?:[_-]?key)?|token|password|passwd"
    r"|subscription[_-]?key|auth))"
)

#: (pattern, replacement). Specific before generic.
PATTERNS: list[tuple[str, str]] = [
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "<REDACTED-PRIVATE-KEY>"),
    (r"sk-ant-[A-Za-z0-9_\-]{8,}", "sk-ant-<REDACTED>"),
    (r"\bsk-[A-Za-z0-9_\-]{20,}", "sk-<REDACTED>"),
    (r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{16,}", "<REDACTED-GITHUB-TOKEN>"),
    (r"\bglpat-[A-Za-z0-9_\-]{16,}", "<REDACTED-GITLAB-TOKEN>"),
    (r"\bxox[abprs]-[A-Za-z0-9-]{10,}", "<REDACTED-SLACK-TOKEN>"),
    (r"\bhf_[A-Za-z0-9]{20,}", "<REDACTED-HF-TOKEN>"),
    (r"\bAKIA[0-9A-Z]{16}\b", "<REDACTED-AWS-KEY>"),
    (r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}", r"\1<REDACTED>"),
    (r"(?i)(basic\s+)[A-Za-z0-9+/]{16,}={0,2}", r"\1<REDACTED>"),
    (r"([a-z][a-z0-9+.-]*://[^\s:/@]+:)[^\s@/]{3,}(@)", r"\1<REDACTED>\2"),
    (rf"((?<![A-Za-z0-9]){_NAME}[\"']?\s*[:=]\s*[\"']?){_VALUE}", r"\1<REDACTED>"),
    (rf"(--{_NAME}[= ]\s*[\"']?){_VALUE}", r"\1<REDACTED>"),
]

SECRETS = [re.compile(p) for p, _ in PATTERNS]
_REFERENCE = re.compile(r"[$(\[<{]|REDACTED")
# -- end of copy --
TEXT_LIMIT = 4 << 20
#: Placeholder text in source code is code (e.g. this file's own pattern), not a gap.
CODE_SUFFIXES = frozenset(
    ".py .sh .bash .js .mjs .ts .c .cc .cpp .h .hpp .go .rs .java .cu .hip".split()
)


def _load(name, default):
    p = Path(name)
    return json.loads(p.read_text()) if p.is_file() else default


def _int(v, default=None):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _list(v):
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return list(v)


def _matches(root, pattern):
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if fnmatch.fnmatchcase(rel, pattern):
            out.append(p)
    return out


def _parse(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415
        except ImportError:
            return None, "PyYAML is not importable here, so a YAML file cannot be read"
        return yaml.safe_load(text), None
    return json.loads(text), None


def _dig(doc, dotted):
    cur = doc
    for part in str(dotted).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, None
    return True, cur


def _prose(text):
    """{line number: text} outside fences, with `inline code` removed.

    A report that *quotes* a placeholder (`[More Information Needed]`) is
    describing it, not leaving one; only prose still holding it is unfilled."""
    out, fence = {}, False
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
            continue
        if not fence:
            out[n] = re.sub(r"`[^`]*`", "", line)
    return out


def _files(root, pattern):
    """Files matching `pattern`; a matching directory contributes its files."""
    out = []
    for p in _matches(root, pattern):
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in sorted(p.rglob("*")) if f.is_file()]
    return list(dict.fromkeys(out))  # a file under a matching dir may match itself too


def _headings(text):
    out, fence = {}, False
    current = None
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        if fence:
            continue
        m = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if m:
            current = m.group(1)
            out[current] = ""
        elif current is not None:
            out[current] += line.strip()
    return out


def check(root, args):
    problems, notes = [], []
    items = root / "items"

    for name in _list(args.get("require_items")):
        p = items / name
        if not p.exists():
            problems.append(
                f"items/{name} is missing — write it (see the task readme for what it holds)"
            )
        elif p.is_file() and p.stat().st_size == 0:
            problems.append(f"items/{name} is empty — it must carry the content, not a placeholder")
        elif p.is_dir() and not any(p.iterdir()):
            problems.append(f"items/{name}/ is an empty directory")

    for pattern in _list(args.get("require_files")):
        hits = [p for p in _matches(root, pattern) if p.is_file() and p.stat().st_size > 0]
        if not hits:
            problems.append(f"no non-empty file matches {pattern!r} under the handoff")

    for pattern, floor in (args.get("min_lines") or {}).items():
        floor = _int(floor, 0)
        hits = _files(root, pattern)
        if not hits:
            problems.append(f"min_lines: nothing matches {pattern!r}")
        for p in hits:
            n = sum(1 for ln in p.read_text(errors="replace").splitlines() if ln.strip())
            if n < floor:
                problems.append(
                    f"{p.relative_to(root)} has {n} non-blank lines; the floor is {floor} "
                    f"(a floor, not a target: it says the file is a stub)"
                )

    for pattern, spec in (args.get("min_matches") or {}).items():
        regex, floor, why = spec.get("pattern", ""), _int(spec.get("n"), 1), spec.get("why", "")
        count = 0
        for p in _files(root, pattern):
            try:
                count += len(re.findall(regex, p.read_text(errors="replace"), re.M))
            except re.error as exc:
                problems.append(f"min_matches {pattern!r}: bad pattern {regex!r}: {exc}")
                break
        if count < floor:
            problems.append(
                f"{pattern}: {count} matches of {regex!r}; the floor is {floor} — {why}".rstrip(
                    " —"
                )
            )
        else:
            notes.append(f"{pattern}: {count} matches of {regex!r} (floor {floor})")

    for pattern, keys in (args.get("json_files") or {}).items():
        hits = [p for p in _matches(root, pattern) if p.is_file()]
        if not hits:
            problems.append(f"no file matches {pattern!r} (expected a structured result there)")
        for p in hits:
            try:
                doc, why = _parse(p)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{p.relative_to(root)} does not parse: {exc}")
                continue
            if why:
                notes.append(f"{p.relative_to(root)}: {why}")
                continue
            for key in _list(keys):
                found, value = _dig(doc, key)
                if not found or value in (None, "", [], {}):
                    problems.append(f"{p.relative_to(root)} has no value at {key!r}")

    for pattern in _list(args.get("executable")):
        for p in _matches(root, pattern):
            if p.is_file() and not os.access(p, os.X_OK):
                problems.append(f"{p.relative_to(root)} is not executable — chmod +x it")

    for pattern in _list(args.get("shell_parses")):
        for p in _matches(root, pattern):
            if not p.is_file():
                continue
            first = p.read_text(errors="replace").split("\n", 1)[0]
            shell = "bash" if "bash" in first else "sh"
            r = subprocess.run([shell, "-n", str(p)], capture_output=True, text=True, check=False)
            if r.returncode != 0:
                problems.append(
                    f"{p.relative_to(root)} does not parse under {shell}: {r.stderr.strip()[:300]}"
                )

    extra = _list(args.get("readme_sections"))
    if extra:
        readme = root / "README.md"
        found = _headings(readme.read_text(errors="replace")) if readme.is_file() else {}
        for name in extra:
            if name not in found:
                problems.append(f"README.md has no heading {name!r}")
            elif not found[name].strip():
                problems.append(f"README.md section {name!r} is empty")

    for spec in args.get("numeric") or []:
        f, key = spec.get("file"), spec.get("key")
        hits = [p for p in _matches(root, str(f)) if p.is_file()]
        if not hits:
            problems.append(f"numeric check: no file matches {f!r}")
            continue
        try:
            doc, why = _parse(hits[0])
        except Exception as exc:  # noqa: BLE001
            problems.append(f"numeric check: {hits[0].relative_to(root)} does not parse: {exc}")
            continue
        found, value = _dig(doc, key)
        value = _float(value) if found else None
        if value is None:
            problems.append(f"numeric check: {hits[0].relative_to(root)} has no number at {key!r}")
            continue
        lo, hi = _float(spec.get("min")), _float(spec.get("max"))
        why = spec.get("why", "")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            problems.append(f"{key} = {value} is outside [{lo}, {hi}] — {why}".rstrip(" —"))
        else:
            notes.append(f"{key} = {value} within [{lo}, {hi}]")

    scan_ph = args.get("forbid_placeholders", True) not in (False, "false", "0", 0)
    scan_sec = args.get("forbid_secrets", True) not in (False, "false", "0", 0)
    max_mb = _float(args.get("max_file_mb"))
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        size = p.stat().st_size
        rel = p.relative_to(root)
        if max_mb is not None and size > max_mb * 2**20:
            problems.append(
                f"{rel} is {size / 2**20:.1f} MB, over the {max_mb} MB cap — reference it by path + sha256"
            )
        elif size > 4 << 20:
            notes.append(f"{rel} is {size / 2**20:.1f} MB")
        if size > TEXT_LIMIT or not (scan_ph or scan_sec):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        prose = _prose(text) if scan_ph and p.suffix not in CODE_SUFFIXES else {}
        for n, line in enumerate(text.splitlines(), 1):
            if scan_ph and PLACEHOLDER.search(prose.get(n, "")):
                problems.append(f"{rel}:{n} still holds template text: {line.strip()[:80]!r}")
            if scan_sec:
                for m in (m for pat in SECRETS for m in pat.finditer(line)):
                    if m.group(0).startswith("-----BEGIN") or not _REFERENCE.search(m.group(0)):
                        problems.append(
                            f"{rel}:{n} carries what looks like a secret value — "
                            "replace it with the variable's NAME and where it comes from"
                        )
                        break
    return problems, notes


def main():
    ids = _load("inputs.json", [])
    args = _load("args.json", {})
    materials = _load("materials.json", {})
    verdict, report = {}, []
    for hid in ids:
        try:
            staged = materials.get(hid)
            root = Path(staged) if staged else None
            if root is None or not root.is_dir():
                verdict[hid] = False
                report.append(
                    f"{hid}: REFUSED\n  PROBLEM: no staged content (materials.json has {staged!r})"
                )
                continue
            problems, notes = check(root, args)
            verdict[hid] = not problems
            head = "passed" if not problems else "REFUSED"
            report.append(f"{hid}: {head}")
            report += [f"  PROBLEM: {p}" for p in problems]
            report += [f"  note: {n}" for n in notes]
        except Exception:  # noqa: BLE001 — a verdict must always be written
            verdict[hid] = False
            report.append(
                f"{hid}: DID NOT RUN (validator crashed — a validator bug, not the handoff's)\n"
                + traceback.format_exc()
            )
    Path("validator_report.txt").write_text("\n".join(report) + "\n")
    Path("verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
