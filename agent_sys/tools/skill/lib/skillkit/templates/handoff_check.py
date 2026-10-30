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

Calibration: it checks presence, structure, parseability and stated numeric
bounds — what a correct result always satisfies — never wording, style, length
beyond a floor, or an exact number. Only PROBLEM lines fail; `note:` lines never do.

args (all optional):
  require_items:   [name, ...]          items/<name> exists and is non-empty
  require_files:   [glob, ...]          each glob (relative to content/) matches >= 1 non-empty file
  min_lines:       {glob: n}            every matching file (or file under a matching dir) has >= n non-blank lines
  min_matches:     {glob: {pattern, n, why}}  >= n regex matches across the matching files (e.g. citations)
  json_files:      {glob: [key, ...]}   every match parses (.json/.yaml/.yml) and has the dotted keys
  executable:      [glob, ...]          every match has the executable bit
  shell_parses:    [glob, ...]          every match passes `sh -n` (bash -n when its shebang says bash)
  readme_sections: [name, ...]          extra document-root README headings (agent_sys does not enforce a kind's)
  numeric:         [{file, key, min, max, why}]  EVERY file `file` matches holds a finite number
                                        at `key` within bounds (NaN/inf fail)
  forbid_placeholders: true             template text in prose; quoted in `code` or a fence it is a mention
  forbid_secrets:  true                 literal secret values in every file, gzip/bz2/xz/zip decompressed
                                        first; more than 64 MB of text is refused as unscanned; of binary
                                        content the printable runs of the first 64 MB are scanned
  max_file_mb:     n                    no single file above n MB (unset = only a note above 4 MB)
"""

import bz2
import codecs
import fnmatch
import gzip
import json
import lzma
import math
import os
import re
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

PLACEHOLDER = re.compile(r"\[More Information Needed\]|<fill[^>]*>|TODO: fill", re.I)
# -- copied from skillkit/secretscan.py, which explains each part (a test keeps the two equal) --
_END = r"(?:[\s\"',;}{()\[\]]|$)"
_PART = r"[A-Za-z_][A-Za-z_]*[0-9]{0,2}"
_VALUE = (
    r"(?!-)(?!~?\.{0,2}/)(?![a-z][a-z0-9+.-]*://)"
    rf"(?!{_PART}(?:\.{_PART})+(?:/|{_END}))"  # os.environ/NAME
    rf"(?!_*[a-z]+[0-9]{{0,2}}(?:_+(?:[a-z]+[0-9]{{0,2}}|[0-9]{{1,2}}))+_*{_END})"
    rf"(?!_*[A-Z][A-Z0-9]*(?:_+[A-Z0-9]+)+{_END})(?!_*[A-Za-z]+{_END})"
    r"(?=[^\s\"',;}{]*[A-Za-z])(?:(?=[^\s\"',;}{]*\d)[^\s\"',;}{]{8,}|[^\s\"',;}{]{16,})"
)
_NAME = (
    r"(?i:[A-Za-z0-9_.-]*(?:api[_-]?key|access[_-]?key|secret(?:[_-]?key)?|token|password"
    r"|passwd|subscription[_-]?key|auth))"
)
_ENV_NAME = r"[A-Z][A-Z0-9_]*_(?:KEY|PASS|PWD)"
_ARG = (
    r"(?![A-Z]+(?:_[A-Z0-9]+)*(?:[\s\"']|$))"
    r"(?!(?i:pass(?:word|wd)?|pwd?|secret|token|x{3,}|\*+)(?:[\s\"']|$))[^\s\"']{3,}"
)
PATTERNS: list[tuple[str, str]] = [
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "<REDACTED-PRIVATE-KEY>"),
    (r"sk-ant-[A-Za-z0-9_\-]{8,}", "sk-ant-<REDACTED>"),
    (r"\bsk-[A-Za-z0-9_\-]{20,}", "sk-<REDACTED>"),
    (r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{16,}", "<REDACTED-GITHUB-TOKEN>"),
    (r"\bglpat-[A-Za-z0-9_\-]{16,}", "<REDACTED-GITLAB-TOKEN>"),
    (r"\bxox[abprs]-[A-Za-z0-9-]{10,}", "<REDACTED-SLACK-TOKEN>"),
    (r"\bhf_[A-Za-z0-9]{20,}", "<REDACTED-HF-TOKEN>"),
    (r"\bdckr_pat_[A-Za-z0-9_\-]{10,}", "<REDACTED-DOCKER-TOKEN>"),
    (r"\bAKIA[0-9A-Z]{16}\b", "<REDACTED-AWS-KEY>"),
    (r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}", "<REDACTED-JWT>"),
    (r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}", r"\1<REDACTED>"),
    (r"(?i)(basic\s+)[A-Za-z0-9+/]{16,}={0,2}", r"\1<REDACTED>"),
    (r"([a-z][a-z0-9+.-]*://[^\s:/@]+:)[^\s@/]{3,}(@)", r"\1<REDACTED>\2"),
    (rf"(\bsshpass\s+-p\s*[\"']?){_ARG}", r"\1<REDACTED>"),
    (rf"(\bcurl\b[^\n|;&]*?\s(?:-u|--user)[\s=]*[\"']?[^\s:\"']+:){_ARG}", r"\1<REDACTED>"),
    (
        r"(\b(?:docker|podman|helm|nerdctl)\s+(?:registry\s+)?login\b[^\n|;&]*?\s(?:-p|--password)[\s=]+[\"']?)"
        + _ARG,
        r"\1<REDACTED>",
    ),
    (rf"(\blogin\s+\S+\s+password\s+){_VALUE}", r"\1<REDACTED>"),  # .netrc
    (rf"((?<![A-Za-z0-9]){_NAME}[\"']?\s*[:=]\s*[\"']?){_VALUE}", r"\1<REDACTED>"),
    (
        rf"((?<![A-Za-z0-9]){_ENV_NAME}[\"']?\s*[:=]\s*[\"']?)(?=[^\s\"',;}}{{]*\d){_VALUE}",
        r"\1<REDACTED>",
    ),
    (rf"(--{_NAME}[= ]\s*[\"']?){_VALUE}", r"\1<REDACTED>"),
]
SECRETS = [re.compile(p) for p, _ in PATTERNS]
_REFERENCE = re.compile(r"[$(\[<{]|REDACTED")


def _is_reference(m):
    """Judged on the VALUE only (after group 1), so a curl `-u` user:password pair is
    still a literal password even though the command line holds a `$`."""
    if m.group(0).startswith("-----BEGIN"):
        return False
    value = m.group(0)[m.end(1) - m.start(0) :] if m.re.groups else m.group(0)
    return bool(_REFERENCE.search(value))


# -- end of copy --
TEXT_LIMIT = 4 << 20
#: How much (decompressed) text of one file the secret scan reads.
SCAN_LIMIT = 64 << 20
#: A printable-ASCII run in binary content, where a secret would be.
_PRINTABLE = re.compile(rb"[\t\x20-\x7e]{8,}")
#: Placeholder text in source code is code (e.g. this file's own pattern), not a gap.
CODE_SUFFIXES = frozenset(
    ".py .sh .bash .js .mjs .ts .c .cc .cpp .h .hpp .go .rs .java .cu .hip".split()
)
#: Magic bytes of the single-stream compressors the secret scan opens.
_OPENERS = ((b"\x1f\x8b", gzip.open), (b"BZh", bz2.open), (b"\xfd7zXZ\x00", lzma.open))


def _load(name, default):
    p = Path(name)
    return json.loads(p.read_text()) if p.is_file() else default


def _int(v, default=None):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _float(v):
    """A finite float, else None: NaN passes every `<` / `>` bound, so it is no number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _list(v):
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return list(v)


def _off(v):
    return v in (False, "false", "0", 0)


def _because(msg, why):
    return f"{msg} — {why}" if why else msg


def _matches(root, pattern):
    return [
        p
        for p in sorted(root.rglob("*"))
        if fnmatch.fnmatchcase(p.relative_to(root).as_posix(), pattern)
    ]


def _files(root, pattern):
    """Files matching `pattern`; a matching directory contributes its files."""
    out = []
    for p in _matches(root, pattern):
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in sorted(p.rglob("*")) if f.is_file()]
    return list(dict.fromkeys(out))  # a file under a matching dir may match itself too


def _parse(path):
    """(document, None), or (None, why it cannot be read here)."""
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


def _unfenced(text):
    """(line number, line) outside ``` / ~~~ fences."""
    fence = False
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        elif not fence:
            yield n, line


def _headings(text):
    """{heading: its body}; a heading inside a fence is not one, but the fence is body."""
    out, current, fence = {}, None, False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        m = None if fence else re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if m:
            current = m.group(1)
            out[current] = ""
        elif current is not None:
            out[current] += line.strip()
    return out


def check(root, args):
    problems, notes = [], []

    for name in _list(args.get("require_items")):
        p = root / "items" / name
        if not p.exists():
            problems.append(
                f"items/{name} is missing — write it (see the task readme for what it holds)"
            )
        elif p.is_file() and p.stat().st_size == 0:
            problems.append(f"items/{name} is empty — it must carry the content, not a placeholder")
        elif p.is_dir() and not any(p.iterdir()):
            problems.append(f"items/{name}/ is an empty directory")

    for pattern in _list(args.get("require_files")):
        if not any(p.is_file() and p.stat().st_size > 0 for p in _matches(root, pattern)):
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
        regex, floor = spec.get("pattern", ""), _int(spec.get("n"), 1)
        count = 0
        for p in _files(root, pattern):
            try:
                count += len(re.findall(regex, p.read_text(errors="replace"), re.M))
            except re.error as exc:
                problems.append(f"min_matches {pattern!r}: bad pattern {regex!r}: {exc}")
                break
        msg = f"{pattern}: {count} matches of {regex!r}"
        if count < floor:
            problems.append(_because(f"{msg}; the floor is {floor}", spec.get("why")))
        else:
            notes.append(f"{msg} (floor {floor})")

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
                nan = isinstance(value, float) and not math.isfinite(value)
                if not found or nan or value in (None, "", [], {}):
                    problems.append(f"{p.relative_to(root)} has no value at {key!r}")

    for pattern in _list(args.get("executable")):
        for p in _matches(root, pattern):
            if p.is_file() and not os.access(p, os.X_OK):
                problems.append(f"{p.relative_to(root)} is not executable — chmod +x it")

    for pattern in _list(args.get("shell_parses")):
        for p in (p for p in _matches(root, pattern) if p.is_file()):
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
        _numeric(root, spec, problems, notes)

    scan_ph = not _off(args.get("forbid_placeholders", True))
    scan_sec = not _off(args.get("forbid_secrets", True))
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
        if scan_ph and size <= TEXT_LIMIT and p.suffix not in CODE_SUFFIXES:
            _scan_placeholders(p, rel, problems)
        if scan_sec:
            _scan_secrets(p, rel, problems, notes)
    return problems, notes


def _numeric(root, spec, problems, notes):
    """Every file the glob matches must hold a finite number at `key` within the
    bounds (as json_files checks every match): a glob is not "any one of"."""
    f, key = spec.get("file"), spec.get("key")
    hits = [p for p in _matches(root, str(f)) if p.is_file()]
    if not hits:
        problems.append(f"numeric check: no file matches {f!r}")
        return
    lo, hi = _float(spec.get("min")), _float(spec.get("max"))
    bad = [b for b in ("min", "max") if spec.get(b) is not None and _float(spec[b]) is None]
    if bad:
        problems.append(f"numeric check {key!r}: {'/'.join(bad)} is not a finite number")
        return
    for p in hits:
        rel = p.relative_to(root)
        try:
            doc, _ = _parse(p)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"numeric check: {rel} does not parse: {exc}")
            continue
        found, value = _dig(doc, key)
        value = _float(value) if found else None
        if value is None:
            problems.append(
                f"numeric check: {rel} has no finite number at {key!r} "
                f"(NaN/inf is a failed measurement; narrow the glob if {rel} is not a result)"
            )
        elif (lo is not None and value < lo) or (hi is not None and value > hi):
            problems.append(
                _because(f"{rel}: {key} = {value} is outside [{lo}, {hi}]", spec.get("why"))
            )
        else:
            notes.append(f"{rel}: {key} = {value} within [{lo}, {hi}]")


def _scan_placeholders(p, rel, problems):
    """Prose still holding template text is unfilled. A report that *quotes* a
    placeholder (in `inline code` or a fence) is describing it, not leaving one."""
    try:
        text = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return
    for n, line in _unfenced(text):
        if PLACEHOLDER.search(re.sub(r"`[^`]*`", "", line)):
            problems.append(f"{rel}:{n} still holds template text: {line.strip()[:80]!r}")


def _scan_text(p):
    """(text, kind, cut) — what the secret scan reads of `p`.

    gzip / bz2 / xz / zip are decompressed (so a gzipped log is scanned, not
    skipped); anything else is read as bytes. At most SCAN_LIMIT bytes are
    read; `cut` says there was more. `kind` is "text", "binary", "archive"
    (text inside) or "archive-binary". Of binary content only the printable
    runs (as `strings` would show them) are scanned."""
    with open(p, "rb") as fh:
        head = fh.read(6)
    opener = next((o for magic, o in _OPENERS if head.startswith(magic)), None)
    kind = "archive"
    if opener is not None:
        with opener(p, "rb") as fh:
            data = fh.read(SCAN_LIMIT + 1)
    elif head[:4] == b"PK\x03\x04":
        parts, left = [], SCAN_LIMIT + 1
        with zipfile.ZipFile(p) as zf:
            for info in zf.infolist():
                if left <= 0:
                    break
                with zf.open(info) as fh:
                    chunk = fh.read(left)
                parts.append(chunk + b"\n")
                left -= len(chunk) + 1
        data = b"".join(parts)
    else:
        with open(p, "rb") as fh:
            data = fh.read(SCAN_LIMIT + 1)
        kind = "text"
    cut = len(data) > SCAN_LIMIT
    data = data[:SCAN_LIMIT]
    try:
        text = codecs.getincrementaldecoder("utf-8")().decode(data, final=not cut)
    except UnicodeDecodeError:
        text = "\n".join(r.decode("ascii") for r in _PRINTABLE.findall(data))
        kind = "binary" if kind == "text" else "archive-binary"
    return text, kind, cut


def _scan_secrets(p, rel, problems, notes):
    try:
        text, kind, cut = _scan_text(p)
    except Exception as exc:  # noqa: BLE001 — a corrupt archive is unscannable, not clean
        problems.append(
            f"{rel} could not be read to scan it for secrets ({exc}) — fix or drop the file"
        )
        return
    where = f"{rel} (decompressed)" if kind.startswith("archive") else str(rel)
    for n, line in enumerate(text.splitlines(), 1):
        if any(not _is_reference(m) for pat in SECRETS for m in pat.finditer(line)):
            problems.append(
                f"{where}:{n} carries what looks like a secret value — "
                "replace it with the variable's NAME and where it comes from"
            )
    if cut and kind.endswith("binary"):
        notes.append(
            f"{where}: binary; only its first {SCAN_LIMIT >> 20} MB were scanned for secrets"
        )
    elif cut:
        problems.append(
            f"{where} holds more than {SCAN_LIMIT >> 20} MB of text, more than the secret scan "
            "reads — trim it, or reference it by path + sha256 instead of carrying it"
        )


def main():
    ids = _load("inputs.json", [])
    args = _load("args.json", {})
    materials = _load("materials.json", {})
    verdict, report = {}, []
    for hid in ids:
        try:
            staged = materials.get(hid)
            if not staged or not Path(staged).is_dir():
                verdict[hid] = False
                report.append(
                    f"{hid}: REFUSED\n  PROBLEM: no staged content (materials.json has {staged!r})"
                )
                continue
            problems, notes = check(Path(staged), args)
            verdict[hid] = not problems
            report.append(f"{hid}: {'passed' if not problems else 'REFUSED'}")
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
