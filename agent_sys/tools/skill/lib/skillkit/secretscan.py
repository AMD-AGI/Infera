"""What a secret looks like, for redaction (history digests) and refusal (audit,
the generic validator).

The aim is the realistic cases — vendor tokens with a known prefix, a value
assigned to a variable or flag whose name says it is a credential, credentials
in a URL, a private key — without flagging `max_tokens: 2000`. A *reference*
to a secret (`$HF_TOKEN`, `os.environ["K"]`, `<REDACTED>`) is what a handoff
should contain and is never flagged.

`templates/handoff_check.py` carries a copy of `PATTERNS` (it must run with
no import from here); `tests/tools/test_skillkit.py` keeps the two equal.
"""

from __future__ import annotations

import re

#: A value: 8+ non-delimiter characters, at least one a letter.
_VALUE = r"(?=[^\s\"',;}{]*[A-Za-z])[^\s\"',;}{]{8,}"
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

COMPILED = [(re.compile(p), r) for p, r in PATTERNS]
_PEM_BLOCK = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
)
_REFERENCE = re.compile(r"[$(\[<{]|REDACTED")


def redact(text: str) -> str:
    """Replace literal secrets; leave references (`$HF_TOKEN`) readable."""
    text = _PEM_BLOCK.sub("<REDACTED-PRIVATE-KEY>", text)
    for pat, rep in COMPILED:
        text = pat.sub(lambda m, rep=rep: m.group(0) if _is_reference(m) else m.expand(rep), text)
    return text


def _is_reference(m: re.Match) -> bool:
    return bool(_REFERENCE.search(m.group(0))) and not m.group(0).startswith("-----BEGIN")


def looks_secret(line: str) -> bool:
    """A pattern hit whose matched text is a literal, not a reference."""
    for pat, _ in COMPILED:
        for m in pat.finditer(line):
            if not _is_reference(m):
                return True
    return False
