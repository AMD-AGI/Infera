"""What a secret looks like, for redaction (history digests) and refusal (audit,
the generic validator).

The aim is the realistic cases — vendor tokens with a known prefix, a value
assigned to a variable or flag whose name says it is a credential, credentials
in a URL, a private key — without flagging `max_tokens: 2000`. A *reference*
to a secret (`$HF_TOKEN`, `os.environ["K"]`, `<REDACTED>`) is what a handoff
should contain and is never flagged.

`templates/handoff_check.py` carries a copy of `PATTERNS` (it must run with
no import from here); `tests/tools/test_skillkit_checks.py` keeps the two equal.
"""

from __future__ import annotations

import re

#: Where a bare value ends.
_END = r"(?:[\s\"',;}{()\[\]]|$)"
#: A value: not a flag, a path or a URL; has a letter; either a digit (8+ chars)
#: or 16+ chars; and not shaped like a *reference* — a dotted path or call
#: (`server_args.api_key`, `tok.encode(x)`), a lower_snake identifier
#: (`kv_bytes_per_token=kv_bytes_per_token`), an UPPER_SNAKE constant name
#: (`"M4_FRAGMENT_FIRST_CALL"`) or a letters-only word (`MultiDomainBasicAuth`,
#: a type annotation). In a dotted path or lower_snake identifier each part is
#: letters with at most two trailing digits (`cfg.v2`, `layers_0_attn`), so
#: `admin_2024_prod`, `s3cr3t_p4ss` and dotted tokens (`MTk4Nj...Cl2FMQ.ZnCj...`)
#: are values. A random-looking value is none of those, so
#: `9F3A2B7C1D4E5F6071829A` (uppercase hex, no underscore) still is one, while
#: `initial-cluster-token: etcd-cluster` (no digit, < 16 chars) is not.
#: One part of a dotted path: an identifier, letters then <= 2 digits.
_PART = r"[A-Za-z_][A-Za-z_]*[0-9]{0,2}"
_VALUE = (
    r"(?!-)(?!~?\.{0,2}/)(?![a-z][a-z0-9+.-]*://)"
    rf"(?!{_PART}(?:\.{_PART})+(?:/|{_END}))"  # os.environ/NAME
    rf"(?!_*[a-z]+[0-9]{{0,2}}(?:_+(?:[a-z]+[0-9]{{0,2}}|[0-9]{{1,2}}))+_*{_END})"
    rf"(?!_*[A-Z][A-Z0-9]*(?:_+[A-Z0-9]+)+{_END})(?!_*[A-Za-z]+{_END})"
    r"(?=[^\s\"',;}{]*[A-Za-z])(?:(?=[^\s\"',;}{]*\d)[^\s\"',;}{]{8,}|[^\s\"',;}{]{16,})"
)
#: A credential-ish name at the END of an identifier (so `max_tokens` is not one).
_NAME = (
    r"(?i:[A-Za-z0-9_.-]*(?:api[_-]?key|access[_-]?key|secret(?:[_-]?key)?|token|password"
    r"|passwd|subscription[_-]?key|auth))"
)
#: `*_KEY` / `*_PASS` / `*_PWD` in env-variable case only (`AZURE_OPENAI_KEY`;
#: lower-case `cache_key` is everyday code), and only with a digit in the value
#: (`CONFIGURATION_KEY = "server-default-result-storage"` is a name, not a key).
_ENV_NAME = r"[A-Z][A-Z0-9_]*_(?:KEY|PASS|PWD)"

#: A password given to a command (`sshpass -p`, `curl -u user:pw`, `docker
#: login -p`): not a usage placeholder — an UPPER word (`USER:PASSWORD`,
#: `MY_PASS`) or a stand-in word (`user:pass`, `xxx`, `***`).
_ARG = (
    r"(?![A-Z]+(?:_[A-Z0-9]+)*(?:[\s\"']|$))"
    r"(?!(?i:pass(?:word|wd)?|pwd?|secret|token|x{3,}|\*+)(?:[\s\"']|$))[^\s\"']{3,}"
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
    (r"\bdckr_pat_[A-Za-z0-9_\-]{10,}", "<REDACTED-DOCKER-TOKEN>"),
    (r"\bAKIA[0-9A-Z]{16}\b", "<REDACTED-AWS-KEY>"),
    (r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}", "<REDACTED-JWT>"),
    (r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}", r"\1<REDACTED>"),
    (r"(?i)(basic\s+)[A-Za-z0-9+/]{16,}={0,2}", r"\1<REDACTED>"),
    (r"([a-z][a-z0-9+.-]*://[^\s:/@]+:)[^\s@/]{3,}(@)", r"\1<REDACTED>\2"),
    # Passwords given to a command as an argument.
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
    """Judged on the VALUE only (after group 1), so a curl `-u` user:password pair is
    still a literal password even though the command line holds a `$`."""
    if m.group(0).startswith("-----BEGIN"):
        return False
    value = m.group(0)[m.end(1) - m.start(0) :] if m.re.groups else m.group(0)
    return bool(_REFERENCE.search(value))


def looks_secret(line: str) -> bool:
    """A pattern hit whose matched text is a literal, not a reference."""
    for pat, _ in COMPILED:
        for m in pat.finditer(line):
            if not _is_reference(m):
                return True
    return False
