"""Redact secrets before anything is written to a result directory or a git-tracked file.

Rules (applied to env dicts, command lines and free text):
- an env/flag NAME matching SECRET_NAME_RE has its value replaced by "<redacted>";
- token-shaped values (hf_..., ghp_..., sk-..., github_pat_..., long hex/base64 blobs after key=)
  are replaced wherever they appear;
- URLs with user:password@ lose the credential part.
Pure standard library so it runs on a login node and in CPU tests.
"""

import re

SECRET_NAME_RE = re.compile(r"(TOKEN|SECRET|PASSW|PASSWD|PASSWORD|API[_-]?KEY|ACCESS[_-]?KEY|PRIVATE|CREDENTIAL|AUTH|COOKIE|SESSION)",
                            re.IGNORECASE)
_TOKEN_VALUE_RES = [
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]
_URL_CRED_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@")
_KV_RE = re.compile(r"(?P<key>(?:--)?[A-Za-z0-9_.-]+)(?P<sep>[=\s]+)(?P<val>\"[^\"]*\"|'[^']*'|[^\s]+)")
REDACTED = "<redacted>"


def is_secret_name(name: str) -> bool:
    return bool(SECRET_NAME_RE.search(name))


def redact_text(text: str) -> str:
    out = _URL_CRED_RE.sub(lambda m: m.group("scheme") + REDACTED + "@", text)
    for r in _TOKEN_VALUE_RES:
        out = r.sub(REDACTED, out)

    def _kv(m):
        key = m.group("key").lstrip("-")
        if is_secret_name(key):
            return m.group("key") + m.group("sep") + REDACTED
        return m.group(0)

    return _KV_RE.sub(_kv, out)


def redact_env(env: dict) -> dict:
    return {k: (REDACTED if is_secret_name(k) else redact_text(str(v))) for k, v in sorted(env.items())}


def redact_argv(argv: list) -> list:
    out, hide_next = [], False
    for a in argv:
        s = str(a)
        if hide_next:
            out.append(REDACTED)
            hide_next = False
            continue
        if s.startswith("--") and "=" not in s and is_secret_name(s):
            out.append(s)
            hide_next = True
            continue
        out.append(redact_text(s))
    return out


def redact_docker_inspect(obj):
    """Redact `docker inspect` output: Config.Env entries (K=V), Config.Cmd/Entrypoint/Args, and any free text."""
    items = obj if isinstance(obj, list) else [obj]
    for it in items:
        cfg = it.get("Config") or {}
        env = cfg.get("Env") or []
        out = []
        for e in env:
            k, sep, v = e.partition("=")
            out.append(k + sep + (REDACTED if is_secret_name(k) else redact_text(v)))
        if env:
            cfg["Env"] = out
        for key in ("Cmd", "Entrypoint"):
            if isinstance(cfg.get(key), list):
                cfg[key] = redact_argv(cfg[key])
        if isinstance(it.get("Args"), list):
            it["Args"] = redact_argv(it["Args"])
    return obj


if __name__ == "__main__":
    import json
    import sys
    if sys.argv[1:] == ["--docker-inspect"]:
        json.dump(redact_docker_inspect(json.load(sys.stdin)), sys.stdout, indent=2)
    else:
        sys.stdout.write(redact_text(sys.stdin.read()))
