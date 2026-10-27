"""skillkit's checks: the secret patterns (calibrated on a corpus both ways),
the generic validator's numeric / compressed-file handling, scaffold's one-sink
rule, and the digest's Workflow side files."""

from __future__ import annotations

import bz2
import gzip
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

AGENT_SYS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_SYS / "tools" / "skill" / "lib"))

from skillkit import history, pkg, scaffold, secretscan  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "handoff_check_under_test",
    AGENT_SYS / "tools" / "skill" / "lib" / "skillkit" / "templates" / "handoff_check.py",
)
handoff_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(handoff_check)

#: Ordinary code / config / log lines: references, identifiers, constants, flags, usage placeholders.
NOT_SECRETS = """\
api_key = server_args.api_key
api_key = agent_cfg.llm.api_key
self.api_key = api_key
per_token = per_token_per_layer * cfg.num_layers
token = tok.encode(x)
kv_bytes_per_token=kv_bytes_per_token
"prefill_rate_us_per_token": "prefill_rate_us_per_token"
access_token=self._access_token
token = get_token(cfg)
token = getToken(cfg)
export HF_TOKEN=${HF_TOKEN}
--hf-token $HF_TOKEN
token=$HF_TOKEN
k=os.environ["K"]
API_KEY = os.environ['API_KEY']
OPENAI_KEY: ${{ secrets.OPENAI_KEY }}
token = "M4_FRAGMENT_FIRST_CALL"
max_tokens: 2000
total_tokens=1234567890
--max-new-tokens 512
initial-cluster-token: etcd-cluster
commit 1bd905fe3a2b7c1d4e5f6071829a0b1c2d3e4f5a
cache_key = 5d41402abc4b2a76b9719d911017c592
auth_file: /etc/auth/credentials.json
token_url: https://auth.example.com/oauth2/token
docker run -u 1000:1000 img
docker login -u bob --password-stdin
sshpass -p "$SSH_PW" ssh host
curl -u "$USER:$PASS" http://x
the password is required for login
password: <REDACTED>
auth = HTTPBasicAuth(user, pw)
secret_name: my-secret-v1
api_key: os.environ/OPENAI_API_KEY
token="__team_scope_auth_check__"
SERVER_DEFAULT_RESULT_STORAGE_CONFIGURATION_KEY = "server-default-result-storage"
LABELS_TRACEPARENT_KEY = "__OTEL_TRACEPARENT"
self.auth: MultiDomainBasicAuth = MultiDomainBasicAuth(index_urls=index_urls)
MODEL_KEY=layers_0_attn
token = self.tok_v2.encode(x)
curl -u USER:PASSWORD http://x
curl -u user:pass http://x
sshpass -p PASSWORD ssh host
docker login -u USER -p MY_PASSWORD registry.io""".splitlines()

#: Real-looking credentials, including identifier-shaped values (digits inside a part).
SECRETS = (
    """\
sshpass -p 'Hunter2Pass99' ssh host
sshpass -p HUNTER2PASS ssh host
curl -u admin:Sup3rS3cretPw http://x
curl $URL -u admin:Sup3rS3cretPw
curl -u admin:admin http://localhost:9200
docker login -u bob -p dckr_pat_AbCdEfGh12345678
docker login -u bob -p S3cretPassw0rd registry.io
machine host login bob password s3cretPassw0rd
export AZURE_OPENAI_KEY=0123456789abcdef0123456789abcdef
export MY_API_KEY=9F3A2B7C1D4E5F6071829A
DB_PASS=Tr0ub4dor3xyz
export HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz12
OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwx
GH=ghp_abcdefghijklmnopqrstuvwxyz
Authorization: Bearer abcdefghijklmnopqrstuv
password=hunter2hunter2
token=$HF_TOKEN password: correct-horse-battery
password: correct-horse-battery
token = 'hunter2hunter2'
x-api-key: AbCd1234EfGh5678
--api-key abcdef123456
"Ocp-Apim-Subscription-Key": "abcd1234efgh"
AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.SflKxwRJSMeKKF2QT4fwpM
https://bob:hunter22@example.com/x
password = "my_secret_2024"
api_key = "key_8f14e45fceea167a5a36dedd4bea2543"
export OPENAI_API_KEY=sk_live_abc123def456ghi789
secret: s3cr3t_p4ss_w0rd
DB_PASSWORD=admin_2024_prod
export HF_TOKEN=__init__abc123
auth: user1.pass2word
access_token = ya29.a0AfH6SMBx3kq9Lp2Zr7Tq""".splitlines()
    + [
        "-----BEGIN OPENSSH " + "PRIVATE KEY-----",
        # Dotted, but a token. Built at runtime so the host's push protection does not take it for a real one.
        'token = "' + "MTk4NjIyNDgzNDcxOTI1MjQ4" + ".Cl2FMQ." + "ZnCjm1XVW7vRze4b7Cq4se7kKWs" + '"',
    ]
)


@pytest.mark.parametrize("line", NOT_SECRETS + SECRETS)
def test_secret_corpus_and_the_validators_copy_agree(line):
    secret = line in SECRETS
    assert secretscan.looks_secret(line) == secret
    assert (secretscan.redact(line) != line) == secret
    # handoff_check carries its own copy of the patterns: same list, same verdict.
    copy = any(
        not handoff_check._is_reference(m) for p in handoff_check.SECRETS for m in p.finditer(line)
    )
    assert handoff_check.PATTERNS == secretscan.PATTERNS and copy == secret


def _content(tmp_path: Path, files: dict[str, bytes | str]) -> Path:
    root = tmp_path / "content"
    (root / "items").mkdir(parents=True)
    (root / "README.md").write_text("# r\n\n## Purpose\nA result.\n")
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        (p.write_bytes if isinstance(body, bytes) else p.write_text)(body)
    return root


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in members.items():
            zf.writestr(name, body)
    return buf.getvalue()


LEAK = b"started\nexport HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz12\n"
BLOB = bytes(range(256))


@pytest.mark.parametrize(
    "name,pack",
    [
        ("server.log.gz", gzip.compress),
        ("server.log.bz2", bz2.compress),
        ("logs.zip", lambda b: _zip({"a.log": b})),
        ("latin1.log", lambda b: b"\xff\xfe junk\n" + b),
        ("a.bin", lambda b: BLOB + b + b"\x00" + BLOB),
        ("b.tar.gz", lambda b: gzip.compress(BLOB + b + b"\x00" + BLOB)),
    ],
)
def test_secret_scan_reads_compressed_non_utf8_and_binary_files(tmp_path, name, pack):
    problems, _ = handoff_check.check(_content(tmp_path, {f"items/logs/{name}": pack(LEAK)}), {})
    assert any(name in p and "secret" in p for p in problems), problems
    clean = pack(LEAK.replace(b"hf_abcdefghijklmnopqrstuvwxyz12", b"$HF_TOKEN"))
    assert handoff_check.check(_content(tmp_path / "c", {f"items/logs/{name}": clean}), {})[0] == []


@pytest.mark.parametrize(
    "name,body,expect",
    [
        ("x.log.gz", b"\x1f\x8bnot gzip at all", "problem:could not be read to scan"),
        ("big.log.gz", gzip.compress(b"ok line\n" * 100), "problem:more than the secret scan"),
        ("image.tar.gz", gzip.compress(BLOB * 4), "note:binary"),
        ("lib.whl", _zip({"lib.so": BLOB * 4}), "note:binary"),
    ],
)
def test_unscannable_files_are_refused_and_big_binaries_noted(
    tmp_path, monkeypatch, name, body, expect
):
    # Text beyond the scan limit is refused as unscanned; a binary only gets a note.
    monkeypatch.setattr(handoff_check, "SCAN_LIMIT", 256)
    problems, notes = handoff_check.check(_content(tmp_path, {f"items/{name}": body}), {})
    where, text = expect.split(":")
    assert any(name in m and text in m for m in (problems if where == "problem" else notes))
    assert where == "problem" or problems == []


def test_numeric_needs_a_finite_number_in_every_matching_file(tmp_path):
    root = _content(
        tmp_path,
        {
            "items/result/a.json": json.dumps({"error_rate": 0.0}),
            "items/result/b.json": json.dumps({"error_rate": 0.5}),
            "items/result/c.json": '{"error_rate": NaN, "ok": Infinity}',
        },
    )
    spec = {"file": "items/result/*.json", "key": "error_rate", "max": 0.05}
    args = {"numeric": [spec], "json_files": {"items/result/c.json": ["ok"]}}
    problems, notes = handoff_check.check(root, args)
    assert any("b.json" in p and "outside" in p for p in problems), problems
    assert any("c.json" in p and "no finite number" in p for p in problems), problems
    assert any("no value at 'ok'" in p for p in problems), problems
    assert [n for n in notes if "within" in n] == [
        "items/result/a.json: error_rate = 0.0 within [None, 0.05]"
    ]


def _two_task_plan(**b_extra) -> dict:
    task = {"kind": "ai", "readme": "# t"}
    return {
        "package": "par",
        "goal": "two tasks",
        "tasks": [
            {**task, "name": "a", "goal": "do a", "outputs": ["ka"]},
            {**task, "name": "b", "goal": "do b", "outputs": ["kb"], **b_extra},
        ],
        "kinds": [{"name": k, "content_type": "text"} for k in ("ka", "kb")],
    }


@pytest.mark.parametrize(
    "join,refusal",
    [
        ({}, r"\{'a': \['ka'\]\}.*'b'.*consume"),
        ({"froms": ["a"]}, "does not export"),  # froms orders a before b but exports nothing
        ({"inputs": ["ka"]}, None),
        ({"inputs": ["ka"], "froms": ["a"]}, None),
    ],
)
def test_scaffold_allows_one_sink_only(tmp_path, join, refusal):
    dest = tmp_path / "p"
    if refusal:
        with pytest.raises(scaffold.PlanError, match=refusal):
            scaffold.scaffold(_two_task_plan(**join), dest)
        assert not dest.exists()
    else:
        scaffold.scaffold(_two_task_plan(**join), dest)
        assert pkg.show(dest)[0] == 0


@pytest.mark.parametrize("present,expect", [(True, "return 42"), (False, "NOT captured")])
def test_digest_reads_a_scriptpath_workflow_at_digest_time(tmp_path, present, expect):
    if present:
        (tmp_path / "wf.js").write_text("export default async function run() { return 42 }\n")
    rec = {
        "type": "assistant",
        "timestamp": "2026-10-01T01:00:00Z",
        "cwd": str(tmp_path),
        "message": {
            "content": [{"type": "tool_use", "name": "Workflow", "input": {"scriptPath": "wf.js"}}]
        },
    }
    (tmp_path / "s.jsonl").write_text(json.dumps(rec) + "\n")
    d = history.digest(tmp_path / "s.jsonl", history.DigestOptions())
    assert (
        len(d.workflows) == 1 and expect in d.workflows[0] and "scriptPath wf.js" in d.workflows[0]
    )
