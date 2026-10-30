"""What backup publishes is what was audited, and no secret is published
silently — against a local bare repo (no network)."""

from __future__ import annotations

import bz2
import gzip
import lzma
import os
import re
import subprocess
import tracemalloc

import pytest

from tests.tools import test_skillkit_remote as base  # also puts skillkit on sys.path
from tests.tools.test_skillkit_remote import HELLO, handoff, remote_files

# isort: split
from skillkit import audit, remote, secretscan

env, repo = base.env, base.repo  # the fixtures

SECRET = "HF_TOKEN=hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2"
LINE = f"export {SECRET}\n".encode()


def _audit(rel: str, data: bytes) -> audit.Audit:
    a = audit.Audit(root="x")
    audit.scan(a, rel, data)
    return a


@pytest.mark.parametrize(
    "fname,data",
    [
        ("run.log", b"start\n" + LINE),
        ("run.log.gz", gzip.compress(LINE)),
        ("run.log.bz2", bz2.compress(LINE)),
        ("run.log.xz", lzma.compress(LINE)),
        ("nested.gz", gzip.compress(gzip.compress(LINE))),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_secrets_are_found_also_compressed(repo, tmp_path, fname, data):
    c = handoff(tmp_path)
    (c / "items" / fname).write_bytes(data)
    with pytest.raises(remote.Refused, match="secret-looking value"):
        remote.backup(repo, c.parent, name="x", allow_big=True, allow_unscanned=True)
    assert not any("/x/" in f for f in remote_files(repo))


@pytest.mark.parametrize(
    "fname,data,why",
    [
        ("a.zst", b"(\xb5/\xfd" + os.urandom(64), "zstd"),
        ("bad.gz", b"\x1f\x8b garbage", "decompress"),
    ],
    ids=lambda v: v if isinstance(v, str) and "." in v else "",
)
def test_unscannable_needs_its_own_consent(repo, tmp_path, fname, data, why):
    c = handoff(tmp_path)
    (c / "items" / fname).write_bytes(data)
    a = audit.audit(c)
    assert not a.secrets and why in a.unscanned[0] and any(fname in q for q in a.ask_user)
    with pytest.raises(remote.Refused, match="not scannable"):
        remote.backup(repo, c.parent, name="z", allow_big=True)  # a size flag is not consent
    assert remote.backup(repo, c.parent, name="z", allow_unscanned=True)["relative"].endswith("/z")


def test_binary_string_tables_ask_and_utf16_is_decoded(repo, tmp_path):
    """NULs never become part of a value; a hit in a binary (a string table as
    often as a secret) is a question; UTF-16 text (PowerShell `>`) blocks."""
    noise = b"\x7fELF\x00\x00password: \x00\x00\x00Incorrect\x00auth=\x00\xff\xfe\x00" * 50
    a = _audit("tool", noise)
    assert not a.secrets and not a.binary_hits
    table = b"\x00\x01sk-ssh-ed25519-cert-v01@openssh.com\x00\x02" * 3
    a = _audit("ssh-add", table)
    assert not a.secrets and a.binary_hits and any("ssh-add" in q for q in a.ask_user)
    for enc in ("utf-16", "utf-16-le", "utf-16-be"):
        assert _audit("r.txt", ("x\n" + LINE.decode()).encode(enc)).secrets == ["r.txt:2"]
    c = handoff(tmp_path)
    (c / "items" / "ssh-add").write_bytes(table)
    with pytest.raises(remote.Refused, match="binary"):
        remote.backup(repo, c.parent, name="b", allow_big=True)
    (c / "items" / "r.txt").write_bytes(LINE.decode().encode("utf-16"))
    with pytest.raises(remote.Refused, match="secret-looking value"):
        remote.backup(repo, c.parent, name="b", allow_unscanned=True)


def test_decompression_bomb_is_unscanned(monkeypatch):
    monkeypatch.setattr(audit, "DECOMPRESS_LIMIT", 1000)
    assert "over" in _audit("bomb.gz", gzip.compress(b"a" * 5000)).unscanned[0]


def test_big_files_are_scanned_and_need_their_own_flag(repo, tmp_path):
    c = handoff(tmp_path)
    big = c / "items" / "big.log"
    big.write_bytes(LINE + b"a" * (5 << 20))
    with pytest.raises(remote.Refused, match="secret"):
        remote.backup(repo, c.parent, name="g", allow_big=True, allow_unscanned=True)
    big.write_bytes(b"step=1 loss=0.5\n" * (400 << 10))  # 6.4 MB, clean
    with pytest.raises(remote.Refused, match="over 4 MB"):
        remote.backup(repo, c.parent, name="g", allow_unscanned=True)
    assert remote.backup(repo, c.parent, name="g", allow_big=True)["relative"].endswith("/g")


def test_task_backup_audits_symlink_targets(repo, tmp_path):
    pkg = tmp_path / "pkg"
    subprocess.run(["cp", "-r", str(HELLO), str(pkg)], check=True)
    (tmp_path / "outside.env").write_bytes(LINE)
    (pkg / "assets" / "creds.env").symlink_to(tmp_path / "outside.env")
    with pytest.raises(remote.Refused, match="secret"):
        remote.backup(repo, pkg)


def test_sealed_records_are_audited(repo, tmp_path):
    c = handoff(tmp_path)
    (c.parent / "manifest.yaml").write_text("kind: x\n")
    record = c.parent / "validation.yaml"
    record.write_text(
        "log: |\n  curl -H 'Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345'\n"
    )
    with pytest.raises(remote.Refused, match="validation.yaml"):
        remote.backup(repo, c.parent, name="val")
    record.write_text("message: refuses [More Information Needed]\n")  # quoted, not unfilled
    out = remote.backup(repo, c.parent, name="val")
    assert out["sealed"] is True and out["handoff_kind"] == "x"
    assert "alice/draft/handoffs/val/validation.yaml" in remote_files(repo)


def test_publishes_the_bytes_that_were_audited(repo, tmp_path, monkeypatch):
    log = handoff(tmp_path) / "items" / "run.log"
    log.write_text("clean\n")
    real_publish = remote._publish

    def late_write(cfg, build, **kw):
        log.write_bytes(b"clean\n" + LINE)  # appended after the audit
        return real_publish(cfg, build, **kw)

    monkeypatch.setattr(remote, "_publish", late_write)
    out = remote.backup(repo, log.parents[2], name="t")
    show = ["git", "-C", repo.url, "show", f"main:{out['relative']}/content/items/run.log"]
    shown = subprocess.check_output(show, text=True)
    assert shown == "clean\n"


def test_handoff_changing_while_read_is_refused(repo, tmp_path, monkeypatch):
    log = handoff(tmp_path) / "items" / "run.log"
    log.write_text("a\n")
    real = remote._source_files

    def racing(src, kind):
        out = real(src, kind)
        log.write_text("b\n")
        return out

    monkeypatch.setattr(remote, "_source_files", racing)
    with pytest.raises(remote.RemoteError, match="changed while"):
        remote.backup(repo, log.parents[2], name="r")


def test_backup_streams_instead_of_holding_every_file(repo, tmp_path):
    c = handoff(tmp_path)
    for i in range(3):
        (c / "items" / f"w{i}.log").write_bytes(b"step=1 loss=0.5\n" * (1 << 20))  # 16 MB each
    tracemalloc.start()
    try:
        remote.backup(repo, c.parent, name="m", allow_big=True)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 12 << 20, peak  # reading every file first held 48 MB


@pytest.mark.parametrize("cut", range(0, 40, 7))
def test_secret_at_a_chunk_boundary_and_in_long_lines(monkeypatch, cut):
    monkeypatch.setattr(audit, "CHUNK", 4096)
    data = b"z" * (4096 * 3 - cut) + f" {SECRET}\n".encode() + b"q" * 9000
    assert _audit("f.txt", data).secrets == ["f.txt:1"]
    line = "token " * 5000 + "x" * (audit._PIECE - 10 + cut) + f" {SECRET} " + "y" * 50000
    assert _audit("one.json", line.encode()).secrets == ["one.json:1"]


#: One sample per secretscan pattern, in order: the audit's hint pre-filter
#: must let each through. A new pattern needs a sample here (and maybe a hint).
SAMPLES = [
    "-----BEGIN RSA " + "PRIVATE KEY-----",  # split: push protection
    "sk-ant-abcdefgh1234",
    "sk-abcdefghijklmnopqrstu1",
    "ghp_abcdefghijklmnop1234",
    "glpat-abcdefghijklmnop12",
    "xoxb-1234567890abc",
    "hf_A1b2C3d4E5f6G7h8I9j0K1l2",
    "dckr_pat_abcdefghij12",
    "AKIAABCDEFGHIJKLMNOP",
    "eyJhbGciOiJI.eyJzdWIiOiIx.abcdefgh1234",
    "Bearer abcdefghijklmnop1234",
    "Basic dXNlcjpwYXNzd29yZDEyMzQ=",
    "https://bob:hunter2@x.org",
    "sshpass -p hunter22 ssh host",
    "curl -u bob:hunter22 http://x",
    "docker login -p hunter22 reg",
    "machine x login bob password Hunt3rPass99",
    "password=Hunt3rPass99",
    "MY_DB_PWD=Hunt3r99xyz",
    "--api-key Hunt3rPass99",
]


def test_hint_prefilter_covers_every_secret_pattern():
    assert len(SAMPLES) == len(secretscan.PATTERNS)
    for (pat, _rep), sample in zip(secretscan.PATTERNS, SAMPLES):
        assert re.search(pat, sample), (pat, sample)
        assert _audit("s.txt", f"noise\n  {sample}\nmore\n".encode()).secrets == ["s.txt:2"]
