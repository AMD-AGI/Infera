# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Capture must keep unavailable facts and dirty sources visible."""

import subprocess

from kernel_registry import capture


def test_missing_accelerator_does_not_invent_hardware(monkeypatch):
    monkeypatch.setattr(
        capture,
        "_probe",
        lambda argv: {
            "command": argv,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "error": "not installed",
        },
    )
    record = capture.capture_environment("cpu development")
    assert all(value is None for value in record["hardware"].values())
    assert record["packages"]["pytest"]
    assert record["container"]["digest"] is None
    assert "hardware.driver: not observed" in record["missing"]
    assert record["provenance"]["accelerator_probes"]["rocminfo"]["error"] == "not installed"


def test_supplied_image_is_not_claimed_as_observed(monkeypatch):
    monkeypatch.setattr(
        capture,
        "_probe",
        lambda argv: {
            "command": argv,
            "stdout": "",
            "stderr": "",
            "returncode": 1,
        },
    )
    record = capture.capture_environment(
        "image", image="test:v1", image_digest="sha256:" + "a" * 64
    )
    assert record["container"] == {"image": "test:v1", "digest": "sha256:" + "a" * 64}
    assert "caller-provided" in record["provenance"]["container_identity"]


def test_dirty_repository_is_flagged_and_remote_credentials_are_removed(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "--allow-empty",
        "-m",
        "init",
    )
    git("remote", "add", "origin", "https://user:private-token@example.org/kernel.git?token=secret")
    (tmp_path / "untracked.py").write_text("# Unsaved kernel revision\n")
    record = capture.capture_environment("dirty", repos=[tmp_path])
    repository = record["repositories"][0]
    assert repository["dirty"] is True
    assert repository["patch"] is None
    assert len(repository["commit"]) == 40
    assert repository["url"] == "https://example.org/kernel.git"
    assert "private-token" not in str(record)
    assert any("retained dirty patch" in missing for missing in record["missing"])


def test_timeout_is_recorded_without_raising(monkeypatch):
    def timeout(argv, **kwargs):
        assert kwargs["timeout"] == 10
        assert "shell" not in kwargs
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(capture.subprocess, "run", timeout)
    result = capture._probe(["rocminfo"])
    assert result["returncode"] is None
    assert "timed out" in result["error"]


def test_local_repository_locator_is_never_published(tmp_path, monkeypatch):
    def probe(argv):
        value = ""
        if argv[-3:] == ["remote", "get-url", "origin"]:
            value = str(tmp_path / "private-upstream")
        return {"command": argv, "stdout": value, "stderr": "", "returncode": 0}

    monkeypatch.setattr(capture, "_probe", probe)
    record = capture.capture_environment("portable", repos=[tmp_path])
    assert record["repositories"][0]["url"] is None
    assert record["repositories"][0]["name"] == "source-1"
    assert str(tmp_path) not in str(record)
    assert "repositories.source-1: upstream URL unavailable" in record["missing"]


def test_scp_origin_does_not_publish_login_name():
    assert (
        capture._remote_url("private-user@github.com:org/kernel.git")
        == "ssh://github.com/org/kernel.git"
    )
    assert capture._remote_url("file:///private/source") is None


def test_amd_gpu_inventory_uses_only_observed_names(monkeypatch):
    outputs = {
        "rocminfo": "Name: AMD EPYC\nMarketing Name: CPU must not be GPU model\n"
        "Name: gfx950\nMarketing Name: AMD Instinct MI355X\n"
        "Name: gfx950\nMarketing Name: \nVendor Name: AMD\n",
        "rocm-smi": "Driver version: 6.12.12\n",
    }

    def probe(argv):
        return {
            "command": argv,
            "stdout": outputs.get(argv[0], ""),
            "stderr": "",
            "returncode": 0 if argv[0] in outputs else 1,
        }

    monkeypatch.setattr(capture, "_probe", probe)
    record = capture.capture_environment("gpu")
    assert record["hardware"] == {
        "device": "AMD Instinct MI355X",
        "architecture": "gfx950",
        "count": 2,
        "driver": "6.12.12",
    }
