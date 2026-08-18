# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Installer registry: name -> instance. v1 is a simple dict."""

from __future__ import annotations

from .installers.apt import AptInstaller
from .installers.base import Installer
from .installers.bin import BinInstaller
from .installers.claude import ClaudeInstaller
from .installers.embed import EmbedInstaller
from .installers.oneline import OnelineInstaller
from .installers.uv import UvInstaller

REGISTRY: dict[str, Installer] = {
    "uv": UvInstaller(),
    "apt": AptInstaller(),
    "bin": BinInstaller(),
    "oneline": OnelineInstaller(),
    "embed": EmbedInstaller(),
    "claude": ClaudeInstaller(),
}


def get_installer(name: str) -> Installer:
    try:
        return REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"unknown installer {name!r} (have {sorted(REGISTRY)})") from e
