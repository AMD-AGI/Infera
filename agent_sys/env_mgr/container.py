# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Container lifecycle management for agent_sys.

Provides the ``--docker`` CLI feature: build the control-plane image when it is
absent, start a container with the right mounts, and forward the command into
it.  Host SSH keys and Claude credentials are detected and bind-mounted at
runtime (never copied into the image).

This module lives above the decoupling wall: it imports nothing from the
shipped installer subsystem and nothing from the isolation / domain subsystem.
"""

from __future__ import annotations

import grp
import logging
import os
import pwd
import shutil
import subprocess
import tempfile
from pathlib import Path

__all__ = ["ContainerManager"]

log = logging.getLogger(__name__)

DEFAULT_IMAGE = "infera/agent-sys:latest"
DEFAULT_CONTAINER_NAME = "agent-sys-container"


class ContainerManager:
    """Build, start, and exec into an agent_sys Docker container.

    Parameters
    ----------
    image : str
        Docker image tag.
    container_name : str
        Name for the container.
    detect_ssh : bool
        If True, detect and mount host ``~/.ssh`` read-only.
    detect_claude : bool
        If True, detect and mount host ``~/.claude``.
    """

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        container_name: str = DEFAULT_CONTAINER_NAME,
        detect_ssh: bool = True,
        detect_claude: bool = True,
    ) -> None:
        self.image = image
        self.container_name = container_name
        self.detect_ssh = detect_ssh
        self.detect_claude = detect_claude

    # ---------------------------------------------------------------------- #

    @staticmethod
    def _docker_bin() -> str:
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError(
                "docker is not installed or not on PATH.  "
                "Install Docker and try again, or run without --docker."
            )
        return docker

    def image_exists(self) -> bool:
        result = subprocess.run(
            [self._docker_bin(), "image", "inspect", self.image],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def container_running(self) -> bool:
        result = subprocess.run(
            [
                self._docker_bin(),
                "container",
                "inspect",
                "-f",
                "{{.State.Running}}",
                self.container_name,
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    # ---------------------------------------------------------------------- #

    def build(self, repo_root: str | Path | None = None) -> None:
        """Build the control-plane image from the repository Dockerfile."""
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[2]
        repo_root = Path(repo_root)
        dockerfile = repo_root / "deploy" / "docker" / "Dockerfile.agent-sys"
        if not dockerfile.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile}")
        log.info("building image %s from %s", self.image, repo_root)
        subprocess.run(
            [
                self._docker_bin(),
                "build",
                "--file",
                str(dockerfile),
                "--tag",
                self.image,
                "--progress=plain",
                str(repo_root),
            ],
            check=True,
        )

    # ---------------------------------------------------------------------- #

    def start(self, *, workroot: str | None = None) -> None:
        """Start a background container, detecting host config as requested.

        If the container is already running, this is a no-op.  If the image
        does not exist, it is built first.
        """
        if self.container_running():
            log.info("container %s is already running", self.container_name)
            return

        if not self.image_exists():
            log.info("image %s not found, building it", self.image)
            self.build()

        docker = self._docker_bin()
        home = os.environ.get("HOME", os.path.expanduser("~"))
        container_home = "/home/agent"
        uid = os.getuid()
        gid = os.getgid()

        if workroot is None:
            workroot = os.environ.get(
                "INFERA_AGENT_SYSTEM_WORKROOT",
                os.path.join(home, ".agent_sys_runs"),
            )
        os.makedirs(workroot, exist_ok=True)

        # A passwd/group entry so ssh's getpwuid() does not refuse.
        passwd_dir = os.path.join(home, ".agent_sys")
        os.makedirs(passwd_dir, exist_ok=True)
        passwd_file = os.path.join(passwd_dir, "passwd")
        group_file = os.path.join(passwd_dir, "group")

        try:
            username = pwd.getpwuid(uid).pw_name
        except KeyError:
            username = "agent"
        try:
            groupname = grp.getgrgid(gid).gr_name
        except KeyError:
            groupname = "agent"

        with open(passwd_file, "w") as f:
            f.write(f"root:x:0:0:root:/root:/bin/bash\n")
            f.write(f"{username}:x:{uid}:{gid}:agent:{container_home}:/bin/bash\n")
        with open(group_file, "w") as f:
            f.write(f"root:x:0:\n{groupname}:x:{gid}:\n")

        cmd: list[str] = [
            docker,
            "run",
            "-d",
            "--name",
            self.container_name,
            "--user",
            f"{uid}:{gid}",
            "--network",
            "host",
            "-v",
            f"{passwd_file}:/etc/passwd:ro",
            "-v",
            f"{group_file}:/etc/group:ro",
            "-e",
            f"HOME={container_home}",
            "-e",
            "GIT_SSH_COMMAND=ssh -o StrictHostKeyChecking=accept-new",
            "-e",
            f"AGENT_SYS_NO_PERMISSIONS={os.environ.get('AGENT_SYS_NO_PERMISSIONS', '1')}",
            "-e",
            "AGENT_SYS_REPO=/opt/Infera",
            "-e",
            f"INFERA_AGENT_SYSTEM_WORKROOT={workroot}",
            "-v",
            f"{workroot}:{workroot}",
        ]

        if self.detect_ssh:
            ssh_dir = os.path.join(home, ".ssh")
            if os.path.isdir(ssh_dir):
                cmd += ["-v", f"{ssh_dir}:{container_home}/.ssh:ro"]
                log.info("mounting host SSH config from %s", ssh_dir)
            else:
                log.info("no ~/.ssh directory found, skipping SSH mount")

        if self.detect_claude:
            claude_dir = os.path.join(home, ".claude")
            if os.path.isdir(claude_dir):
                cmd += ["-v", f"{claude_dir}:{container_home}/.claude"]
                log.info("mounting host Claude config from %s", claude_dir)
            else:
                log.info("no ~/.claude directory found, skipping Claude mount")

        cmd += [self.image, "sleep", "infinity"]

        log.info("starting container %s", self.container_name)
        subprocess.run(cmd, check=True)

    # ---------------------------------------------------------------------- #

    def exec(self, argv: list[str], *, workdir: str | None = None) -> int:
        """Run a command inside the container and return its exit code."""
        cmd = [self._docker_bin(), "exec"]
        if workdir:
            cmd += ["-w", workdir]
        cmd += [self.container_name, *argv]
        result = subprocess.run(cmd)
        return result.returncode

    def stop(self) -> None:
        """Stop and remove the container."""
        subprocess.run(
            [self._docker_bin(), "rm", "-f", self.container_name],
            capture_output=True,
        )
