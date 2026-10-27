"""The command line (cli.py): a AGENT_SYS_HOME made on first use, one server per AGENT_SYS_HOME, `agent-sys-v2 serve` in a tmux of its
own that never restarts another server's window, the socket's directory, --version."""

import subprocess
import sys
import time

import pytest
from conftest import wait_for

from agent_sys_v2 import __version__, cli
from agent_sys_v2.model import load_config
from agent_sys_v2.ops import init_home


def test_init_home_makes_only_what_is_missing(tmp_path):
    home = tmp_path / "home"
    assert init_home(home) == ["projects/", "rules/", "knowledge/entries/", "agent_sys.yaml"]
    assert load_config(home) == {
        "max_running": 4,
        "language": "en",
        "protocol_language": "zh",
        "claude_config": "isolated",
    }
    assert "# cli_path:" in (home / "agent_sys.yaml").read_text(encoding="utf-8")
    (home / "agent_sys.yaml").write_text("max_running: 2\n", encoding="utf-8")
    assert init_home(home) == []
    assert (home / "agent_sys.yaml").read_text(encoding="utf-8") == "max_running: 2\n"


def test_new_makes_the_home_too(tmp_path, capsys):
    home = tmp_path / "fresh"
    assert cli.main(["new", "demo", "--home", str(home)]) == 0
    assert (home / "projects/demo/project.yaml").is_file() and (home / "agent_sys.yaml").is_file()
    assert "创建 projects/、rules/、knowledge/entries/、agent_sys.yaml" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == f"agent-sys-v2 {__version__}"


def test_runs_from_the_root_of_an_infera_checkout(tmp_path):
    """`python -m agent_sys_v2...` (the server in tmux, the runners, the checks, the agent's ledger check) puts the
    current directory first on sys.path. At the root of an Infera checkout agent_sys_v2/ is the project directory, not
    the package: it must not hide the installed package (an editable install too)."""
    (tmp_path / "agent_sys_v2").mkdir()
    out = subprocess.run(
        [sys.executable, "-m", "agent_sys_v2.cli", "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == f"agent-sys-v2 {__version__}", out.stderr


def test_one_server_per_home(tmp_path):
    home = tmp_path / "home"
    init_home(home)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time; from agent_sys_v2.cli import lock_home; "
            'lock = lock_home(sys.argv[1], "host-a pid 1, socket s\\n"); print("held", flush=True); '
            "time.sleep(60)",
            str(home),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        assert cli.lock_home(home) is None
        assert cli.lock_holder(home) == "host-a pid 1, socket s"
    finally:
        holder.kill()
        holder.wait()
    lock = cli.lock_home(home, "me\n")
    assert lock is not None
    lock.close()


def test_socket_directory_must_be_private(tmp_path, capsys):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    assert not cli.private_dir(shared)
    assert "其他用户也能进入" in capsys.readouterr().err
    made = tmp_path / "a" / "sock"
    assert cli.private_dir(made) and made.stat().st_mode & 0o777 == 0o700


def windows(tmux):
    out = subprocess.run(
        ["tmux", "-L", tmux, "list-windows", "-t", "agent_sys", "-F", "#W #{pane_current_command}"],
        capture_output=True,
        text=True,
    ).stdout
    return dict(line.split(" ", 1) for line in out.splitlines())


def test_serve_in_tmux_leaves_other_servers_alone(tmp_path, own_tmux, monkeypatch, capsys):
    """Another server's window is not restarted in this one's place, and a second server on the same AGENT_SYS_HOME is
    refused (two schedulers would start the same tasks twice)."""
    home = tmp_path / "home"
    init_home(home)
    socks = tmp_path / "s"
    cli.private_dir(socks)
    # a window called server that serves another socket (here: a stand-in that only sleeps)
    subprocess.run(
        [
            "tmux",
            "-L",
            own_tmux,
            "new-session",
            "-d",
            "-s",
            "agent_sys",
            "-n",
            "server",
            f"sleep 600 # --sock {socks}/other.sock",
        ],
        check=True,
    )
    assert cli.serve_in_tmux(home, socks / "a.sock") == 0
    wait_for(lambda: cli.listening(socks / "a.sock"))
    assert windows(own_tmux)["server"] == "sleep"  # not respawned
    assert "server-2" in windows(own_tmux)
    start = subprocess.run(
        [
            "tmux",
            "-L",
            own_tmux,
            "display",
            "-p",
            "-t",
            "agent_sys:server-2",
            "#{pane_start_command}",
        ],
        capture_output=True,
        text=True,
    ).stdout
    assert f"AGENT_SYS_TMUX={own_tmux}" in start  # its runs go into this tmux too
    assert "AGENT_SYS_HOSTS=testserver" in start  # not the running tmux server's environment
    assert "server-2 窗口启动" in capsys.readouterr().out

    assert cli.serve_in_tmux(home, socks / "b.sock") == 1
    assert (
        "已有 agent_sys server 在运行" in capsys.readouterr().err
        and not (socks / "b.sock").exists()
    )
    assert cli.serve_in_tmux(home, socks / "a.sock") == 0  # the same one: already running
    assert "已在运行" in capsys.readouterr().out
    other = tmp_path / "other"
    init_home(other)
    assert cli.serve_in_tmux(other, socks / "a.sock") == 1  # a.sock serves another AGENT_SYS_HOME
    assert "另一个 AGENT_SYS_HOME" in capsys.readouterr().err


def test_serve_in_tmux_says_when_the_server_does_not_start(tmp_path, own_tmux, monkeypatch, capsys):
    """A server that exits at once (here: its socket's directory is not private) is reported with what its window
    shows, not as started; `serve` again restarts it in the same window."""
    home = tmp_path / "home"
    init_home(home)
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    start = time.time()
    assert cli.serve_in_tmux(home, shared / "a.sock") == 1
    assert time.time() - start < cli.START_SECONDS  # not waiting for a server that has exited
    out = capsys.readouterr()
    assert "没有启动起来" in out.err and str(shared) in out.err and "ssh" not in out.out

    shared.chmod(0o700)
    assert cli.serve_in_tmux(home, shared / "a.sock") == 0
    assert cli.listening(shared / "a.sock") and list(windows(own_tmux)) == ["server"]
    assert "server 窗口启动" in capsys.readouterr().out
