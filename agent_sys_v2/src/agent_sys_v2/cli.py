"""agent_sys v2 command line (`agent-sys-v2`, or `python -m agent_sys_v2.cli`).

  agent-sys-v2 serve [--home DIR] [--sock PATH]                       start the server in tmux -L $AGENT_SYS_TMUX
  agent-sys-v2 serve --foreground [--home DIR] [--sock PATH]          run the server here (tmux runs this)
  agent-sys-v2 new <project> [--template infera-perf] [--title T] [--home DIR]   new project from a template
  agent-sys-v2 --version

AGENT_SYS_HOME comes from --home, else the AGENT_SYS_HOME environment variable, else ~/agent_sys_home; what it lacks
(projects/, rules/, knowledge/entries/, agent_sys.yaml) is made. One server per AGENT_SYS_HOME: a second one would
start the same tasks again. The server listens on a Unix socket, /tmp/agent_sys-$UID/agent_sys.sock by default, in a
directory only you can enter (mode 0700: only you can connect); open it from your desktop with
`ssh -N -L 8765:<socket> <this host>` and http://localhost:8765. The server and the runs go into tmux -L
$AGENT_SYS_TMUX (default agent_sys): a test or demo server uses a tmux, a socket and an AGENT_SYS_HOME of its own.
"""

import argparse
import fcntl
import os
import re
import shlex
import socket
import sys
import time
from pathlib import Path

from . import __version__, i18n, tmux
from .i18n import _
from .model import DEFAULT_HOME, AgentSysError
from .ops import TEMPLATES, init_home, list_templates, new_project, template_tasks

DEFAULT_SOCK = Path(f"/tmp/agent_sys-{os.getuid()}/agent_sys.sock")
START_SECONDS = 30  # `serve` waits this long for the server it put in tmux to listen


def listening(sock):
    with socket.socket(socket.AF_UNIX) as s:
        try:
            s.connect(str(sock))
            return True
        except OSError:
            return False


def lock_home(home, holder=""):
    """The AGENT_SYS_HOME's server lock, held for as long as the returned file stays open; None if another server holds
    it (two schedulers would start the same task twice). flock on NFS is a lock across hosts too."""
    f = open(Path(home) / ".server.lock", "a+", encoding="utf-8")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(holder)
    f.flush()
    return f


def lock_holder(home):
    """Who holds the AGENT_SYS_HOME's server lock (host, pid, socket), as it wrote it."""
    try:
        return (Path(home) / ".server.lock").read_text(encoding="utf-8").strip() or _("未知")
    except OSError:
        return _("未知")


def home_taken(home):
    print(
        _(
            "agent_sys: {home} 已有 agent_sys server 在运行（{holder}）",
            home=home,
            holder=lock_holder(home),
        ),
        file=sys.stderr,
    )
    return 1


def private_dir(path):
    """The socket's directory: made with mode 0700 if missing (the default one is reset to 0700). uvicorn makes the
    socket itself rw for everyone, so the directory is what keeps other users out: refuse one they can enter."""
    if not path.is_dir():
        path.mkdir(mode=0o700, parents=True)
    elif path == DEFAULT_SOCK.parent:
        path.chmod(0o700)
    st = path.stat()
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        print(
            _(
                "agent_sys: socket 所在的目录 {path} 其他用户也能进入，请换一个只有你能访问的目录（例如 {example}/）",
                path=path,
                example=DEFAULT_SOCK.parent,
            ),
            file=sys.stderr,
        )
        return False
    return True


def serve_here(home, sock):
    import uvicorn

    from .server import create_app

    tmux.clean_own_env()
    if not private_dir(sock.parent):
        return 1
    if listening(sock):
        print(_("agent_sys: {sock} 上已有 agent_sys server 在运行", sock=sock), file=sys.stderr)
        return 1
    lock = lock_home(home, f"{socket.gethostname()} pid {os.getpid()}, socket {sock}\n")
    if lock is None:
        return home_taken(home)
    sock.unlink(missing_ok=True)  # left over from a server that was killed
    print(
        _("agent_sys server：AGENT_SYS_HOME={home}，socket {sock}", home=home, sock=sock),
        flush=True,
    )
    # SSE connections stay open: give them 3 s on shutdown instead of waiting for the browser to go away
    uvicorn.run(create_app(home), uds=str(sock), log_level="warning", timeout_graceful_shutdown=3)
    lock.close()
    return 0


def server_window(sock):
    """The window of the tmux session that runs the server of this socket (found by its command, so that a server of
    another socket, or a run whose task happens to be called server, is never restarted in its place)."""
    flag = re.escape(shlex.join(["--sock", str(sock)])) + r"""(?=[\s;'"]|$)"""
    for line in tmux.run(
        "list-windows", "-t", tmux.SESSION, "-F", "#{window_id} #{pane_start_command}"
    ).stdout.splitlines():
        wid, _sep, start = line.partition(" ")
        if re.search(flag, start):
            return wid
    return None


def wait_started(wid, sock, exited):
    """Wait for the server in tmux window wid to listen on sock. None once it does; else what the window shows, as soon
    as the server has exited (its error, then the line `exited`) or after START_SECONDS. Only the screen: a respawned
    window keeps the scrollback of the server before."""
    end = time.time() + START_SECONDS
    while not listening(sock):
        shown = tmux.run("capture-pane", "-p", "-J", "-t", wid).stdout
        if exited in shown or time.time() > end:
            return "\n".join(line.rstrip() for line in shown.strip().splitlines())
        time.sleep(0.2)
    return None


def serve_in_tmux(home, sock):
    """Run `agent-sys-v2 serve --foreground` in tmux -L $AGENT_SYS_TMUX, unless this AGENT_SYS_HOME's server already
    listens on sock; then print how to open it. Refused when another server holds the AGENT_SYS_HOME (on another
    socket), or sock (for another AGENT_SYS_HOME)."""
    lock = lock_home(home)
    if lock is None:  # this AGENT_SYS_HOME's server runs: on this socket, or on another one
        if not (listening(sock) and lock_holder(home).endswith(f"socket {sock}")):
            return home_taken(home)
        print(_("agent_sys server 已在运行：{sock}", sock=sock))
    else:
        lock.close()  # only a check: the server in tmux takes the lock itself
        if listening(sock):
            print(
                _(
                    "agent_sys: {sock} 上运行的是另一个 AGENT_SYS_HOME 的 server，请用 --sock 换一个 socket",
                    sock=sock,
                ),
                file=sys.stderr,
            )
            return 1
        if start_in_tmux(home, sock):
            return 1
    print(
        _(
            "在你的电脑上运行：ssh -N -L 8765:{sock} {host}\n然后用浏览器打开 http://localhost:8765",
            sock=sock,
            host=socket.getfqdn(),
        )
    )
    return 0


def start_in_tmux(home, sock):
    """The server in tmux: restarted in place in the window that already serves this socket, else in a new window
    "server" ("server-2", … if taken), in a new session if there is none. 0 once it listens; if it does not, what
    its window shows is printed (the window stays, with the error) and 1 returned."""
    # AGENT_SYS_TMUX and AGENT_SYS_HOSTS given on the command line: a new window gets the tmux server's environment,
    # not this one's, and the server's runs must go where it is
    hosts = os.environ.get("AGENT_SYS_HOSTS")
    cmd = shlex.join(
        [
            "env",
            f"AGENT_SYS_TMUX={tmux.TMUX}",
            *([f"AGENT_SYS_HOSTS={hosts}"] if hosts else []),
            sys.executable,
            "-m",
            "agent_sys_v2.cli",
            "serve",
            "--foreground",
            "--home",
            str(home),
            "--sock",
            str(sock),
        ]
    )
    exited = _("agent_sys server 已退出，按回车关闭窗口")
    cmd += f"; echo {shlex.quote(exited)}; read"  # keep the error on screen

    # New windows take tmux's global environment, which is the parent session's if a Claude Code session
    # started the tmux server. Clean it, for this window and for the runners' windows.
    for line in tmux.run("show-environment", "-g").stdout.splitlines():
        name = line.lstrip("-").split("=")[0]
        if tmux.PARENT_SESSION.match(name):
            tmux.run("set-environment", "-g", "-u", name)
    try:
        if wid := server_window(sock):
            window = tmux.run("display-message", "-p", "-t", wid, "#W").stdout.strip()
            tmux.run("respawn-window", "-k", "-t", wid, cmd)
        else:
            window = tmux.window_name("server")
            wid = tmux.open_window(window, cmd, Path.cwd())
    except AgentSysError as e:
        print(f"agent_sys: {e}", file=sys.stderr)
        return 1
    shown = wait_started(wid, sock, exited)
    if shown is not None:
        print(
            _(
                "agent_sys server 没有启动起来（tmux -L {tmux} 的 {window} 窗口），窗口里显示：",
                tmux=tmux.TMUX,
                window=window,
            ),
            shown,
            _("查看日志：{attach}", attach=tmux.attach(window)),
            sep="\n",
            file=sys.stderr,
        )
        return 1
    print(
        _(
            "agent_sys server 已在 tmux -L {tmux} 的 {window} 窗口启动：{sock}",
            tmux=tmux.TMUX,
            window=window,
            sock=sock,
        )
    )
    print(_("查看日志：{attach}", attach=tmux.attach(window)))
    return 0


def main(argv=None):
    default_home = Path(os.environ.get("AGENT_SYS_HOME") or DEFAULT_HOME)
    i18n.use(default_home)  # the help, before --home is read
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--home",
        type=Path,
        default=default_home,
        help=_(
            "AGENT_SYS_HOME：默认取环境变量 AGENT_SYS_HOME，没有则 {default}（当前 %(default)s）",
            default=DEFAULT_HOME,
        ),
    )
    ap = argparse.ArgumentParser(
        prog="agent-sys-v2", description=_("agent_sys：Infera 性能工程的 agent 工作台")
    )
    ap.add_argument("--version", action="version", version=f"agent-sys-v2 {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", parents=[common], help=_("用模板新建项目"))
    p.add_argument("project", help=_("项目名，即 projects/ 下的目录名"))
    p.add_argument(
        "--template",
        default="blank",
        help=_(
            "模板：{templates}（默认 blank）",
            templates=" / ".join(t["id"] for t in list_templates()),
        ),
    )
    p.add_argument("--title", help=_("项目标题（默认用项目名）"))
    s = sub.add_parser("serve", parents=[common], help=_("启动 agent_sys server（默认放进 tmux）"))
    s.add_argument(
        "--sock", type=Path, default=DEFAULT_SOCK, help=_("Unix socket（默认 %(default)s）")
    )
    s.add_argument("--foreground", action="store_true", help=_("在当前终端运行，不进 tmux"))
    args = ap.parse_args(argv)

    home = args.home.expanduser().resolve()
    made = init_home(home)
    i18n.use(home)
    if made:
        print(
            _("agent_sys: 已在 {home} 创建 {made}", home=home, made=_("、").join(made)),
            file=sys.stderr,
        )
    if args.cmd == "serve":
        sock = args.sock.expanduser().absolute()
        return serve_here(home, sock) if args.foreground else serve_in_tmux(home, sock)
    try:
        path = new_project(home, args.project, args.template, args.title)
    except AgentSysError as e:
        print(f"agent_sys: {e}", file=sys.stderr)
        return 1
    print(path)
    if tasks := template_tasks(TEMPLATES / args.template):
        print(
            _(
                "agent_sys: 模板带来的 {n} 个任务等你审核：在看板上检查后点「开始运行」，或手动开始其中一个任务",
                n=len(tasks),
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
