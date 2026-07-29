#!/usr/bin/env python3
"""Log the POST-VOTE decision, i.e. what the fix actually did.

r01's probe logs the four raw terms and reconstructs the *pre-fix* decision.
That is the right thing for diagnosing the bug, but it cannot verify the fix:
after the fix the terms still diverge (they are inherently rank-dependent) --
what must become uniform is the value the code acts on.

This probe inserts one line immediately after the all-reduce, so the recorded
value is exactly what `can_cuda_graph` is set from.

Emits: GLM52_VOTE dp=<r> it=<n> local=<bool> voted=<bool>
  local != voted on some rank  => the vote changed that rank's mind (working)
  voted uniform across ranks   => the invariant the fix exists to restore
"""
import os
import shutil
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"
MARKER = "GLM52_VOTE"

ANCHOR = """                _needs_eager_local = bool(_vote.item())
"""

PROBE = '''                _needs_eager_local = bool(_vote.item())
                import os as _o, sys as _s

                if _o.environ.get("GLM52_VOTE_PROBE", "0") == "1":
                    from sglang.srt.layers.dp_attention import (
                        get_attention_dp_rank as _gdr2,
                    )
                    _s.stderr.write(
                        "GLM52_VOTE dp=%s it=%s local=%s voted=%s\\n"
                        % (
                            _gdr2(),
                            getattr(self, "_glm52_it", -1),
                            _local_before_vote,
                            _needs_eager_local,
                        )
                    )
                    _s.stderr.flush()
'''

# capture the pre-vote value so local vs voted can be compared
ANCHOR2 = """                _vote = torch.tensor(
                    [1 if _needs_eager_local else 0], dtype=torch.int32
                )
"""
PROBE2 = """                _local_before_vote = _needs_eager_local
                _vote = torch.tensor(
                    [1 if _needs_eager_local else 0], dtype=torch.int32
                )
"""


def purge_pyc(path):
    os.utime(path, None)
    d, f = os.path.split(path)
    pd = os.path.join(d, "__pycache__")
    if os.path.isdir(pd):
        for n in os.listdir(pd):
            if n.startswith(f[:-3] + "."):
                os.remove(os.path.join(pd, n))


def main():
    bak = TARGET + ".votebak"
    if "--revert" in sys.argv:
        if os.path.exists(bak):
            shutil.copyfile(bak, TARGET)
            os.remove(bak)
            purge_pyc(TARGET)
            print("reverted")
            return 0
        print("no backup")
        return 1

    src = open(TARGET).read()
    if MARKER in src:
        print("already applied")
        return 0
    for a in (ANCHOR, ANCHOR2):
        if src.count(a) != 1:
            print(f"anchor matched {src.count(a)} times, want 1 -- refusing")
            return 2
    # os/sys are imported locally inside the probe -- eagle_worker_v2 has no
    # module-level `import os`, which the earlier version of this guard caught.

    if not os.path.exists(bak):
        shutil.copyfile(TARGET, bak)
    src = src.replace(ANCHOR2, PROBE2, 1).replace(ANCHOR, PROBE, 1)
    open(TARGET, "w").write(src)
    purge_pyc(TARGET)
    import py_compile

    py_compile.compile(TARGET, doraise=True)
    print("applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
