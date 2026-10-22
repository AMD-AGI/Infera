#!/usr/bin/env python3
"""E-dbg -- ROCm debug agent smoke test inside the image (one GPU).

1. --check  : locate librocm-debug-agent.so.2 / librocm-dbgapi.so*, print versions.
2. faults   : a triton kernel `yihou_oob_store_kernel` stores 1 TiB past a small
              allocation -> GPU memory fault. Run eagerly and inside a CUDA-graph
              replay, each in a subprocess, without and with the agent; check
              whether the agent output names the kernel.
3. overhead : steady-state loops (big matmuls, tiny triton launches, graph replays)
              with and without HSA_TOOLS_LIB / HSA_ENABLE_DEBUG.

Agent options (ROCm/rocm-systems projects/rocr-debug-agent README):
  -a/--all  -p/--precise-memory  -e/--precise-alu-exceptions
  -s/--save-code-objects[=DIR]  -c/--load-all-code-objects  -z/--lazy
  -o/--output <file>  -d/--disable-linux-signals  -l/--log-level <none|info|warning|error>
On a queue error the agent prints the stopped waves (pc, stop reason, registers)
and a "Disassembly for function <kernel symbol>" block -- that is where the name is.

Output lines: `CHECK ...`, `FAULT ...`, `BENCH ...`, and final `VERDICT ...`.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KNAME = "yihou_oob_store_kernel"


def _log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------- check
def find_libs():
    pats = {
        "debug_agent": ["/opt/rocm*/lib/librocm-debug-agent.so*", "/opt/rocm*/lib64/librocm-debug-agent.so*",
                        "/usr/lib*/librocm-debug-agent.so*", "/usr/lib/x86_64-linux-gnu/librocm-debug-agent.so*"],
        "dbgapi": ["/opt/rocm*/lib/librocm-dbgapi.so*", "/opt/rocm*/lib64/librocm-dbgapi.so*",
                   "/usr/lib*/librocm-dbgapi.so*", "/usr/lib/x86_64-linux-gnu/librocm-dbgapi.so*"],
    }
    found = {}
    for k, ps in pats.items():
        hits = []
        for p in ps:
            hits += glob.glob(p)
        found[k] = sorted(set(hits))
    return found


def check():
    info = {}
    for f in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        if os.path.exists(f):
            info[f] = open(f).read().strip()
    info["rocm_dirs"] = sorted(glob.glob("/opt/rocm*"))
    libs = find_libs()
    info.update(libs)
    try:
        out = subprocess.run("dpkg -l 2>/dev/null | grep -Ei 'debug-agent|dbgapi|rocm-core|hip-runtime-amd' ",
                             shell=True, capture_output=True, text=True).stdout
        info["dpkg"] = [l for l in out.splitlines() if l.strip()]
    except Exception as e:
        info["dpkg"] = repr(e)
    try:
        info["ptrace_scope"] = open("/proc/sys/kernel/yama/ptrace_scope").read().strip()
    except Exception:
        info["ptrace_scope"] = "n/a"
    try:
        st = open("/proc/self/status").read()
        info["CapEff"] = re.search(r"CapEff:\s*(\S+)", st).group(1)
    except Exception:
        pass
    # does the agent link resolve (dbgapi present + loadable)?
    if libs["debug_agent"]:
        ld = subprocess.run(["ldd", libs["debug_agent"][0]], capture_output=True, text=True).stdout
        info["ldd_debug_agent_notfound"] = [l.strip() for l in ld.splitlines() if "not found" in l]
        info["ldd_dbgapi"] = [l.strip() for l in ld.splitlines() if "dbgapi" in l]
    for k, v in info.items():
        _log(f"CHECK {k}: {v}")
    return info


# ------------------------------------------------------------------------ children
def _triton_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def yihou_oob_store_kernel(p, addr, BLOCK: tl.constexpr):
        # store to the absolute address `addr` (warm-up passes p's own address)
        i = tl.arange(0, BLOCK)
        q = addr.to(tl.int64).to(tl.pointer_type(tl.int32))
        tl.store(q + i, tl.full((BLOCK,), 7, tl.int32))

    @triton.jit
    def yihou_tiny_add_kernel(x, y, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(y + i, tl.load(x + i, mask=m) + 1.0, mask=m)

    return yihou_oob_store_kernel, yihou_tiny_add_kernel


def child_fault(mode):
    import torch

    oob, _ = _triton_kernels()
    torch.cuda.set_device(0)
    buf = torch.zeros(1024, dtype=torch.int32, device="cuda")
    oob[(1,)](buf, buf.data_ptr(), BLOCK=64)  # in-bounds warm-up -> compiles + loads code object
    torch.cuda.synchronize()
    # A store 1 TiB past the buffer did NOT fault on 138 (VA mapped); use an absolute
    # low address (null page region) that is never GPU-mapped.
    bad = int(os.environ.get("YIHOU_BAD_ADDR", "0x1000"), 0)
    _log(f"[child] mode={mode} buf={buf.data_ptr():#x} bad_addr={bad:#x}")
    if mode == "eager":
        oob[(1,)](buf, bad, BLOCK=64)
    else:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            oob[(1,)](buf, buf.data_ptr(), BLOCK=64)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            oob[(1,)](buf, bad, BLOCK=64)
        torch.cuda.synchronize()
        _log("[child] graph captured; replaying")
        g.replay()
    torch.cuda.synchronize()
    _log("[child] NO FAULT (unexpected)")


def child_bench(iters_mm, iters_tiny, replays):
    import torch

    t_start = time.time()
    _, tiny = _triton_kernels()
    torch.cuda.set_device(0)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(4096, device="cuda")
    y = torch.empty_like(x)
    for _ in range(20):
        a @ b
        tiny[(4,)](x, y, 4096, BLOCK=1024)
    torch.cuda.synchronize()
    t_init = time.time() - t_start

    t0 = time.time()
    for _ in range(iters_mm):
        c = a @ b
    torch.cuda.synchronize()
    t_mm = time.time() - t0

    t0 = time.time()
    for _ in range(iters_tiny):
        tiny[(4,)](x, y, 4096, BLOCK=1024)
    torch.cuda.synchronize()
    t_tiny = time.time() - t0

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        tiny[(4,)](x, y, 4096, BLOCK=1024)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(100):
            tiny[(4,)](x, y, 4096, BLOCK=1024)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(replays):
        g.replay()
    torch.cuda.synchronize()
    t_graph = time.time() - t0
    del c
    _log("BENCHRES " + json.dumps(dict(init_s=round(t_init, 3), mm_s=round(t_mm, 4),
                                       tiny_s=round(t_tiny, 4), graph_s=round(t_graph, 4),
                                       iters_mm=iters_mm, iters_tiny=iters_tiny, replays=replays)))


# ------------------------------------------------------------------------- parent
def agent_env(lib, opts, enable_debug=True):
    e = {"HSA_TOOLS_LIB": lib, "ROCM_DEBUG_AGENT_OPTIONS": opts}
    if enable_debug:
        e["HSA_ENABLE_DEBUG"] = "1"
    return e


def run_child(argv, extra_env, timeout, logf):
    env = dict(os.environ)
    for k in ("HSA_TOOLS_LIB", "HSA_ENABLE_DEBUG", "ROCM_DEBUG_AGENT_OPTIONS"):
        env.pop(k, None)
    env.update(extra_env)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__)] + argv, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors="replace", timeout=timeout)
        rc, out = p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        rc = "TIMEOUT"
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
    with open(logf, "w") as f:
        f.write(f"# env {json.dumps(extra_env)}\n# argv {argv}\n" + out)
    return rc, out, round(time.time() - t0, 1)


def parent(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    info = check()
    libs = info.get("debug_agent") or []
    have_agent = bool(libs) and bool(info.get("dbgapi")) and not info.get("ldd_debug_agent_notfound")
    lib = libs[0] if libs else "librocm-debug-agent.so.2"
    if not have_agent:
        _log("CHECK debug agent NOT usable in this image (lib or dbgapi missing / unresolved). "
             f"Build the thin layer: {HERE}/Dockerfile.dbgagent.yihou ; fault/bench steps still run "
             "for the no-agent baseline.")

    def opts(tag, full):
        o = f"--output={outdir}/agent.{tag}.yihou.txt --disable-linux-signals"
        if full:
            o += f" --all --precise-memory --save-code-objects={outdir}/codeobj.{tag}.yihou"
            os.makedirs(f"{outdir}/codeobj.{tag}.yihou", exist_ok=True)
        return o

    # ---- faults
    fault_rows = []
    variants = [("noagent", {})]
    if have_agent:
        variants += [("lean", agent_env(lib, None)), ("full", agent_env(lib, None))]
    for mode in ("eager", "graph"):
        for vtag, env in variants:
            tag = f"{mode}.{vtag}"
            if env:
                env = dict(env, ROCM_DEBUG_AGENT_OPTIONS=opts(tag, vtag == "full"))
            rc, out, secs = run_child(["--child-fault", mode], env, args.timeout,
                                      f"{outdir}/fault.{tag}.yihou.log")
            agent_txt = ""
            af = f"{outdir}/agent.{tag}.yihou.txt"
            if os.path.exists(af):
                agent_txt = open(af, errors="replace").read()
            both = out + "\n" + agent_txt
            row = dict(tag=tag, rc=rc, secs=secs,
                       maf="Memory access fault" in both or "MEMORY_VIOLATION" in both
                           or "HSA_AMD_GPU_MEMORY_FAULT_EVENT" in both,
                       agent_output_bytes=len(agent_txt),
                       names_kernel=KNAME in both,
                       disasm_block=("Disassembly for function" in both),
                       stop_reasons=sorted(set(re.findall(r"reason: ([A-Z_]+)", both))))
            for line in both.splitlines():
                if "Memory access fault" in line or "Disassembly for function" in line:
                    row.setdefault("lines", []).append(line.strip()[:240])
            _log("FAULT " + json.dumps(row))
            fault_rows.append(row)

    # ---- overhead
    bench_rows = {}
    bvariants = [("noagent", {}), ("enable_debug_only", {"HSA_ENABLE_DEBUG": "1"})]
    if have_agent:
        bvariants += [("lean", agent_env(lib, f"--disable-linux-signals --output={outdir}/agent.bench-lean.yihou.txt")),
                      ("lean_nodebugflag", agent_env(lib, f"--disable-linux-signals --output={outdir}/agent.bench-lean2.yihou.txt", enable_debug=False)),
                      ("full", agent_env(lib, f"--disable-linux-signals --all --precise-memory --output={outdir}/agent.bench-full.yihou.txt"))]
    for rep in range(args.bench_reps):
        for vtag, env in bvariants:
            rc, out, secs = run_child(["--child-bench", str(args.iters_mm), str(args.iters_tiny), str(args.replays)],
                                      env, args.timeout, f"{outdir}/bench.{vtag}.r{rep}.yihou.log")
            m = [l for l in out.splitlines() if l.startswith("BENCHRES ")]
            res = json.loads(m[-1][9:]) if m else {"error": f"rc={rc}"}
            res.update(rc=rc, wall_s=secs)
            _log(f"BENCH {vtag} r{rep} " + json.dumps(res))
            bench_rows.setdefault(vtag, []).append(res)

    def best(v, k):
        xs = [r[k] for r in bench_rows.get(v, []) if k in r]
        return min(xs) if xs else None

    for v in bench_rows:
        if v == "noagent":
            continue
        parts = []
        for k in ("mm_s", "tiny_s", "graph_s", "init_s"):
            a, b = best("noagent", k), best(v, k)
            if a and b:
                parts.append(f"{k}={b / a:.3f}x")
        _log(f"BENCH ratio {v}/noagent (min of reps): " + " ".join(parts))

    # ---- verdict
    fault_ok = all(r["maf"] for r in fault_rows if r["tag"].endswith("noagent"))
    named = {r["tag"]: r["names_kernel"] for r in fault_rows if not r["tag"].endswith("noagent")}
    if not have_agent:
        v = f"AGENT ABSENT; baseline faults reproduced={fault_ok}; build Dockerfile.dbgagent.yihou and rerun"
    else:
        v = (f"AGENT PRESENT; baseline faults reproduced={fault_ok}; kernel named per variant={named}; "
             "see BENCH ratio lines for overhead")
    _log("VERDICT " + v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--child-fault", choices=["eager", "graph"])
    ap.add_argument("--child-bench", nargs=3, type=int, metavar=("ITERS_MM", "ITERS_TINY", "REPLAYS"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "out.dbg.yihou"))
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--iters-mm", type=int, default=2000)
    ap.add_argument("--iters-tiny", type=int, default=20000)
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--bench-reps", type=int, default=2)
    args = ap.parse_args()
    if args.check:
        check()
    elif args.child_fault:
        child_fault(args.child_fault)
    elif args.child_bench:
        child_bench(*args.child_bench)
    else:
        parent(args)


if __name__ == "__main__":
    main()
