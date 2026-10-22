#!/usr/bin/env bash
# Verify infera-sglang:v0521-yihou-1009 in a CPU-only container. Read-only.
set -uo pipefail
fail=0
ok()  { echo "  OK   $*"; }
bad() { echo "  FAIL $*"; fail=1; }

echo "== 1. imports"
python3 -c "import infera, sglang; print('infera', infera.__file__); print('sglang', sglang.__version__, sglang.__file__)" && ok imports || bad imports

echo "== 2. router flag"
/usr/local/bin/infera-router --help 2>&1 | grep -- '--pd-dp-rank-affinity' && ok router || bad router

echo "== 3. patch markers: source + fresh bytecode + image __pycache__ freshness"
python3 - <<'EOF' || fail=1
import importlib.util, marshal, os, py_compile, struct, sys, tempfile, types
root = os.path.dirname(importlib.util.find_spec("sglang").origin) + "/srt/"
# file : [(kind, needle[, scope])]; kind src = substring of source;
# pyc = string constant / name somewhere in the compiled module;
# scope = code-object name whose names+consts must contain the needle.
CHECKS = {
  "layers/attention/dsa/dsa_indexer.py": [("pyc", "_p1v2_trim"), ("pyc", "_p1v2_rows")],
  "layers/attention/dsa_backend.py": [
      ("pyc", "extend_seq_lens{,_cpu} must not be None for DRAFT_EXTEND_V2"),
      ("pyc", "_glm52_match_page_table_rows"),
      ("scope", "_glm52_match_page_table_rows", "forward_decode"),
      ("scope", "_glm52_match_page_table_rows", "_forward_trtllm")],
  "models/glm4_moe.py": [("scope", "fused_shared_experts_architecture", "GlmMoeDsaForCausalLMNextN"),
                         ("scope", "GlmMoeDsaForCausalLMNextN", "GlmMoeDsaForCausalLMNextN")],
  "disaggregation/prefill.py": [("pyc", "_infera_nonfinal_chunk_wait_event")],
  "disaggregation/mooncake/conn.py": [("src", "kv_chunk.wait_event.synchronize()")],
  "arg_groups/speculative_hook.py": [("src", "PD decode does not receive draft_probs from prefill"),
                                     ("scope", "disaggregation_mode", "_handle_eagle_family")],
  "entrypoints/openai/serving_responses.py": [("pyc", "_infera_responses_unstreamed_tool_args")],
  "mem_cache/pool_host/mla.py": [("pyc", "GLM52_ROCM_STAGED_WRITE_BACK")],
  "managers/scheduler_components/dp_attn.py": [("pyc", "_spec_input_cuda_graph_compatible")],
  "speculative/eagle_draft_cuda_graph_runner.py": [("pyc", "can_run_dp_draft_cuda_graph")],
}

def walk(co):
    yield co
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            yield from walk(c)

def strs(co):
    out = set(co.co_names) | set(co.co_varnames)
    out |= {c for c in co.co_consts if isinstance(c, str)}
    return out

bad = 0
tmp = tempfile.mkdtemp()
for rel, checks in CHECKS.items():
    p = root + rel
    src = open(p).read()
    # Bytecode of record: compiled fresh from the image's source, outside the tree.
    # marshal only reads a .pyc we just compiled from that source.
    cfile = py_compile.compile(p, cfile=os.path.join(tmp, rel.replace("/", "_") + "c"), doraise=True)
    code = marshal.loads(open(cfile, "rb").read()[16:])
    allstr = set().union(*(strs(c) for c in walk(code)))
    for chk in checks:
        kind, needle = chk[0], chk[1]
        if kind == "src":
            r = needle in src
        elif kind == "pyc":
            r = needle in src and any(needle in s for s in allstr)
        else:
            scopes = [c for c in walk(code) if c.co_name == chk[2]]
            r = bool(scopes) and all(needle in strs(c) for c in scopes)
        print(f"  {'OK  ' if r else 'FAIL'} {rel} :: {kind} {needle!r}" + (f" in {chk[2]}" if kind == "scope" else ""))
        bad |= not r
    # The image's own cache: a pyc whose recorded mtime/size matches the source
    # would be executed instead of the source -- it must be fresh or absent.
    d, b = os.path.split(p)
    st = os.stat(p)
    for f in os.listdir(os.path.join(d, "__pycache__")) if os.path.isdir(os.path.join(d, "__pycache__")) else []:
        if f.startswith(b[:-3] + "."):
            h = open(os.path.join(d, "__pycache__", f), "rb").read(16)
            flags, mt, sz = struct.unpack("<III", h[4:16])
            fresh = flags == 0 and mt == int(st.st_mtime) & 0xFFFFFFFF and sz == st.st_size & 0xFFFFFFFF
            if fresh:
                c2 = marshal.loads(open(os.path.join(d, "__pycache__", f), "rb").read()[16:])
                fresh = c2 == code
            print(f"  {'OK  ' if fresh else 'FAIL'} {rel} :: image __pycache__/{f} {'matches source' if fresh else 'STALE'}")
            bad |= not fresh
sys.exit(1 if bad else 0)
EOF

echo "== 4. mooncake"
# The LOADED engine.so is what matters; /sgl-workspace/Mooncake is the base's
# checkout and is not what BUILD_MOONCAKE=1 installs.
echo "  base checkout /sgl-workspace/Mooncake: $(cd /sgl-workspace/Mooncake && git rev-parse HEAD)"
so=$(python3 -c 'import mooncake.engine as e; print(e.__file__)')
echo "  loaded engine.so: $so"
echo "  sha256: $(sha256sum "$so" | cut -d' ' -f1)"
ls /opt/infera/MOONCAKE* /opt/mooncake* 2>/dev/null
for s in isHipReachableTarget MC_DISABLE_HIP MOONCAKE_DISABLE_HIP_DMABUF "HIP transport installed"; do
  n=$(strings "$so" | grep -c "$s"); [ "$n" -gt 0 ] && ok "engine.so has '$s'" || bad "engine.so lacks '$s'"
done

echo "== 5. --dsa-*-backend triton parses"
python3 - <<'EOF' && ok "dsa triton" || bad "dsa triton"
import argparse
from sglang.srt.server_args import ServerArgs
p = argparse.ArgumentParser()
ServerArgs.add_cli_args(p)
a = p.parse_args(["--model-path", "x", "--dsa-prefill-backend", "triton", "--dsa-decode-backend", "triton"])
print("  parsed:", a.dsa_prefill_backend, a.dsa_decode_backend)
assert a.dsa_prefill_backend == "triton" and a.dsa_decode_backend == "triton"
EOF

echo "== RESULT fail=$fail"
exit $fail
