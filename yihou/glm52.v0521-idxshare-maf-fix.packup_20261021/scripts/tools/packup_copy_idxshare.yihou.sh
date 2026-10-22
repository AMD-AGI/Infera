#!/usr/bin/env bash
# Packup copy step for glm52.v0521-idxshare-maf-fix.packup_20261021 (2026-10-21). COPY-ONLY: reads the
# experiment sources, writes only inside $P. Never deletes, never moves. Run on a spur login host (small I/O).
set -eu
P=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-idxshare-maf-fix.packup_20261021
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
D=/shared_nfs/yihou_glm52/idxshare_dbg_20261020
W=/shared_nfs/yihou_glm52/g4tune_20261019
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
L=/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012/debug/poll.md
[[ "$P" == *yihou* ]] || exit 1
mkdir -p "$P"/scripts/{harness,mounts-src,session,tools,diagnosis} "$P"/analysis/ledger \
         "$P"/results/{rounds,bench,bad_dumps,maf,debug_agent,seedguard,dryruns,launch,pyspy}

# --- harness overlays + base chain + mount tables + topology (verbatim) ---
for f in config.yihou.{g4new,dbgE1,dbgE2,idlefix,xferfix2,offload,tokw2,mc4340n16,mc4340n17,mc4340n18,E2-target,v0521-base,seedguard}.sh \
         mounts.yihou.v2-xferfix2.tsv mounts.yihou.v2-xferfix2-gtt.tsv mounts.yihou.v2-xferfix2-gtt-idlefix.tsv \
         mounts.yihou.v2-xferfix2-gtt-seedguard.tsv mounts.yihou.v2-xferfix2-gtt-seedguard-topkguard{,2,3,4,4sk,5sk2,5sk3,5sk3fix}.tsv \
         topology.yihou.{097,250,268,054}.tsv run.v3.yihou.sh tokdiff.yihou.sh chain_cap2.yihou.sh pipeline.v3.yihou.sh warmup_pyspy.yihou.sh; do
    cp -p "$H/$f" "$P/scripts/harness/"
done
# --- mounted engine sources (the fix, the guards, and the base overlays every round mounts) ---
for f in eager_runner.yihou-idlefix.py top_k.yihou-skguard{,2,3}.py dsa_indexer.yihou-topkguard{,2,3,4,5}.py \
         eagle_disaggregation.yihou-seedguard.py eagle_worker_v2.yihou-seedguard.py \
         kv_events_publisher.yihou-noblock.py forward_mla_rocm.yihou-noabsorbcfg.py conn.yihou-xferfix2.py \
         prefill.yihou-nonfinal-event.py pool_host_common.yihou-pr42553.py; do
    cp -p "$H/mounts-v0521/src/$f" "$P/scripts/mounts-src/"
done
# --- session: env files, launch/stop helpers ---
cp -p "$D"/env.*.yihou.sh "$W"/env.g4n{1,1b,2,2b,3,3b,4}-*.yihou.sh "$P/scripts/session/"
cp -p "$W/launch_after_prep.yihou.sh" "$W/stop_round.yihou.sh" "$W/chain_resume.yihou.sh" "$D/stop_before_maxwall.yihou.sh" \
      "$N/prep_node.yihou.sh" "$P/scripts/session/"
cp -p "$N/tools_hits.yihou.py" "$P/scripts/tools/"
cp -p "$D"/test_skguard.yihou.py "$D"/test_skguard2.yihou.py "$D/top_k.image.py" "$P/scripts/diagnosis/"
# --- analysis: report, root cause, prep notes, ledger snapshot ---
cp -p "$D/REPORT.idxshare-maf.20261020.md" "$D/ROOTCAUSE.20261020.md" "$D/NOTES.prep.20261020.md" "$P/analysis/"
cp -p "$L" "$P/analysis/ledger/poll.md"
cp -p "$D/image_v0521_local.diff" "$P/patches/0000-image-local-diff.vs-9f5d8712.diff"
# --- dryruns + launch logs ---
cp -p "$D"/dryrun.*.yihou.txt "$W"/dryrun.g4n{1,1b,2,2b,3,3b,4}-*.yihou.txt "$P/results/dryruns/"
for r in g4n1 g4n1b g4n2 g4n2b g4n3 g4n3b g4n4 dbge1a dbge1b dbge2a dbge2b dbge2c dbge2d dbge2e dbge3a dbge3b dbge4a dbge4b dbge5a dbge5b dbge6a idxf1a idxf1b idxf1c; do
    mkdir -p "$P/results/rounds/$r"
    cp -p "$R/$r/status.tsv" "$P/results/rounds/$r/"
    [[ -e "$R/$r.chain.log" ]] && cp -p "$R/$r.chain.log" "$P/results/rounds/$r/"
    [[ -e "$W/launch.$r.yihou.log" ]] && cp -p "$W/launch.$r.yihou.log" "$P/results/launch/"
done
# --- bench JSONs (the scored rounds) ---
for ra in g4n1/A1 g4n3/A1 g4n3/A2 g4n3b/A1 g4n3b/A2 dbge6a/A1 idxf1a/A1 idxf1a/A2; do
    r=${ra%/*}; a=${ra#*/}; mkdir -p "$P/results/bench/$r-c56-$a"
    cp -p "$R/$r/agentx-c56-$a/agentx_conc56.json" "$P/results/bench/$r-c56-$a/"
done
# --- MAF lines (all faulting rounds) ---
{
for r in g4n2 g4n2b g4n1 g4n4 dbge1a dbge1b dbge4a dbge4b; do
    echo "=== $r $R/$r/server-logs/decode-0.a1.log"
    grep -B1 -A4 "Memory access fault" "$R/$r/server-logs/decode-0.a1.log" | cut -c1-300
done
echo "=== MAF/BAD counts per round (grep -c 'Memory access fault' / 'SKGUARD-EAGER BAD fast_topk')"
for r in g4n1 g4n1b g4n2 g4n2b g4n3 g4n3b g4n4 dbge1a dbge1b dbge2a dbge2b dbge2c dbge2d dbge2e dbge3a dbge3b dbge4a dbge4b dbge5a dbge5b dbge6a idxf1a; do
    f="$R/$r/server-logs/decode-0.a1.log"
    echo "$r MAF=$(grep -c 'Memory access fault' "$f" || true) BAD=$(grep -c 'SKGUARD-EAGER BAD fast_topk' "$f" || true)"
done
} > "$P/results/maf/maf_lines.txt"
# --- SKGUARD BAD dumps (first line of each record; the stack continuation lines are not included) ---
for r in dbge4a dbge4b dbge5a dbge5b; do
    grep "SKGUARD-EAGER BAD" "$R/$r/server-logs/decode-0.a1.log" | cut -c1-4000 > "$P/results/bad_dumps/$r.skguard_bad.txt" || true
done
# one full BAD record incl. Python stack (dbge5a, first BAD)
f="$R/dbge5a/server-logs/decode-0.a1.log"; n=$(grep -n "SKGUARD-EAGER BAD fast_topk" "$f" | head -1 | cut -d: -f1)
sed -n "${n},$((n+120))p" "$f" | cut -c1-4000 > "$P/results/bad_dumps/dbge5a.first_bad_full.txt"
# --- seedguard / skguard counters of the fix-with-guards round dbge6a ---
grep -E "YIHOU-SEEDGUARD|YIHOU-SKGUARD " "$R/dbge6a/server-logs/decode-0.a1.log" | tail -n 40 | cut -c1-1500 > "$P/results/seedguard/dbge6a.guard_counters.tail.txt" || true
# --- ROCm debug agent output + disassembly ---
cp -p "$D/dbge1a.wave7124.txt" "$D/dbge1a.fault_tail.txt" "$D/dbge1b.wave.txt" "$D/coop.dis.yihou.txt" "$D/coop.co.yihou" "$P/results/debug_agent/"
sed -n '150,250p' "$D/dbge1a.wave7124.txt" > "$P/results/debug_agent/dbge1a.wave7124.regs.txt"
cp -p "$D/pyspy.dbge2d.dp0.raw" "$P/results/pyspy/"
# --- not copied (> 4 MB): list with absolute path and size ---
{
printf 'bytes\tpath\n'
for r in g4n1 g4n2 g4n2b g4n4 dbge1a dbge1b dbge4a dbge4b dbge5a dbge5b dbge6a idxf1a g4n3; do
    for f in "$R/$r"/server-logs/*.log; do printf '%s\t%s\n' "$(stat -c %s "$f")" "$f"; done
done
} > "$P/results/not_copied.tsv"
echo "copy done: $(find "$P" -type f | wc -l) files"
