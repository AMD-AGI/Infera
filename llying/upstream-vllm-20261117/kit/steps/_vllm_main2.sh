# Sourced by the main2-* steps: as _vllm_main.sh, but against the newer main in wt/main2 (11-19 15:30: 8828834fb), which
# the -main branches were rebased onto before opening the PRs. The newest ROCm nightly is still 9c1d45c2; between it
# and 8828834fb the only ROCm compiled change is csrc/fused_qknorm_rope_kernel.cu (#68386, an int32 offset fix in the
# fused QK-norm RoPE kernel), which the U13 / U19 tests do not use, so that one file is excepted as well.
#   BRANCH (the -main branch), TESTS (array), K (optional pytest -k), PIP_EXTRA (optional)
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready nightly || exit 1
w=$(wt "$BRANCH")
b=$(wt main2)
for t in "$b" "$w"; do
    compiled=$(git -C "$t" diff --name-only "$NIGHTLY_COMMIT" HEAD -- csrc/ cmake/ CMakeLists.txt setup.py \
        | grep -v -E '^csrc/cpu/|^cmake/cpu_extension\.cmake|^cmake/external_projects/vllm_flash_attn\.cmake|^csrc/fused_qknorm_rope_kernel\.cu$')
    [[ -z "$compiled" ]] || { echo "$t changes compiled code vs the image ($NIGHTLY_COMMIT): $compiled -- rebuild needed"; exit 2; }
done
kexpr=${K:+-k \"$K\"}
run_tree() {  # NAME TREE TESTS...
    local name=$1 tree=$2
    shift 2
    mkdir -p "$NL/pip"
    docker rm -f "$name" > /dev/null 2>&1
    # shellcheck disable=SC2046
    # The tree's own vllm/ must not shadow the synced install (it has no compiled extensions), so the tests run
    # from /work with only tests/ and pyproject.toml of the tree.
    docker run --rm --name "$name" $(gpu_args) ${EXTRA_DOCKER:-} -v "$tree:/src:ro" -v "$tree/tests:/work/tests:ro" \
        -v "$tree/pyproject.toml:/work/pyproject.toml:ro" -v "$U/checks:/checks:ro" \
        -v "$OUT:/out" -v "$NL/pip:/root/.cache/pip" -e PYTHONDONTWRITEBYTECODE=1 \
        -e PYTEST_ADDOPTS="-p no:cacheprovider" -w /work --entrypoint bash "$IMG_NIGHTLY" -c "
pip install -q $PIP_TEST ${PIP_EXTRA:-} > /out/pip-$name.log 2>&1
python3 /checks/sync_vllm.py /src
python3 -c 'import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, \"hip\", torch.version.hip)'
if [ -z \"${PER_FILE:-}\" ]; then
  python3 -m pytest -v -rfEs --junitxml=/out/$name.xml $kexpr ${PYTEST_EXTRA:-} $* > /out/$name.out 2>&1; rc=\$?
else
  # One process per test path (a process that dies takes only its own file); a path whose process died is retried once.
  parts=; i=0; rc=0
  for t in $*; do
    i=\$((i + 1)); x=/out/$name.part\$i.xml
    for attempt in 1 2; do
      python3 -m pytest -v -rfEs --junitxml=\$x $kexpr ${PYTEST_EXTRA:-} \$t >> /out/$name.out 2>&1 || rc=1
      [ -s \$x ] && break
      echo \"== \$t: process died (attempt \$attempt)\" >> /out/$name.out
    done
    parts=\"\$parts \$x\"
  done
  python3 /checks/junit_merge.py /out/$name.xml \$parts
fi
tail -40 /out/$name.out; exit \$rc"
}
echo "== base main $(git -C "$b" rev-parse --short HEAD) / PR $BRANCH $(git -C "$w" rev-parse --short HEAD) on $IMG_NIGHTLY"
run_tree "up-$STEP-pr" "$w" "${TESTS[@]}" 2>&1 | tail -60
base_tests=()
for t in "${TESTS[@]}"; do [[ -e "$b/${t%%::*}" ]] && base_tests+=("$t"); done
if (( ${#base_tests[@]} )); then
    bc=$(base_cache "$IMG_NIGHTLY $(git -C "$b" rev-parse HEAD) ${base_tests[*]} K=${K:-} X=${PYTEST_EXTRA:-} F=${PER_FILE:-} $(sha256sum < "$U/checks/sync_vllm.py")")
    if [[ -s "$bc" ]]; then
        cp "$bc" "$OUT/up-$STEP-base.xml"; echo "== base: reused $bc"
    else
        echo "== base"
        run_tree "up-$STEP-base" "$b" "${base_tests[@]}" 2>&1 | tail -30
        [[ -s "$OUT/up-$STEP-base.xml" ]] && mkdir -p "$(dirname "$bc")" && cp "$OUT/up-$STEP-base.xml" "$bc"
    fi
fi
python3 "$U/checks/junit_diff.py" "$OUT/up-$STEP-base.xml" "$OUT/up-$STEP-pr.xml" | tee "$OUT/verdict.txt"
exit "${PIPESTATUS[0]}"
