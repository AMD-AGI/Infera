# Sourced by the vllm unit steps: run TESTS (pytest args) on the PR worktree and on its base, compare.
#   IMG BRANCH BASE_WT BASE_REF TESTS (array) [K (pytest -k expression)]
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready "$NEED" || exit 1
w=$(wt "$BRANCH")
b=$(wt "$BASE_WT")
kexpr=${K:+-k \"$K\"}
echo "== PR $BRANCH ($(git -C "$w" rev-parse --short HEAD)) on $IMG"
pytest_in "$IMG" "$w" "$BASE_REF" "up-$STEP-pr" "-p no:randomly $kexpr ${TESTS[*]}" 2>&1 | tail -60
base_tests=()
for t in "${TESTS[@]}"; do [[ -e "$b/${t%%::*}" ]] && base_tests+=("$t"); done
if (( ${#base_tests[@]} )); then
    bc=$(base_cache "$IMG $(git -C "$b" rev-parse HEAD) $BASE_REF ${base_tests[*]} K=${K:-} P=${PIP_EXTRA:-}")
    if [[ -s "$bc" ]]; then
        cp "$bc" "$OUT/up-$STEP-base.xml"; echo "== base: reused $bc"
    else
        echo "== base $BASE_WT ($(git -C "$b" rev-parse --short HEAD))"
        pytest_in "$IMG" "$b" "$BASE_REF" "up-$STEP-base" "-p no:randomly $kexpr ${base_tests[*]}" 2>&1 | tail -30
        [[ -s "$OUT/up-$STEP-base.xml" ]] && mkdir -p "$(dirname "$bc")" && cp "$OUT/up-$STEP-base.xml" "$bc"
    fi
fi
python3 "$U/checks/junit_diff.py" "$OUT/up-$STEP-base.xml" "$OUT/up-$STEP-pr.xml" | tee "$OUT/verdict.txt"
exit "${PIPESTATUS[0]}"
