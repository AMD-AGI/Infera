# Edit a VLLM_ARGS array in place. args_set FLAG VALUE: replace FLAG's value, or append FLAG VALUE.
# args_drop FLAG: remove a valueless FLAG (e.g. --monokernel).
args_set() {
    local i
    for i in "${!VLLM_ARGS[@]}"; do
        if [ "${VLLM_ARGS[$i]}" = "$1" ]; then VLLM_ARGS[$((i + 1))]="$2"; return; fi
    done
    VLLM_ARGS+=("$1" "$2")
}
args_drop() {
    local a out=()
    for a in "${VLLM_ARGS[@]}"; do [ "$a" = "$1" ] || out+=("$a"); done
    VLLM_ARGS=("${out[@]}")
}
