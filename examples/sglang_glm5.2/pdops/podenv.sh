# Load the container's real environment. An ssh session into the pod does not
# inherit it; the platform's own terminal does. Export everything rather than a
# whitelist — the image sets ~15 SGLANG_*/AITER_* vars and dropping any of them
# silently changes the engine's behaviour (e.g. SGLANG_USE_AITER decides the DSA
# page size, so losing it makes the two PD legs disagree on the KV layout).
while IFS= read -r -d '' kv; do
  case "$kv" in
    _=*|PWD=*|OLDPWD=*|SHLVL=*) continue ;;
  esac
  export "$kv" 2>/dev/null || true
done < /proc/1/environ
