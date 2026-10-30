#!/usr/bin/env bash
# Queue a point srtctl already rendered (a dir holding its sbatch_script.sh and config.yaml, e.g. the
# job archive) to run inside the holder allocation. The item copies both into
# <checkout>/outputs/holder-<holder job>-<name>/ and runs the script there; do_sweep and everything
# below it are what the submitted job runs. Optionally the model container is replaced (a staged squash).
#   holder_item.sh <NN-name> <rendered dir> [container]
set -euo pipefail
name="${1:?NN-name}"; src="$(cd "${2:?rendered dir}" && pwd)"; container="${3:-}"
H=/apps/liyingli/holder
source_dir="$(sed -n 's/^SRTCTL_SOURCE="\(.*\)"$/\1/p' "$src/sbatch_script.sh")"
[ -d "$source_dir" ] || { echo "srt-slurm checkout $source_dir is gone" >&2; exit 1; }
mkdir -p "$H/queue" "$H/items/$name"
cp "$src/sbatch_script.sh" "$src/config.yaml" "$H/items/$name/"
# The environment srtctl handed to sbatch (render_point.sh); a submitted job inherits it.
[ -f "$src/env0" ] && cp "$src/env0" "$H/items/$name/"
if [ -n "$container" ]; then
    python3 - "$H/items/$name/config.yaml" "$container" <<'PY'
import sys, re
path, container = sys.argv[1:]
text = open(path).read()
new, n = re.subn(r"(\nmodel:\n(?:  .*\n)*?  container:)[^\n]*\n(?:    \S[^\n]*\n)?", rf"\1 {container}\n", text, count=1)
assert n == 1, "model.container not found"
open(path, "w").write(new)
PY
fi
cat > "$H/queue/$name.sh.tmp" <<EOF
#!/usr/bin/env bash
# holder item $name, rendered by srtctl in $src
set -euo pipefail
dst="$source_dir/outputs/holder-\${SLURM_JOB_ID}-$name"
mkdir -p "\$dst/logs"
cp "$H/items/$name/config.yaml" "\$dst/config.yaml"
sed 's|^OUTPUT_DIR=.*|OUTPUT_DIR="'"\$dst"'"|' "$H/items/$name/sbatch_script.sh" > "\$dst/sbatch_script.sh"
echo "item $name: output \$dst"
if [ -f "$H/items/$name/env0" ]; then
    # Restore the submit-time environment, keeping this allocation's own Slurm variables.
    while IFS= read -r -d '' kv; do
        case "\${kv%%=*}" in SLURM_*|SBATCH_*|SALLOC_*|SRUN_*|PWD|OLDPWD|SHLVL|_|HOSTNAME|BASH_FUNC_*) ;; *) export "\$kv" ;; esac
    done < "$H/items/$name/env0"
fi
bash "\$dst/sbatch_script.sh" > "\$dst/logs/sweep_holder.log" 2>&1
EOF
mv "$H/queue/$name.sh.tmp" "$H/queue/$name.sh"
echo "queued $name ($src${container:+, container $container})"
