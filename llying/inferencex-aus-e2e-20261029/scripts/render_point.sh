#!/usr/bin/env bash
# Render one infera-sglang matrix point exactly as CI does (manual_launch -> infx.launch -> srtctl apply)
# but without submitting: a stand-in `sbatch` first on PATH records the environment srtctl hands to
# sbatch (env -0), its arguments and the batch script, and answers with a fake job id; once srtctl has
# written outputs/<fake id>/config.yaml the launcher is stopped. The holder then runs that script with
# that environment inside its own allocation.
#   render_point.sh <label> <exp-name> [manual_launch options]  -> /apps/liyingli/holder/render/<label>/
set -euo pipefail
label="${1:?label}"; exp="${2:?exp-name}"; shift 2
E2E="${E2E:?E2E (the inferencex-e2e workspace of this point)}"
out="/apps/liyingli/holder/render/$label"
rm -rf "$out"; mkdir -p "$out/bin"
fake="$(( 90000000 + $(date +%s) % 9999999 ))"
cat > "$out/bin/sbatch" <<EOF
#!/usr/bin/env bash
# stand-in sbatch for render_point.sh: record, do not submit
env -0 > "$out/env0"
printf '%s\n' "\$@" > "$out/sbatch_args"
for a in "\$@"; do [ -f "\$a" ] && cp "\$a" "$out/sbatch_script.sh"; done
echo "Submitted batch job $fake"
EOF
chmod +x "$out/bin/sbatch"
PATH="$out/bin:$PATH" E2E="$E2E" setsid bash /apps/liyingli/jobs/launch_point.sh "render-$label" "$exp" "$@" > "$out/launch.out"
for _ in $(seq 1 120); do
    cfg="$(ls -d "$E2E"/srt-slurm-*/outputs/"$fake"/config.yaml 2>/dev/null | head -1 || true)"
    [ -n "$cfg" ] && [ -s "$out/env0" ] && break
    sleep 5
done
[ -n "${cfg:-}" ] || { echo "render $label: srtctl wrote no config for fake job $fake" >&2; tail -20 "/apps/liyingli/runs/render-$label.log" >&2; exit 1; }
sleep 3
cp "$cfg" "$out/config.yaml"
pid="$(pgrep -f "exp-names $exp" | head -1 || true)"
if [ -n "$pid" ]; then
    sid="$(ps -o sid= -p "$pid" | tr -d '[:space:]')"
    pkill -TERM -s "$sid" || true
fi
sleep 3
echo "rendered $label: $out (fake job $fake)"
