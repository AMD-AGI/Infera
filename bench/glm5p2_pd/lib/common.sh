#!/usr/bin/env bash
# Small shared helpers for the hand-operated multi-P/D scripts.
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$COMMON_DIR/.." && pwd)"
INFERA_ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
declare -a CLI_ASSIGNMENTS=()
declare -a SSH_ARGS=()

log() { printf '[%s] %s\n' "${COMPONENT:-glm5p2-pd}" "$*"; }
die() { printf '[%s] ERROR: %s\n' "${COMPONENT:-glm5p2-pd}" "$*" >&2; exit 1; }

print_command() {
    local item
    printf '+'
    for item in "$@"; do printf ' %q' "$item"; done
    printf '\n'
}

bool01() {
    case "${1:-}" in
        1|true|yes|on) printf '1\n' ;;
        0|false|no|off|"") printf '0\n' ;;
        *) die "expected a boolean, got '$1'" ;;
    esac
}

_absolute_existing_file() {
    local path="$1" parent
    parent="$(cd "$(dirname "$path")" 2>/dev/null && pwd)" ||
        die "directory does not exist: $(dirname "$path")"
    path="$parent/$(basename "$path")"
    [[ -r "$path" ]] || die "file is not readable: $path"
    printf '%s\n' "$path"
}

load_config() {
    local assignment name requested_config
    CLI_ASSIGNMENTS=("$@")
    for assignment in "$@"; do
        [[ "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] ||
            die "expected VAR=value, got '$assignment'"
        if [[ "${assignment%%=*}" == CONFIG ]]; then
            requested_config="${assignment#*=}"
        fi
    done

    CONFIG_FILE="$(_absolute_existing_file "${requested_config:-${CONFIG:-$BENCH_DIR/config.sh}}")"
    set -a
    # config.sh is intentionally executable shell, like rocm-llm-bench.
    source "$CONFIG_FILE"
    set +a

    # Command-line assignments are final, including intentional empty values.
    for assignment in "$@"; do
        name="${assignment%%=*}"
        printf -v "$name" '%s' "${assignment#*=}"
        export "$name"
    done

    TOPOLOGY_FILE="$(_absolute_existing_file "${TOPOLOGY:-$BENCH_DIR/topology.tsv}")"
    REMOTE_BENCH_DIR="${REMOTE_BENCH_DIR:-$BENCH_DIR}"
    LOG_DIR="${LOG_DIR:-$BENCH_DIR/logs}"
    export CONFIG="$CONFIG_FILE" CONFIG_FILE TOPOLOGY="$TOPOLOGY_FILE" TOPOLOGY_FILE
    export REMOTE_BENCH_DIR LOG_DIR
    validate_config
    validate_topology
}

validate_config() {
    local required value
    for required in IMAGE MODEL SERVED_MODEL CONTROL_NODE CONTAINER_PREFIX \
        NODE_GPU_COUNT PREFILL_GPU_DEVICES DECODE_GPU_DEVICES ETCD_PORT \
        ROUTER_PORT ENGINE_PORT_BASE BOOTSTRAP_PORT_BASE KV_EVENT_PORT_BASE \
        SNAPSHOT_PORT_BASE PREFILL_TP PREFILL_EP PREFILL_DP DECODE_TP \
        DECODE_EP DECODE_DP; do
        [[ -n "${!required:-}" ]] || die "config is missing $required"
    done
    [[ "$CONTROL_NODE" =~ ^[A-Za-z0-9][A-Za-z0-9_.@-]*$ ]] ||
        die "unsafe CONTROL_NODE=$CONTROL_NODE"
    [[ "$CONTAINER_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
        die "unsafe CONTAINER_PREFIX=$CONTAINER_PREFIX"

    for required in NODE_GPU_COUNT ETCD_PORT ROUTER_PORT ENGINE_PORT_BASE \
        BOOTSTRAP_PORT_BASE KV_EVENT_PORT_BASE SNAPSHOT_PORT_BASE PREFILL_TP \
        PREFILL_EP PREFILL_DP PREFILL_MAX_RUNNING PREFILL_GRAPH_MAX_BS \
        DECODE_TP DECODE_EP DECODE_DP DECODE_MAX_RUNNING DECODE_GRAPH_MAX_BS; do
        value="${!required:-}"
        [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$required must be a positive integer"
    done
    for required in PREFILL_DPA PREFILL_HICACHE DECODE_DPA DECODE_HICACHE DECODE_MTP; do
        bool01 "${!required:-0}" >/dev/null
    done
    [[ "$(bool01 "${DECODE_HICACHE:-0}")" == 0 || "$(bool01 "${DECODE_MTP:-0}")" == 0 ]] ||
        die "Decode MTP and HiCache cannot both be enabled"

    python3 - "$PREFILL_GPU_DEVICES" "$DECODE_GPU_DEVICES" \
        "$NODE_GPU_COUNT" "$PREFILL_TP" "$PREFILL_DP" "$DECODE_TP" "$DECODE_DP" <<'PY'
import sys

for role, text, tp, dp in (
    ("prefill", sys.argv[1], int(sys.argv[4]), int(sys.argv[5])),
    ("decode", sys.argv[2], int(sys.argv[6]), int(sys.argv[7])),
):
    devices = [part.strip() for part in text.split(",") if part.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise SystemExit(f"{role} GPU list is empty or contains duplicates")
    if any(not item.isdigit() for item in devices):
        raise SystemExit(f"{role} GPU list must contain numeric device IDs")
    if any(int(item) >= int(sys.argv[3]) for item in devices):
        raise SystemExit(f"{role} GPU list exceeds NODE_GPU_COUNT={sys.argv[3]}")
    if tp * dp != len(devices):
        raise SystemExit(
            f"{role}: TP({tp}) * DP({dp}) must equal selected GPUs({len(devices)})"
        )
PY
}

validate_topology() {
    python3 - "$TOPOLOGY_FILE" "$CONTROL_NODE" "$CONTAINER_PREFIX" \
        "$ETCD_PORT" "$ROUTER_PORT" "$ENGINE_PORT_BASE" "$BOOTSTRAP_PORT_BASE" \
        "$KV_EVENT_PORT_BASE" "$SNAPSHOT_PORT_BASE" \
        "${INFERA_NODEPORT_RANGE:-30000-32767}" <<'PY'
import csv
import ipaddress
import re
import sys

path, control, prefix = sys.argv[1:4]
service_ports = [int(value) for value in sys.argv[4:6]]
bases = [int(value) for value in sys.argv[6:10]]
low, high = (int(value) for value in sys.argv[10].split("-", 1))
with open(path, encoding="utf-8", newline="") as stream:
    reader = csv.DictReader(stream, delimiter="\t")
    if reader.fieldnames != ["role", "node", "data_ip"]:
        raise SystemExit(f"{path}: expected columns: role, node, data_ip")
    rows = list(reader)
if not rows:
    raise SystemExit(f"{path}: topology is empty")

seen_nodes, seen_ips = set(), set()
counts = {"prefill": 0, "decode": 0}
all_ports = list(service_ports)
for number, row in enumerate(rows, 2):
    role = (row.get("role") or "").strip().lower()
    node = (row.get("node") or "").strip()
    ip = (row.get("data_ip") or "").strip()
    if role not in counts:
        raise SystemExit(f"{path}:{number}: role must be prefill or decode")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", node):
        raise SystemExit(f"{path}:{number}: unsafe node name {node!r}")
    try:
        parsed_ip = ipaddress.ip_address(ip)
    except ValueError:
        raise SystemExit(f"{path}:{number}: invalid data_ip {ip!r}") from None
    if parsed_ip.version != 4:
        raise SystemExit(f"{path}:{number}: data_ip must be IPv4")
    if node in seen_nodes or ip in seen_ips:
        raise SystemExit(f"{path}:{number}: node and data_ip must be unique")
    seen_nodes.add(node)
    seen_ips.add(ip)
    counts[role] += 1
    index = number - 2
    all_ports.extend(base + index for base in bases)

if not all(counts.values()):
    raise SystemExit(f"{path}: at least one prefill and one decode are required")
if control not in seen_nodes:
    raise SystemExit(f"{path}: CONTROL_NODE={control!r} is not listed")
if any(not 1 <= port <= 65535 for port in all_ports):
    raise SystemExit("a configured port is outside 1..65535")
if len(all_ports) != len(set(all_ports)):
    raise SystemExit("configured service/worker ports overlap")
inside = [port for port in all_ports if low <= port <= high]
if inside:
    raise SystemExit(f"ports overlap NodePort range {low}-{high}: {inside}")
PY
}

topology_rows() {
    python3 - "$TOPOLOGY_FILE" "$CONTAINER_PREFIX" "$PREFILL_GPU_DEVICES" \
        "$DECODE_GPU_DEVICES" "$ENGINE_PORT_BASE" "$BOOTSTRAP_PORT_BASE" \
        "$KV_EVENT_PORT_BASE" "$SNAPSHOT_PORT_BASE" <<'PY'
import csv
import sys

path, prefix, p_gpus, d_gpus = sys.argv[1:5]
bases = [int(value) for value in sys.argv[5:9]]
counts = {"prefill": 0, "decode": 0}
with open(path, encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
for index, row in enumerate(rows):
    role = row["role"].strip().lower()
    role_index = counts[role]
    counts[role] += 1
    instance = f"{role}-{role_index}"
    values = [
        instance,
        role,
        row["node"].strip(),
        row["data_ip"].strip(),
        p_gpus if role == "prefill" else d_gpus,
        *(str(base + index) for base in bases),
        f"{prefix}-{instance}",
    ]
    print("\x1f".join(values))
PY
}

topology_nodes() {
    python3 - "$TOPOLOGY_FILE" <<'PY'
import csv
import sys
with open(sys.argv[1], encoding="utf-8", newline="") as stream:
    for row in csv.DictReader(stream, delimiter="\t"):
        print(row["node"].strip())
PY
}

topology_count() {
    local wanted="$1"
    python3 - "$TOPOLOGY_FILE" "$wanted" <<'PY'
import csv
import sys
with open(sys.argv[1], encoding="utf-8", newline="") as stream:
    print(sum(row["role"].strip().lower() == sys.argv[2] for row in csv.DictReader(stream, delimiter="\t")))
PY
}

node_ip() {
    local wanted="$1"
    python3 - "$TOPOLOGY_FILE" "$wanted" <<'PY'
import csv
import sys
with open(sys.argv[1], encoding="utf-8", newline="") as stream:
    for row in csv.DictReader(stream, delimiter="\t"):
        if row["node"].strip() == sys.argv[2]:
            print(row["data_ip"].strip())
            break
    else:
        raise SystemExit(f"node not found in topology: {sys.argv[2]}")
PY
}

router_url() { printf 'http://%s:%s\n' "$(node_ip "$CONTROL_NODE")" "$ROUTER_PORT"; }
service_container() { printf '%s-%s\n' "$CONTAINER_PREFIX" "$1"; }

print_topology() {
    local instance role node ip gpus engine bootstrap kv snapshot container
    printf '%-12s %-8s %-24s %-16s %-7s %s\n' INSTANCE ROLE NODE DATA_IP PORT GPUS
    while IFS=$'\x1f' read -r instance role node ip gpus engine bootstrap kv snapshot container; do
        printf '%-12s %-8s %-24s %-16s %-7s %s\n' \
            "$instance" "$role" "$node" "$ip" "$engine" "$gpus"
    done < <(topology_rows)
}

init_ssh() {
    local raw="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
    mapfile -d '' -t SSH_ARGS < <(
        python3 - "$raw" <<'PY'
import shlex
import sys
for item in shlex.split(sys.argv[1]):
    sys.stdout.buffer.write(item.encode() + b"\0")
PY
    )
}

_remote_command() {
    local result="" item quoted
    for item in "$@"; do
        printf -v quoted '%q' "$item"
        result+="${result:+ }$quoted"
    done
    printf '%s\n' "$result"
}

ssh_exec() {
    local node="$1" remote
    shift
    [[ "$node" =~ ^[A-Za-z0-9][A-Za-z0-9_.@-]*$ ]] || die "unsafe SSH node: $node"
    remote="$(_remote_command "$@")"
    command ssh "${SSH_ARGS[@]}" "$node" "$remote"
}

check_nodes_idle() {
    local instance role node ip gpus engine bootstrap kv snapshot container
    while IFS=$'\x1f' read -r instance role node ip gpus engine bootstrap kv snapshot container; do
        log "checking $instance on $node (GPUs $gpus)"
        ssh_exec "$node" bash -s -- \
            "$NODE_GPU_COUNT" "$gpus" "$GPU_IDLE_VRAM_PCT" \
            "$GPU_IDLE_TIMEOUT" "$CONTAINER_PREFIX" <<'REMOTE'
set -euo pipefail
expected_count="$1"
selected="$2"
idle_limit="$3"
idle_timeout="$4"
container_prefix="$5-"

running="$(docker ps -a --format '{{.Names}}' | awk -v prefix="$container_prefix" 'index($0, prefix) == 1')"
if [[ -n "$running" ]]; then
    echo "existing suite containers found; run stop.sh first:" >&2
    printf '%s\n' "$running" >&2
    exit 1
fi

deadline=$((SECONDS + idle_timeout))
while true; do
    report="$(rocm-smi --showmemuse 2>/dev/null)"
    read -r total selected_count maximum < <(
        awk -v wanted=",$selected," -F'): ' '
            /VRAM%/ {
                match($0, /GPU\[[0-9]+\]/)
                id = substr($0, RSTART + 4, RLENGTH - 5)
                total++
                if (index(wanted, "," id ",") != 0) {
                    selected_count++
                    value = $NF + 0
                    if (value > maximum) maximum = value
                }
            }
            END { print total + 0, selected_count + 0, maximum + 0 }
        ' <<<"$report"
    )
    wanted_count="$(awk -F, '{print NF}' <<<"$selected")"
    [[ "$total" == "$expected_count" ]] || {
        echo "expected $expected_count GPUs, rocm-smi reported $total" >&2
        exit 1
    }
    [[ "$selected_count" == "$wanted_count" ]] || {
        echo "cannot read VRAM for selected GPUs $selected" >&2
        exit 1
    }
    (( maximum <= idle_limit )) && break
    (( SECONDS < deadline )) || {
        echo "selected GPUs remained busy at ${maximum}% VRAM" >&2
        exit 1
    }
    echo "GPUs busy at ${maximum}% VRAM; waiting"
    sleep 10
done
printf 'idle: host=%s selected_gpus=%s max_vram_pct=%s\n' \
    "$(hostname)" "$selected" "$maximum"
REMOTE
    done < <(topology_rows)
}

record_image_ids() {
    local node image_id
    for node in "$@"; do
        image_id="$(ssh_exec "$node" docker image inspect --format '{{.Id}}' "$IMAGE")" ||
            die "$IMAGE is unavailable on $node"
        log "image node=$node ref=$IMAGE id=$image_id"
    done
}

start_log() {
    local stamp
    mkdir -p "$LOG_DIR"
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    RUN_LOG="${RUN_LOG:-$LOG_DIR/${COMPONENT:-run}-$stamp-$$.log}"
    export RUN_LOG
    exec > >(tee -a "$RUN_LOG") 2>&1
    log "log=$RUN_LOG"
}
