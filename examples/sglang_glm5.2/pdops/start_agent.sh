#!/usr/bin/env bash
# One-off bootstrap so the master pod can drive this pod over ssh.
# Run this ONCE inside the worker pod:  bash pdops/start_agent.sh
#
# Keys are NOT in git. Generate the client pair once on the master pod, into a
# directory both pods can read (KEYDIR, default: this script's directory):
#   ssh-keygen -t ed25519 -N '' -f "$KEYDIR/id_ed25519"
# Host keys are generated here on first run.
set -euo pipefail

D="${KEYDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PORT="${PORT:-2223}"

if [[ ! -f "$D/id_ed25519.pub" ]]; then
  echo "missing $D/id_ed25519.pub -- generate the client pair first (see header)" >&2
  exit 1
fi

mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
cp -f "$D/id_ed25519.pub" /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# Host keys live on the shared fs so the master already trusts them.
for t in rsa ed25519; do
  [[ -f "$D/hostkey_$t" ]] || ssh-keygen -q -t "$t" -N '' -f "$D/hostkey_$t"
done

cat > "$D/sshd_config" <<EOF
Port ${PORT}
ListenAddress 0.0.0.0
HostKey ${D}/hostkey_rsa
HostKey ${D}/hostkey_ed25519
PermitRootLogin prohibit-password
PubkeyAuthentication yes
PasswordAuthentication no
AuthorizedKeysFile /root/.ssh/authorized_keys
UsePAM no
PidFile ${D}/sshd_$(hostname -s).pid
EOF

pkill -f "sshd_config" 2>/dev/null || true
/usr/sbin/sshd -f "$D/sshd_config"
sleep 1
echo "sshd listening on $(hostname -s):${PORT}"
ss -tln | grep ":${PORT} " || { echo "FAILED to bind ${PORT}"; exit 1; }
