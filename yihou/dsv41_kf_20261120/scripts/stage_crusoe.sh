#!/bin/bash
# Stage the kf P/D + calib kit and the prebuilt fork SDK onto crusoe NFS (/shared_nfs/yihou/dsv41_kf_20261120). Run on a16-19
# after stage_pd.sh (copies the /perf_apps stage, which already has calib/ inside pd/).
set -euo pipefail
SRC=/perf_apps/yihou/dsv41_kf_20261120
DST=/shared_nfs/yihou/dsv41_kf_20261120
H=${SPUR_HOST:-spur005}
ssh -o BatchMode=yes "$H" "mkdir -p $DST/pd $DST/calib-build"
rsync -a --exclude logs --exclude wheels/extra/py-spy -e "ssh -o BatchMode=yes" "$SRC/pd/" "$H:$DST/pd/"
rsync -a -e "ssh -o BatchMode=yes" "$SRC/pd/wheels/extra/py-spy" "$H:$DST/pd/wheels/extra/" 2>/dev/null || true
rsync -a -e "ssh -o BatchMode=yes" "$SRC/calib-build-a1619/build/lib" "$SRC/calib-build-a1619/build/BUILD_OK" "$H:$DST/calib-build/"
ssh -o BatchMode=yes "$H" "mkdir -p $DST/pd/logs && chmod 1777 $DST/pd/logs $DST/pd/gem && chmod -R go-w $DST/pd/scripts $DST/pd/configs $DST/pd/calib; du -sh $DST/pd $DST/calib-build"
echo "staged to $H:$DST"
