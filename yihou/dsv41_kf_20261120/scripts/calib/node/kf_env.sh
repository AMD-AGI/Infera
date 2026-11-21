# kf calib profiler paths (openai node). Host and container see the same paths (bind mounts at identical paths).
source /perf_apps/yihou/dsv41_kf_20261120/pd/scripts/env.sh     # IMAGE, CALIB_LOCAL, W
CP=$W/calib                                 # staged scripts/calib (+ stage/ source tarball, wheels)
LCP=${LCP_OVERRIDE:-$CALIB_LOCAL}                            # node-local: src-<dig>/ build/ logs/ out/
case "$LCP" in */yihou/*) ;; *) echo "LCP not in a yihou dir"; exit 1 ;; esac
