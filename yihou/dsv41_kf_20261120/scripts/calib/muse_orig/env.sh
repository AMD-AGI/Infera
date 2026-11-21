# Sourced by the node-side calib profiler scripts (host side of crsuse2-m2m-v3-180, run via spur exec 285544).
NFS=/shared_nfs/yihou/muse_agentx_opt_20261027
CP=$NFS/calib_profiler                                  # staged copy of research/calib_profiler (no repo/)
DISK=/mnt/m2m_nobackup/yihou/muse_agentx_opt_20261027
LCP=$DISK/calib_profiler                                # node-local: src/ build/ out/ logs/
IMG=lmsysorg/sglang-rocm:muse-spark-open-mi355-1106@sha256:d231181df079a5b271a0459d024492c84a36caf8cb26ee019abad29e17eea4d2
# container-side paths (same mounts as scripts/server_launch.py: NFS root -> /yihou/runtime, DISK -> /yihou/local)
C_CP=/yihou/runtime/calib_profiler
C_SDK=/yihou/local/calib_profiler/build
case "$LCP" in *yihou*) ;; *) echo "LCP not in a yihou dir"; exit 1 ;; esac
