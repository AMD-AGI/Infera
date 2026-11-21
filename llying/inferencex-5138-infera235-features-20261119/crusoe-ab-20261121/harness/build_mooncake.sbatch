#!/usr/bin/env bash
#SBATCH -J lyl-ix5138-mcbuild
#SBATCH -p amd-spur-v2
#SBATCH -A amd-agentx
#SBATCH -q amd-burst-qos
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem 128G
#SBATCH -t 01:30:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/ix5138-crusoe-20261120/logs/mcbuild-%j.out
# Builds Mooncake 748af1a5 (the stock image's) + #4540 + #4541 (src/mooncake, branch ix5138-748af1a5-4540-4541) inside
# the stock image exactly as SGLang's docker/rocm.Dockerfile does, and packs every file `make install` wrote into
# mooncake/mooncake-overlay.tar, which workers that need these two Mooncake fixes for this cluster's RDMA (crusoe only)
# unpack over the image at container start. No GPU. Every step exits 0.
source /home/liyingli/bench_agentx/ix5138-crusoe-20261120/harness/lib.sh
NODE_HOSTS=("$(hostname -s)")
W=$LOCAL/mooncake-build-$SLURM_JOB_ID
say "mooncake build job $SLURM_JOB_ID on $(hostname -s), $(nproc) cpus"
timeout 30 docker info >/dev/null 2>&1 || { say "docker unusable"; exit 0; }
timeout 2400 docker pull -q "$IMAGE" >/dev/null || { say "image pull failed"; exit 0; }
mkdir -p "$W/out" && cp -a "$R/src/mooncake" "$W/Mooncake" && chmod -R a+rwX "$W" || { say "staging the source failed"; exit 0; }
say "building in $W"
timeout 4800 docker run --rm --name "lyl-ix5138-mcbuild-$SLURM_JOB_ID" --network host -v "$W/Mooncake:/build/Mooncake" \
    -v "$W/out:/out" --entrypoint bash "$IMAGE" -c '
set -e
# The source tree is owned by the submitting user, not by root.
git config --global --add safe.directory "*"
cd /build/Mooncake
echo "source: $(git log -1 --format="%h %s")"
bash dependencies.sh -y >/out/deps.log 2>&1 || { echo "dependencies.sh failed"; tail -20 /out/deps.log; exit 1; }
rm -rf /usr/local/go
wget -q https://go.dev/dl/go1.22.2.linux-amd64.tar.gz && tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz && rm go1.22.2.linux-amd64.tar.gz
export PATH=/usr/local/go/bin:$PATH
mkdir -p build && cd build
cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON -DENABLE_MULTI_PROTOCOL=ON -DWITH_STORE=ON -DBUILD_UNIT_TESTS=OFF >/out/cmake.log 2>&1 ||
    { echo "cmake failed"; tail -30 /out/cmake.log; exit 1; }
make -j "$(nproc)" >/out/make.log 2>&1 || { echo "make failed"; grep -m5 -B2 -A5 "error" /out/make.log; exit 1; }
make install >/out/install.log 2>&1 || { echo "make install failed"; tail -20 /out/install.log; exit 1; }
mc=$(python3 -c "import mooncake, os; print(os.path.dirname(mooncake.__file__))")
# CMake keeps the build tree timestamps on install, so the file list is its install manifest, plus the whole Python
# package in case part of it is installed by other means.
{ cat install_manifest.txt; echo; find "$mc" \( -type f -o -type l \) -not -name "*.pyc"; } | grep -v "^$" | sort -u >/out/installed.txt
tar -cf /out/mooncake-overlay.tar -T /out/installed.txt
python3 -c "from mooncake.engine import TransferEngine; print(\"import ok\", TransferEngine)"
echo "installed files: $(wc -l </out/installed.txt); python package at $mc"
for v in MC_RDMA_WRITE_PREFER_DEST_NIC MC_IB_TIMEOUT MC_RDMA_NOTIFY_ENABLED; do
    echo "$v: $(grep -rlaF "$v" "$mc" 2>/dev/null | wc -l) file(s) in the package"
done
' 2>&1 | tail -40
rc=${PIPESTATUS[0]}
say "build container rc=$rc"
if [[ -s "$W/out/mooncake-overlay.tar" ]]; then
    mkdir -p "$R/mooncake" && cp "$W/out/mooncake-overlay.tar" "$W/out/installed.txt" "$W/out/cmake.log" "$R/mooncake/" &&
        say "overlay: $(du -h "$R/mooncake/mooncake-overlay.tar" | cut -f1), $(wc -l <"$R/mooncake/installed.txt") files"
    grep -E "site-packages/mooncake|/usr/local/(lib|bin)" "$R/mooncake/installed.txt" | head -20
fi
docker run --rm -v "$LOCAL:$LOCAL" --entrypoint rm "$IMAGE" -rf "$W" >/dev/null 2>&1
say "done"
exit 0
