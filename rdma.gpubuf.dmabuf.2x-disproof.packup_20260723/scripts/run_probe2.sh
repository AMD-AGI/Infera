IMG=infera/engine-sglang:pd-final
CTR=ionic_probe
docker rm -f $CTR >/dev/null 2>&1
# mount the host's ABI-4 libionic file explicitly (54.0-187), not just the symlink
HOSTLIB=/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187
docker run -d --name $CTR --network=host \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -v /mnt/vast:/mnt/vast \
  -v $HOSTLIB:/host/libionic.so.1.1.54.0-187:ro \
  --entrypoint sleep "$IMG" infinity >/dev/null
# replace container's old libionic with the host ABI-4 one + fix the rdmav34 provider symlink
docker exec $CTR bash -lc '
set -e
cd /usr/lib/x86_64-linux-gnu
cp -f /host/libionic.so.1.1.54.0-187 ./libionic.so.1.1.54.0-187
ln -sf libionic.so.1.1.54.0-187 libionic.so.1
ln -sf libionic.so.1 libionic.so
# the ibverbs provider plugin (rdmav34) must point at the ABI-4 lib too
if [ -d libibverbs ]; then cd libibverbs; ln -sf ../libionic.so.1.1.54.0-187 libionic-rdmav34.so; cd ..; fi
# also check /etc/libibverbs.d or /usr/lib/.../libibverbs.d driver config
ldconfig 2>/dev/null
echo "container libionic now -> $(readlink -f libionic.so.1)"
echo "active_ports: $(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)"
echo "ibv devices: $(ibv_devices 2>/dev/null | tail -n +3 | wc -l)"
'
echo "===== RUN PROBE ====="
docker exec -e PROBE_GIB=4 $CTR bash /mnt/vast/c_huggingface/ionic_vram_probe/run.sh
docker rm -f $CTR >/dev/null 2>&1
