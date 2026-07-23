IMG=infera/engine-sglang:pd-final
CTR=dmabuf_dbg
docker rm -f $CTR >/dev/null 2>&1
HOSTLIB=/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187
docker run -d --name $CTR --network=host --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -v /mnt/vast:/mnt/vast -v $HOSTLIB:/host/libionic.so.1.1.54.0-187:ro \
  --entrypoint sleep "$IMG" infinity >/dev/null
docker exec $CTR bash -lc 'cd /usr/lib/x86_64-linux-gnu; cp -f /host/libionic.so.1.1.54.0-187 .; ln -sf libionic.so.1.1.54.0-187 libionic.so.1; ln -sf libionic.so.1 libionic.so; [ -d libibverbs ] && (cd libibverbs && ln -sf ../libionic.so.1.1.54.0-187 libionic-rdmav34.so); ldconfig 2>/dev/null; echo "active_ports: $(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)"'
echo "===== DEBUG PROBE (v1 vs v2/NONE vs v2/PCIE) ====="
docker exec -e PROBE_GIB=4 $CTR python3 /mnt/vast/c_huggingface/ionic_vram_probe/dmabuf_debug.py
docker rm -f $CTR >/dev/null 2>&1
