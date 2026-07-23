IMG=infera/engine-sglang:pd-final; CTR=hold_big
docker rm -f $CTR >/dev/null 2>&1
HOSTLIB=/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187
docker run -d --name $CTR --network=host --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -v /mnt/vast:/mnt/vast -v $HOSTLIB:/host/libionic.so.1.1.54.0-187:ro --entrypoint sleep "$IMG" infinity >/dev/null
docker exec $CTR bash -lc 'cd /usr/lib/x86_64-linux-gnu; cp -f /host/libionic.so.1.1.54.0-187 .; ln -sf libionic.so.1.1.54.0-187 libionic.so.1; ln -sf libionic.so.1 libionic.so; [ -d libibverbs ] && (cd libibverbs && ln -sf ../libionic.so.1.1.54.0-187 libionic-rdmav34.so); ldconfig 2>/dev/null' >/dev/null
sumvram(){ cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | awk '{s+=$1} END{printf "%.2f", s/1073741824}'; }
# 64 GiB buffer — PD KV-pool scale; if 2x were real this would eat 128GiB
docker exec -d -e PROBE_GIB=64 -e HOLD_S=5 $CTR bash -lc 'python3 /mnt/vast/c_huggingface/ionic_vram_probe/hold_probe.py > /tmp/hb.out 2>&1'
for i in $(seq 1 16); do
  ph=$(docker exec $CTR bash -lc 'tail -1 /tmp/hb.out 2>/dev/null')
  echo "t=$((i*2))s amdgpu_vram_all=$(sumvram)GiB | $ph"
  sleep 2
done
docker rm -f $CTR >/dev/null 2>&1
