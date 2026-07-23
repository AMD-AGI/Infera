CTR=dsv4_pd_sgl
for h in chi2879 chi2865; do
  echo "=== teardown $h ==="
  ssh -o StrictHostKeyChecking=no $h "docker exec $CTR bash -lc 'pkill -9 -f infera.engine 2>/dev/null; pkill -9 -f sglang 2>/dev/null; pkill -9 -f multiprocessing.spawn 2>/dev/null; true' 2>/dev/null; docker rm -f $CTR >/dev/null 2>&1 && echo rm-$CTR; docker rm -f repro-etcd >/dev/null 2>&1 || true" 2>/dev/null
done
sleep 8
for h in chi2879 chi2865; do
  echo "=== $h GPU after teardown ==="
  ssh -o StrictHostKeyChecking=no $h "rocm-smi --showpids 2>/dev/null | grep -iE 'no kfd|python' | head -2; rocm-smi --csv --showmeminfo vram 2>/dev/null | tail -8 | awk -F, '{printf \"%.1fG \",\$3/1073741824}'; echo" 2>/dev/null
done
