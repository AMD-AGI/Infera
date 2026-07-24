echo "=== TTM buddy free timeline (host debugfs ground truth) ==="
echo "ts        | free"
grep 'total:' /mnt/vast/c_huggingface/vmm_dmabuf.log | \
  sed -E 's/^([0-9:]+).*free:[[:space:]]*([0-9]+MiB).*/\1  \2/' | uniq -f1
echo "--- raw sample of the matched line ---"
grep -m1 'total:' /mnt/vast/c_huggingface/vmm_dmabuf.log
