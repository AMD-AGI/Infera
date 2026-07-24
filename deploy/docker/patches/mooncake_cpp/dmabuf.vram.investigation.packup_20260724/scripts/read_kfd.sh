echo "===== hs_dmabuf.log (TTM aligned) ====="
cat /mnt/vast/c_huggingface/hs_dmabuf.log 2>/dev/null
echo
echo "===== dmesg_dmabuf.log (KFD/pin/reserve during run) ====="
cat /mnt/vast/c_huggingface/dmesg_dmabuf.log 2>/dev/null
echo "[[lines: $(wc -l < /mnt/vast/c_huggingface/dmesg_dmabuf.log 2>/dev/null)]]"
