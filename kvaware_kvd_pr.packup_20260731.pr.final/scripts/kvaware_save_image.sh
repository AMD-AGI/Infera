#!/bin/bash
# Re-save the image WITHOUT gzip (vast has 128 TB; gzip -1 on 78 GB was the
# slow part) and fully detached, so an ssh disconnect cannot kill the pipeline.
# The previous attempt died mid-stream: `gzip -t` passed (the gzip layer was
# self-consistent) but `docker load` hit "unexpected EOF" because the TAR
# inside was truncated. Verify the tar, not the gzip.
DST=/mnt/vast/c_huggingface/kvaware_kvd_final
rm -f "$DST/kvaware-kvd.tar.gz" "$DST/kvaware-kvd.tar"
setsid nohup bash -c "
  docker save -o '$DST/kvaware-kvd.tar' infera/engine-sglang:kvaware-kvd
  echo \"docker_save_rc=\$?\" > '$DST/save.status'
  ls -la '$DST/kvaware-kvd.tar' >> '$DST/save.status'
" > /root/resave.log 2>&1 < /dev/null &
echo "detached save started"
