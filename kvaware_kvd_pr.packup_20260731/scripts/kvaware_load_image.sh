#!/bin/bash
KIT=/mnt/vast/c_huggingface/kvaware_kvd_final
setsid nohup bash -c "
  docker load -i '$KIT/kvaware-kvd.tar'
  echo \"docker_load_rc=\$?\" > /root/load.status
  docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep kvaware-kvd >> /root/load.status
" > /root/load2.log 2>&1 < /dev/null &
echo "detached load started"
