W=/shared_nfs/yihou_glm52/g4tune_20261019
for v in 1 0; do
  for try in 1 2 3 4 5 6; do
    timeout 400 spur exec 225141 bash -c "cd /tmp; timeout 380 docker run --rm --name yihou-mcbw5 --network host --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined -e YX_IP=10.245.145.196 -e YX_LOADSIDE=send -e HSA_ENABLE_SDMA=$v -e MC_TE_FILTERS=yihou_no_such_rdma_device -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 -e MC_GID_INDEX=1 -e MC_LOG_LEVEL=INFO -v $W/machine:$W/machine infera-sglang:v0521-yihou-1010-mc4340 python3 $W/machine/mc_bw4.yihou.py > $W/machine/mc_bw5.sdma$v.v3-021.yihou.log 2>&1; echo rc=\$?" 2>&1 | tail -1 | grep -q "rc=" && break
    sleep 20
  done
  echo "== HSA_ENABLE_SDMA=$v (try $try)"; grep "^{" $W/machine/mc_bw5.sdma$v.v3-021.yihou.log 2>/dev/null | cut -c1-170
done
