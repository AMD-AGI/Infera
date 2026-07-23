# poll both legs up to ~28min for ready OR fatal
PF_H=chi2879; DC_H=chi2865; CTR=dsv4_pd_sgl
for i in $(seq 1 56); do
  pf=$(ssh -o StrictHostKeyChecking=no $PF_H "docker exec $CTR bash -lc 'tail -1 /tmp/pd_sgl_prefill_30000.log 2>/dev/null'" 2>/dev/null)
  dc=$(ssh -o StrictHostKeyChecking=no $DC_H "docker exec $CTR bash -lc 'tail -1 /tmp/pd_sgl_decode_30000.log 2>/dev/null'" 2>/dev/null)
  # ready markers
  pfr=$(ssh -o StrictHostKeyChecking=no $PF_H "docker exec $CTR bash -lc 'grep -ciE \"ready to roll|Uvicorn running|Application startup complete\" /tmp/pd_sgl_prefill_30000.log 2>/dev/null'" 2>/dev/null)
  dcr=$(ssh -o StrictHostKeyChecking=no $DC_H "docker exec $CTR bash -lc 'grep -ciE \"ready to roll|Uvicorn running|Application startup complete\" /tmp/pd_sgl_decode_30000.log 2>/dev/null'" 2>/dev/null)
  # fatal markers
  pff=$(ssh -o StrictHostKeyChecking=no $PF_H "docker exec $CTR bash -lc 'grep -ciE \"ibv_reg_mr|Cannot allocate memory|Traceback|HIP error|out of memory|Assertion|SIGABRT|core dump|exited before\" /tmp/pd_sgl_prefill_30000.log 2>/dev/null'" 2>/dev/null)
  dcf=$(ssh -o StrictHostKeyChecking=no $DC_H "docker exec $CTR bash -lc 'grep -ciE \"ibv_reg_mr|Cannot allocate memory|Traceback|HIP error|out of memory|Assertion|SIGABRT|core dump|exited before\" /tmp/pd_sgl_decode_30000.log 2>/dev/null'" 2>/dev/null)
  echo "[$i] PF ready=$pfr fatal=$pff | DC ready=$dcr fatal=$dcf"
  echo "    PF: $pf"
  echo "    DC: $dc"
  if [ "${pfr:-0}" -ge 1 ] && [ "${dcr:-0}" -ge 1 ]; then echo "BOTH_READY"; break; fi
  if [ "${pff:-0}" -ge 1 ] || [ "${dcf:-0}" -ge 1 ]; then echo "FATAL_DETECTED"; break; fi
  sleep 30
done
