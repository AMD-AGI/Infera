# E7: 1P1D PD bring-up (same node first) — DSv4.1-Flash, vLLM MooncakeConnector, vllm-router 0.1.14 PD mode

Leader-finished 2026-11-19 ~15:0xZ after two builder stalls (team-polls #18-#20). Static checks done on a16-19:
`bash -n` all kit scripts, `py_compile pd_smoke.py`, colocated r1/r2 renders unchanged (vLLM flags identical to the 09:xx
render; nofile/timeout/kit-ro present), e7 P/D renders differ only in kv_role, ports, device ids and D
`--no-enable-prefix-caching` (kv-cache-dtype fp8_ds_mla, speculative-config, attention-config identical on both).
NOT verified: anything at runtime (Mooncake RDMA on ionic in image 60bd, router /query bootstrap, DP2 prefill pool).

K=/home/yihou/dev/git.16-19/infera.dsv41/yihou/dsv41_vllm_pd_20261119/scripts/kit ; run ON the granted node; `bash $K/stage.sh`
on a16-19 first (docker cannot read HOME NFS).

1. Correctness smoke (real acceptance, "standard" rejection on both sides):
   - reference: `bash $K/serve.sh e7r-colo-dsmla-ref-smoke 4,5 e7r-<ts> --port-offset 40 --wait`
   - PD:        `bash $K/serve_pd.sh e7s-pd-mooncake-1p1d-dp2-smoke 0,1 2,3 e7s-<ts> --wait`
   - `bash $K/pd_smoke.sh e7s-<ts> http://127.0.0.1:8040 --docker` -> exit 0 PASS / 1 CHECK / 2 unreachable / 3 log failure.
   - If RDMA init fails: read runs/e7s-<ts>-{p,d}/server.log; fallbacks in order: MOONCAKE_DEVICE_NAME=NUMA-local ionic list,
     DOCKER_IONIC_PROVIDER=1, then mooncake_protocol tcp (bring-up only, never a perf config).
2. Perf point (synthetic 3.51, as colocated): `bash $K/serve_pd.sh e7-pd-mooncake-1p1d-dp2 0,1 2,3 e7-<ts> --wait` then
   `run_point.sh e7-<ts> http://127.0.0.1:8000 128 4 official` (GPU denominator 4 = P 2 + D 2).
   Compare against colocated at equal GPUs: 2 x R1 (DP2, 2 GPUs each) — PD must beat that by more than the noise band (~4%).
3. Stop: `bash $K/stop.sh <tag> --prefix`.

Known risks: R1 prefill DP2 pool behind Mooncake untested anywhere; router 0.1.14 falls back to the first engine id if
/query misses a rank (serve_pd pre-waits); PD forces fp8_ds_mla + Triton sparse path (-13.5% colocated at c128).
