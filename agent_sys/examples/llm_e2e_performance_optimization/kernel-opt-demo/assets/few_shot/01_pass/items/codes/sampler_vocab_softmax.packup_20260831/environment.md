# Environment

| | |
|---|---|
| Host | `smc300x-ccs-aus-a17-31`, Ubuntu 22.04.5, kernel 6.5.0-45 |
| CPU | 2× AMD EPYC 9554, 256 logical cores, 1.5 TiB RAM |
| GPU | 8× AMD Instinct MI300X OAM, `gfx942`, 192 GB HBM3, NPS1/SPX, 304 CUs |
| GPU used | one card, via `HIP_VISIBLE_DEVICES` |
| Driver | amdgpu 7.1.3.31500000 |
| Image | `lmsysorg/sglang:v0.5.14-rocm720-mi30x`, digest `sha256:dab8984486941438b3002734b38d1566e47f4ef4fbc20890d8122dcc83b49928` |
| ROCm | 7.2.0 · PyTorch 2.9.1+rocm7.2.0 · Triton 3.6.0 · Python 3.10.12 |
| rocprof-compute | 3.4.0 (release) |
| sglang | v0.5.14, commit `49e384ce9d304648e9959666ecb8ce8cd98d0deb` |
| KernelForge | `main` @ `cd9c5850699b0550c2aa06be83c3645cf4e98e24` |
| Forge model | `Claude-Sonnet-5[1m]`, 49 calls, $29.11 |
| Campaign | experiment `fde6cf6c`, 6 iterations, 171.3 min, best iteration 5 |

Installed on top of the image:

```sh
pip install -e ".[dev]"
pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt
```

The second is required. KernelForge pins `astunparse==1.6.3`/`kaleido==1.3.0`
and ROCm 7.2's rocprofiler-compute 3.4.0 needs `1.6.2`/`0.2.1`; ROCm's side
wins, pip warns, and the warning is expected.

Not captured: no GPU clock lock, no exclusive reservation. Other tenants'
containers were running and idle (0% util) but were not excluded.
