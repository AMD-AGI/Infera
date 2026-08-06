# env/

| file | what |
|---|---|
| `env_crsuse2-m2m-010_prefill.txt` | full `collect_env.sh` snapshot, prefill node, taken **while live** right after the run |
| `env_crsuse2-m2m-081_decode.txt` | same, decode node |
| `env_armA2_prefill.sh` / `env_armA2_decode.sh` | the **exact** env files each leg was launched with |

The `.sh` files are ground truth for what each leg was *asked* to do;
`../environment.md` records what the engine actually *resolved*, read back from
the boot log.

> Reading the snapshots, note that both nodes show **8 ACTIVE ionic rails plus
> mlx5_0**. Only mlx5_0 was used — see `../environment.md` § RDMA fabric. Do not
> infer from the rail list that the run used ionic.
