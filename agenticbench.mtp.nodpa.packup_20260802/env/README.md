# Environment capture

| file | what |
|---|---|
| `env_crsuse2-m2m-231.txt` | live `collect_env.sh` on the **prefill** node — GPU/driver, RDMA rails, CPU, kernel, docker |
| `env_crsuse2-m2m-276.txt` | same on the **decode** node |
| `chunk65536_prefill_server_args.txt` | prefill boot-time `server_args=` for the **MAIN** arm (global chunk 65,536) |
| `chunk8192_prefill_server_args.txt` | prefill boot-time `server_args=` for the **chunk-control** arm (global chunk 8,192) |
| `nodpa_decode_server_args.txt` | the decode leg's boot-time `server_args=` line |

## The `server_args=` files are the proof of the variable

On this stack the engine **rewrites flags**, so the launch command is not evidence.
`server_args=` is what actually ran. These two files are the only artifact in the
kit that proves DP-attention was off on prefill and on for decode:

```bash
for f in env/chunk*_prefill_server_args.txt env/nodpa_decode_server_args.txt; do
  echo -n "$(basename $f): "
  grep -oE "enable_dp_attention=[A-Za-z]+|chunked_prefill_size=[0-9]+|mem_fraction_static=[0-9.]+" $f | tr '\n' ' '; echo
done
```

    chunk65536_prefill: mem_fraction_static=0.7  chunked_prefill_size=65536 enable_dp_attention=False
    chunk8192_prefill:  mem_fraction_static=0.7  chunked_prefill_size=8192  enable_dp_attention=False
    nodpa_decode:       mem_fraction_static=0.85 chunked_prefill_size=8192  enable_dp_attention=True

**`chunked_prefill_size` is a GLOBAL budget that DPA divides by `dp_size`**
(`server_args.py:4902`). So `65536` on the DPA-off prefill leg and `8192` on the
DPA-on one are **the same 65,536 tokens per step machine-wide** — the former on
one rank, the latter as 8,192 × 8 ranks. Reading these files without that fact
inverts the comparison.

The decode leg does not run prefill, so its `chunked_prefill_size` is inert and is
not part of the matched configuration.

Reading that table row by row: **`enable_dp_attention` is the only entry that
differs by design**; `mem_fraction_static` differs because 0.80 does not boot
without DPA (a result, see `../notes/notes.nodpa.md` §5); `chunked_prefill_size`
and `ep_size` are pinned identical on purpose, because the stock leg script would
have moved both.

The `../logs/*_tail6000.log.gz` files cover the **measured window** and do not
contain the boot line — which is why these are captured separately.
