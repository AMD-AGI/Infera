# env/

`env_armB_prefill.sh` and `env_armB_decode.sh` are the **exact** env files the two
legs were launched with — `ab_boot.sh` writes them to shared storage and the
container runs them with `docker exec -d`. They are the ground truth for what each
leg was asked to do; `../environment.md` records what the engine actually resolved.

**No `collect_env.sh` output here.** The script is shipped in `../scripts/`, but it
was never run against the live nodes — both allocations were reclaimed at the wall
clock before the snapshot was taken. `../environment.md` reconstructs the hardware
facts from the run's own logs and marks every field it could not recover.
