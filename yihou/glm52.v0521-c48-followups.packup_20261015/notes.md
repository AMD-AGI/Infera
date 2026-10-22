# Notes — what / why / how / context

1. **wts64 was launched twice, and one bench ran.**
   - What: the 07:16 launch was shown to the user as rejected but did execute (cont_benchl PID 1464883). It flushed and started the C64 bench at 07:17:10.
   - The 07:21:56 relaunch then saw the decode `/flush_cache` return 400 sixty times ("pending requests" = the running bench), and it exited.
   - Context: `rounds/wts48/wts48.cont.log` shows both starts; only one `agentx-c64-A2` exists.
   - How to avoid: check `pgrep -af cont_bench` and the bench dir before relaunching after an ambiguous tool result.
2. **wts64 is a 3rd use of one stack.** The order on the stack was a partial A1 (driver killed 04:47), then A2 C48, then C64. It is not a fresh-stack measurement.
3. **Decode flush returns 400 while requests are pending.** `chain_cap2`/`cont_bench` give up after 60 tries. `cont_benchl2` tolerates a failed decode flush; it was written but not used.
4. **tw90 changes one variable.**
   - The token diff vs pf90 is only `--tokenizer-worker-num 4` on decode.
   - `detokenizer_worker_num` stays 1.
   - Prefill is unchanged.
5. **cx90 needle_128k_d10.**
   - The sampled run ended with `finish=length` and empty content. Reasoning used 4101 tokens against `max_tokens` 4096 and was still quoting the haystack.
   - The same test passed in ofs0c/ofg3c/ofg2c (offload rounds, `../glm52.v0521-offloaderv2.packup_20261015`). In ofs0c it answered in 14.9 s.
   - One sampled run each, so the difference is not attributed to anything.
6. **Single replicates.** No run here was repeated, so the run-to-run spread is not measured.
