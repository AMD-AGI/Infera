Thanks for the pointer. I verified that #46424 merged as `8db96fd40202c068137fc56f524b2a53ce9a1026` and is included in the main snapshot used for our current ports (`8a455f1b9005a05091ad1825f0f3493bfd74d27d`). The TileLang/Triton adapter exits now return token-first outputs, so the existing `_attention_on_real_rows` output/LSE handling applies.

I also ran the two existing CPU regression files on that code: **12 passed, 17 subtests passed**, including `test_sparse_prefill_restores_tp_padding` and `test_the_backend_sees_the_real_rows_and_the_tail_is_zero`. This is CPU contract coverage, not a new distributed ROCm accuracy run.

We have retired our overlapping D23 patch from the filing queue and recorded #46424 as its merged upstream resolution.
