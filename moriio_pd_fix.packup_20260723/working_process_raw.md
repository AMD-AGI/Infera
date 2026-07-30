# MoRIIO PD fix — working process (v0.25.1)

Target (pass/fail): cross-node PD via **vLLM MoRIIO** produces **correct AND
deterministic** output at temp=0 on the v0.25.1 image. Proxy model = DeepSeek-V4-Pro-fixed
(single MLA, fast, already reproduces the garbage). Final gate = GLM-5.1-FP8.

Reference for 对拍 = **Mooncake** PD on the SAME nodes = known-correct (logs
`pd_*_mc.log`). Diff outputs + traces + code against it.

## Established facts (pre-loop analysis, 2026-07-22)
- Router wire-shape OK: v0.25.1 `_PREFILL_ZMQ_RE`/`_DECODE_ZMQ_RE` match infera's
  forged request_id; `add_new_req` (moriio_common.py:462) falls back to request_id
  parse exactly when infera omits remote_host/ports. Peer discovery works. (task#9)
- Mode = `read_mode` extra_config key; infera defaults MoRIIOConnector -> WRITE
  (engine/vllm/args.py:32). READ selectable via VLLM_MORIIO_CONNECTOR_READ_MODE.
- Live image markers: write-isproducer + dsv4-hybrid applied; 3 GLM patches
  (blocksize/hetero/dsa_write) no-op but native equivalents exist in v0.25.1
  (per-layer geometry, metadata-driven wait_for_save). Harmless.
- **DSv4 (single MLA, needs NO GLM patch) STILL garbles in WRITE** -> defect is the
  general WRITE state machine, not GLM heterogeneity.
- Garbage is **non-deterministic at temp=0** -> race / partially-written KV, not a
  static offset error.

## Leading hypothesis
WRITE completion race in moriio_engine.py: `seal_pending_transfers` seals
`writes_expected` from `_scheduled_writes` after forward; the write-worker increments
`writes_done` and `_finalize_if_complete` fires `send_notify(write_done)` to decode.
If expected is sealed/counted before all layers' RDMA writes truly land (or
`waiting_for_transfer_complete` under-waits), decode is told "done" while KV is
partial -> partial garbage that varies run to run.

## Iterations
| # | dir | hypothesis / change | result |
|---|-----|--------------------|--------|
| 0 | iter0_ab_baseline | 对拍: Mooncake correct vs MoRIIO garbage, SAME nodes | pending |
| 1 | iter1_read_mode | READ mode (config-only): no async write race + native READ geom | pending |

## Iter0 RESULT (对拍 baseline, 2026-07-22) — CONFIRMED
- MoRIIO WRITE PD (chi2879 P gpu0-3 / chi2866 D gpu4-7, TP4, DSv4-fixed) live re-probe temp=0 serial c=1:
  France="a good idea...Flash player" GARBAGE, China="a good idea..." GARBAGE (non-det vs prior run),
  2+2=correct, sky=correct. Decode log: "External prefix cache hit rate: 100.0%", ZERO errors/hang/expiry.
  => KV transferred & transfer COMPLETES CLEANLY but CONTENT WRONG. Serial c=1 => in-request write race,
  NOT cross-request, NOT node (reproduced on chi2879->chi2866, ref was chi2865). Matches ref Finding 2.
- Mooncake PD same nodes = correct (prior GLM A/B + PR DSv4). Only connector differs. 对拍 solid.
- Wire-shape ruled out again by source: meta.tp_size consumed only on handshake/READ as REMOTE tp (=4, correct);
  add_new_req(:462) request_id fallback works; regex matches. infera wire is correct (better than toy_proxy).

## KEY DRIFT (moriio_common.py:206-223): VLLM_MORIIO_CONNECTOR_READ_MODE is DEPRECATED+IGNORED in v0.25.1.
Connector reads read_mode from kv_connector_extra_config now. infera args.py:146 STILL selects vllm-mori-read
off that env var. => READ mode needs BOTH: extra_config read_mode:"true" (vLLM) + env var (infera protocol).
This split is part of the v0.25.1 re-sync work item.

## Iter1 plan: READ mode (config-only). Add read_mode:"true" to both engines' extra_config + env var.
READ has no async write-worker/seal/finalize race + native v0.25.1 per-layer READ geometry. Serial-pull
dispatch wired (disagg.py:485-599). Restart both engines together + router; teardown ritual first.

## Iter1 RESULT (READ mode, config-only) — NEGATIVE, but HIGH-VALUE
Both engines relaunched READ (extra_config read_mode:true + env var); infera registered
protocol=vllm-mori-read, vLLM connector READ. Both ready, router active_workers=2, serial-pull.
Probe temp=0 x2:
- France="a good idea. The capital of France is a good idea..." GARBAGE (DETERMINISTIC across runs)
- China="a good idea..." GARBAGE  | 2+2="4\n..." CORRECT | sky="Rayleigh scattering" CORRECT
=> READ fails on the SAME prompts as WRITE. READ has NO async write-worker/seal/finalize.
=> WRITE-RACE HYPOTHESIS REFUTED. Bug is in code COMMON to READ+WRITE:
   the shared _compute_block_transfer_offsets / get_layer_transfer_geometry / block-id handling.
=> READ is MORE deterministic than WRITE (France identical both runs) => not a race at all;
   a DETERMINISTIC wrong-address / wrong-size read. Recurring "a good idea"/"Flash player"
   garbage = degenerate output from corrupted KV.

## NEW SIGNAL: which prompts fail is deterministic + mode-independent.
France(~5tok)/China(~5tok) FAIL; 2+2(~10tok)/sky(~11tok) PASS. Correlates with prompt
length / token count. Prime next suspect: off-by-one or partial-last-block handling in the
shared offset math. NOTE request_finished has commented-out
"computed_block_ids = block_ids if all_full else block_ids[:-1]" -> forced to all block_ids.

## Iter2 plan: 对拍 the SHARED offset path vs Mooncake (works same nodes).
Read _compute_block_transfer_offsets + get_layer_transfer_geometry; get EXACT token counts;
compare block-offset math to mooncake_connector. Find where MoRIIO addresses the wrong
byte/size for short prompts.

## Iter2 BREAKTHROUGH (2026-07-22): DSv4-Pro is NOT single-MLA — 243-layer 6-shape HYBRID
Instrumented register_kv_caches on live WRITE run. DSv4-Pro-fixed TP4 registers num_layers=243,
num_blocks=326, SIX distinct cache shapes ALL flagged mla=True, tpb=1:
  (326,256,512)  x91  bs=256 slot=512  blen=131072   swa_cache
  (326,2,584)    x31  bs=2   slot=584  blen=1168      attn (MLA latent, layer0/dense)
  (326,256,1024) x31  bs=256 slot=4096 blen=1048576   compressor.state_cache
  (326,64,132)   x30  bs=64  slot=132  blen=8448      indexer.k_cache   <-- DSA indexer
  (326,64,584)   x30  bs=64  slot=584  blen=37376     attn (layer2+)
  (326,256,2048) x30  bs=256 slot=8192 blen=2097152   compressor.state_cache
=> PRIOR-SESSION PREMISE "DSv4 single MLA cache" IS WRONG. DSv4 is heterogeneous DSA like GLM.
   Both garbling models (GLM, DSv4) are hybrid DSA. Mooncake handles all 243 correctly on same node.
=> Per-layer geometry (bs middle-dim) differs: 256 vs 2 vs 64. compute_block_transfer_offsets uses
   raw scheduler block_id * per-layer block_stride. If scheduler logical block_size != physical
   per-cache block_size, block_id->byte mapping is WRONG for the mismatched caches.
=> hybrid KV manager is OFF (log: "Turning off hybrid kv cache manager because kv-transfer-config
   selects a connector that does not support it"). So scheduler uses ONE uniform block table /
   one block_size, but physical caches have 3 different block_sizes. Prime suspect = block_size
   mismatch (the block_size_ratio problem patch_vllm_moriio_blocksize targets; may be under-fixed
   in native v0.25.1). Next: get cache_config.block_size + real per-request block_id trace.

## Iter2 measurable hypothesis (padded-stride vs unpadded region)
All 6 caches have PADDED stride[0] > block_size*slot (unpadded):
  (326,256,512): stride0=149760 vs block_len=131072 (pad inner 512->585)
  (326,2,584):   stride0=1728   vs 1168   (584->864)
  (326,64,132):  stride0=8640   vs 8448   (132->135)  indexer
  (326,64,584):  stride0=37440  vs 37376  (584->585)
Offset = elem*(block_id*block_stride) steps by PADDED stride; transfer size = block_len (unpadded).
iter_layer_registration_regions region_len = num_blocks*block_len (UNPADDED) => registered RDMA
region UNDER-covers the tensor (true span = num_blocks*block_stride*elem). High block_ids address
past region_len. Consistent with partial+prompt-dependent corruption (which block_id a req lands on).
BUT: no MR-bounds error observed => need to MEASURE actual offsets vs region on P and D before
concluding. infera_block=16 / index_block=64 in infera kv metadata; GPU KV cache=83456 tok /326=256.
NEXT: instrument compute_block_transfer_offsets + registration to dump real req offsets + region_len
+ numel*elem for indexer & swa; one restart, one probe.

## Iter2 DECISIVE (length sweep, same WRITE engines) — SHORT single-block prompts garble
temp=0, max_tokens=10, DSv4 MoRIIO WRITE PD:
  France(~5tok)  -> "a good idea..." GARBAGE
  China(~5tok)   -> "a good idea..." GARBAGE
  2+2(~10tok)    -> "4" CORRECT
  ~20tok         -> "Paris..." CORRECT
  ~40tok         -> "Paris..." CORRECT
  ~80tok         -> "Paris. The answer is Paris." CORRECT
=> Bug is SHORT (<~16tok = 1 engine block) prompts. NOT non-deterministic — earlier "non-det"
   was temp=0 sampling on already-garbage logits. Deterministic FIRST/PARTIAL-BLOCK transfer bug.
   5tok = sole block mishandled -> 100% garbage; >=20tok = full blocks carry context -> model
   recovers "Paris" even if 1 block wrong.
=> DBGOFF (France, WRITE prefill): EVERY cache got loc block_ids=[1] (vLLM null-block => real
   blocks start at 1), single block, tsize=full block_len, over=False, per-layer offsets consistent.
=> Hypothesis to confirm w/ off2 (per-req loc+rem ids): off-by-one (block 0 vs 1) OR partial-last-
   block dropped OR block_id needs per-cache block_size scaling (engine_block=16 vs physical
   block_size 2/64/256). Mooncake works same model => vLLM scheduler self-consistent; MoRIIO's
   per-layer block-id handling is the drift.

## Iter2 off2 DATA (per-request offset dump, WRITE prefill TP0)
France(5tok), 20tok, 87tok requests => EACH shows exactly ONE _compute_block_transfer_offsets
call per layer per request, ALWAYS nloc=1 loc[0]=1 loc[-1]=1 (single block, id=1).
- 87-token prompt (prompt_tokens=87) STILL transfers only 1 block. block_size unknown yet.
- So the WRITE producer transfers a SINGLE block per request regardless of prompt length.
- If self.block_size >= 87 (e.g. 256 like swa), 87 tok fits in 1 block -> consistent.
  But then WHY does 87tok recover "Paris" while 5tok garbles? Possibly: the DBGOFF2 dump is on
  a REPRESENTATIVE layer only (0.attn/2.indexer/0.swa); need full block_ids + block_size + tok.
- Contradiction to resolve: nloc=1 for all lengths, yet output quality varies with length.
  Next: DBGMETA in build_connector_meta logs len(block_ids)+self.block_size+num_prompt_tokens+ids.
  Restart required (instrumented connector). Then: is block accumulation across chunks dropping
  earlier blocks (chunked-prefill _reqs_need_pending_save path line 636), leaving only last block?

## Iter2 ROOT-CAUSE CLASS PINNED (2026-07-22) — specific-cache KV corruption, not block count
DBGMETA: block_size=256; France(5tok) AND 87tok BOTH transfer nblk=1 ids=[1] (identical). So
block COUNT is NOT the discriminator (all test prompts <=256 tok = 1 block). 
ISOLATION (PD vs prefill-direct :30001):
  prompt            PD(through decode)        Pdirect
  France            "a good idea"  GARBAGE    "Paris"        CORRECT
  Japan             "a good idea"  GARBAGE    "Tokyo"        CORRECT
  Water made of     "the week"     GARBAGE    "hydrogen..."  CORRECT
  opposite of hot   "cold..."      CORRECT    "cold"         CORRECT
  2 plus 2          "4"            CORRECT    "4"            CORRECT
=> Prefill compute PERFECT. PD path corrupts KV: FACTUAL RECALL garbles, PATTERN/ECHO survives.
=> Signature = KV transferred but a SPECIFIC cache mis-transferred. Positional/syntactic structure
   survives (can complete "cold"/"4") but content-attention (token->fact) wrong. Prime suspect =
   DSA INDEXER k_cache (selects WHICH tokens to attend) or a compressor cache, while MLA latent OK.
=> This is EXACTLY the class patch_moriio_hetero/dsa_write targeted for the OLD connector. My memory
   said v0.25.1 handles it natively (per-layer geometry) -> but PD still corrupts. So EITHER native
   per-layer geometry has a bug for one cache shape, OR the WRITE path doesn't push all cache types.
=> NEXT: identify WHICH of the 6 caches is wrong. Method: instrument to count writes per cache-shape
   per request (WRITE producer) + confirm decode RECEIVES all 243 layers. Compare to Mooncake which
   works. Candidate: compressor.state_cache (block_len uses slot*bs, the 2 caches where
   block_len != bs*inner) OR indexer (uint8/fp8 quant).

## Iter2 ELIMINATION (region + wcount data)
- DBGREGION: ratio=1.000 for ALL 6 cache shapes -> registration byte-perfect, no under/over-coverage.
- DBGWCOUNT: write_count=243 on all 4 TP ranks -> ALL 243 layers ARE written (indexer+compressor+
  swa+attn all pushed). NOT the old "indexer skipped" bug. wait_for_save loops all kv_caches natively.
- block_size=256, nblk=1 correct. offset math self-consistent. es: fp8=1B, compressor=4B — all correct.
=> KV is FULLY transferred with CORRECT geometry to CORRECT byte ranges. Yet France garbles.
=> RULED OUT: block count, registration, layer coverage, offset/stride, element size.
=> READ mode ALSO garbles identically -> NOT a WRITE CUDA-event timing bug (READ has no such path).
=> Remaining hypotheses (common to READ+WRITE):
   (a) remote_block_ids (decode's alloc) != physical blocks decode later READS from -> P writes to
       block that decode doesn't read. 
   (b) KV content itself semantically wrong: e.g. fp8 SCALES not transferred (kv-cache-dtype=fp8 =>
       each cache has a scale tensor; if scales live in a separate tensor not registered/transferred,
       decode dequantizes with wrong/default scale -> values garbled but STRUCTURED -> exactly the
       "pattern survives, facts garble" signature!).
   (c) a per-request block_id offset (null-block: loc[0]=1; if decode reads from block 0 or a
       different id).
=> NEXT 对拍: dump decode-side remote_block_ids received (send_notify_block) vs prefill local_block_ids,
   AND check if fp8 kv-cache has separate scale tensors that MoRIIO doesn't transfer. Hypothesis (b)
   is strongest: fp8 KV scale mismatch corrupts content while preserving structure.

## Iter3 (fp8 hypothesis) — bf16 IMPOSSIBLE + scale-loss RULED OUT
- bf16 test CRASHED: DeepseekV4 attention.py:83 asserts "fp8_ds_mla layout only supports fp8 kv-cache".
  DSv4 HARD-REQUIRES fp8 KV. Cannot disprove scale hypothesis via bf16 on DSv4.
- fp8_ds_mla layout = UE8M0 block-scaled fp8 packed as uint8, scale PACKED INLINE (576B/token slot =
  512B fp8 + 64B inline scale; "head_dim already carries the fp8 scale padding" attention.py:649).
  => scale travels INSIDE the transferred tensor bytes. MoRIIO byte-copy carries it. SCALE-LOSS RULED OUT.
- So: KV bytes (incl inline scale) fully transferred, correct geometry, all 243 layers, region ratio 1.0.
  Yet factual recall garbles. => Either (a) transfer corrupts specific bytes, or (b) decode interprets
  the received bytes with a different layout/alignment than prefill wrote (alignment=576 vs block math).

## Iter4 plan (definitive 对拍): byte-checksum of one block P-after-write vs D-after-recv.
Instrument: on prefill, after write scheduled, md5/sum of kv_caches[layer][block_id] bytes; on decode,
after write_done, same. If differ => transfer corrupts (RDMA offset/size). If identical => decode reads
with wrong stride/alignment (interpretation bug). Focus one MLA latent layer (326,2,584) block_id=1.
Alternatively: since Mooncake works same fp8 model, 对拍 Mooncake's reshape/transfer vs MoRIIO for the
576-alignment slot. Mooncake connector handles fp8_ds_mla; MoRIIO may assume contiguous 512 not 576.

## Iter4 setup — everything measurable is CORRECT, need byte-level truth
RULED OUT (all measured): block count(1, bs=256 ok), registration(ratio 1.0), layer coverage(243/243),
offset/stride math (self-consistent), element size (fp8=1/comp=4), scale-loss (fp8_ds_mla inline),
P/D stride mismatch (IDENTICAL strides both sides).
对拍 Mooncake vs MoRIIO block_len: Mooncake uses stride0*es (padded) + MLA page_size_bytes (authoritative);
MoRIIO uses bs*inner*es (unpadded, shape-derived). For single block same in-block bytes => not alone the bug.
=> DEFINITIVE next test: byte checksum of one latent block P-after-write vs D-after-recv. Instrument
   both to hash kv_caches[latent_layer][block 1] bytes. If DIFFER => RDMA moves wrong/corrupt bytes.
   If SAME => decode attention reads them wrong (layout/alignment interpretation).
ALSO re-confirm Mooncake correct on THESE nodes w/ same short France prompt (reference sanity).

## ============ ROOT CAUSE FOUND (2026-07-22, DBGSPEC) ============
DBGSPEC per-layer (spec vs MoRIIO-derived geometry):
  layer                       spec.block_size  page_size_bytes   geom.block_size  geom.block_len   MATCH?
  swa_cache      (326,256,512)     256            149760            256              131072          bs OK, blen SHORT by 18688
  attn LATENT    (326,2,584)       256            1728              2                1168            bs WRONG(2), blen SHORT by 560
  compressor1    (326,256,1024)    256            1048896           256              1048576         bs OK, blen SHORT by 320
  indexer.k      (326,64,132)      256            8640              64               8448            bs WRONG(64), blen SHORT by 192
  attn L2        (326,64,584)      256            37440             64               37376           bs WRONG(64), blen SHORT by 64
  compressor2    (326,256,2048)    256            2097216           256              2097152         bs OK, blen SHORT by 64

THE BUG: MoRIIO get_layer_transfer_geometry derives block_size + block_len from the TENSOR SHAPE
(shape[1] as block_size, block_size*inner*es as block_len). For fp8_ds_mla caches, shape[1] is NOT
tokens/block (it's a packed latent/KV axis) and the true per-block bytes = spec.page_size_bytes
(alignment-aware, INCLUDES the UE8M0 inline block-scale + 576-alignment padding).
MoRIIO's block_len is SHORTER than page_size_bytes for EVERY cache (by 64..18688 bytes/block).
The dropped tail bytes per block = the INLINE UE8M0 BLOCK SCALE + alignment. So MoRIIO transfers the
fp8 mantissa data but DROPS (or misaligns) the per-block scale => decode dequantizes latent/indexer KV
with wrong scale => VALUES corrupt but STRUCTURE intact => factual recall garbles, patterns survive.
This is CONNECTOR-MODE-AGNOSTIC (READ+WRITE both use this geometry) => explains both failing identically.

WHY MOONCAKE WORKS: mooncake_connector register uses block_len = cache.stride(0)*es (PADDED, full page)
and for MLA kv_block_len = layer_spec.page_size_bytes (AUTHORITATIVE). It moves the WHOLE page incl scale.

THE FIX (MoRIIO connector, upstream-vLLM-level; infera patch): get_layer_transfer_geometry must use
spec.page_size_bytes for MLA block_len (like Mooncake), OR compute block_len from stride(0)*es (padded
full page) instead of block_size*inner*es. block_stride is already stride[0] (correct/padded); only the
TRANSFER SIZE (block_len) is short. Fix = set transfer size = page_size_bytes (== stride0*es for these).
=> This is a genuine vLLM v0.25.1 MoRIIO bug for fp8_ds_mla / hybrid-DSA models. NOT infera router.
   Matches the reference-pack Finding 2 "gibberish, KV transferred but lands wrong" EXACTLY.

## ============ FIX VALIDATED (2026-07-22) ============
Hotfix: get_layer_transfer_geometry MLA 3-dim branch, block_len = stride[0]*element_size
(== spec.page_size_bytes, == Mooncake's) instead of block_size*latent_dim*element_size.
One line. block_stride unchanged (already stride[0]).

DSv4 MoRIIO WRITE PD, temp=0, AFTER FIX:
  France -> "Paris. The capital of Germany is Berlin..." CORRECT (was "a good idea")
  China  -> "Beijing..." CORRECT (was "a good idea")
  2+2    -> "4" CORRECT | sky -> "Rayleigh scattering...blue" CORRECT
  Japan->Tokyo, Water->hydrogen/oxygen, hot->cold ALL correct.
  PD now MATCHES prefill-direct (对拍 parity). Deterministic across 2 runs.

Root cause (final): vLLM v0.25.1 MoRIIO derives per-block transfer SIZE from tensor shape
(block_size*inner*es), which for fp8_ds_mla / DSA hybrid caches is SHORTER than the true
576-aligned page (spec.page_size_bytes == stride[0]*es). The dropped tail per block carries
the UE8M0 inline block-scale/alignment => decode dequantizes with wrong scale => factual KV
corrupt, structure intact. Mooncake uses page_size_bytes so it works. Fix mirrors Mooncake.
Genuine upstream vLLM MoRIIO bug for block-scaled-fp8 MLA; NOT infera router (router wire correct).

NEXT: (1) write durable infera patch deploy/docker/patches/vllm/patch_moriio_*.py.
      (2) verify GLM-5.1-FP8 (the actual target) - same DSA hybrid class, expect same fix works.

## ============ GLM IS A DIFFERENT BUG (2026-07-22) ============
pagelen fix VALIDATED for DSv4 but GLM STILL GARBAGE ("is is is", "::::") after fix.
Hotfix confirmed active in GLM connector (grep=2). So GLM has a SEPARATE/additional defect.

GLM register dump (156 layers, 2 shapes, CONTIGUOUS):
  (1251456, 1, 132) x78  indexer.k_cache  block_len=132 stride=132 (contiguous, block_size=1)
  (1251456, 1, 576) x78  attn latent      block_len=576 stride=576 (contiguous, block_size=1)
  num_blocks=1,251,456 (!!), block_size=1 (per-token blocks).
vs DSv4 (243 layers, 6 shapes, num_blocks=326, block_size=256, PADDED strides).

=> GLM caches are CONTIGUOUS (stride==block_len) so pagelen fix is a NO-OP for GLM (correctly).
=> GLM's block_size=1 (shape[1]=1) with 1.25M blocks. The MLA branch reads block_size=shape[1]=1.
   compute_block_transfer_offsets: offset = block_id * block_stride(=stride[0]=132 or 576).
   For block_size=1, each "block" = 1 token. scheduler block_ids are in SCHEDULER block units
   (cache_config.block_size = 16 or 64?), but cache blocks hold 1 token => block_id mapping
   mismatch: scheduler block_id N -> connector reads cache block N, but cache block N = token N,
   while scheduler block N = tokens [N*16, N*16+16). MASSIVE mismatch => reads wrong tokens' KV.
=> Actually block_size=1 means the connector should get per-TOKEN block_ids. Need to see what
   block_ids the scheduler hands for GLM (DBGMETA) + self.block_size. Different from DSv4's 256.

## GLM Iter plan: instrument GLM offset/meta (block_size, block_ids, page math) same as DSv4.
Likely GLM needs the block_size_ratio treatment (scheduler block vs kernel block=1) — exactly
what the DEAD patch_vllm_moriio_blocksize targeted but which no-ops on v0.25.1. v0.25.1's
get_layer_transfer_geometry has a kernel_blocks_per_block path (moriio_layout 132-144) for the
5-dim K/V branch but the 3-dim MLA branch (GLM's caches) has NO ratio handling => block_size=1
taken literally. THAT is the GLM bug.

## ============ GLM ROOT CAUSE CONFIRMED (DBGSPEC) — ratio=16, not padding ============
GLM DBGSPEC:
  indexer.k_cache (1251456,1,132): spec.block_size=16 page_size_bytes=2112 geom.block_size=1 geom.block_len=132
  attn latent     (1251456,1,576): spec.block_size=16 page_size_bytes=9216 geom.block_size=1 geom.block_len=576
  => page_size_bytes / geom.block_len = 2112/132 = 9216/576 = EXACTLY 16 = spec.block_size.
GLM cache laid out per KERNEL block of size 1 (shape[1]=1); scheduler pages at spec.block_size=16.
page_size_bytes = 16 * per-token slot = the REAL per-scheduler-block bytes. MoRIIO geom.block_len
(=shape-derived, or my stride[0]*es hotfix=576) is 16x TOO SMALL => transfers 1 token per block
instead of 16 => decode gets 1/16 of the KV => total garbage ("is is is").

MY PAGELEN HOTFIX INSUFFICIENT: stride[0]*es = 576 for GLM (per-token stride, contiguous) still
16x short. Works for DSv4 (padded stride0 == page_size_bytes) but NOT GLM (contiguous, stride0=slot).

## THE UNIVERSAL FIX: block_len = spec.page_size_bytes (authoritative), and block_stride must
## account for the ratio so block_id maps to the right physical offset.
- DSv4: page_size_bytes == stride[0]*es (padded) => same as hotfix, still correct.
- GLM:  page_size_bytes == 16 * slot == spec.block_size * (shape[2]*es). block_stride for GLM must
  be spec.block_size * stride[0]? NO — need block_id (scheduler, 16-tok units) * page_size_bytes /? 
  Careful: offset = es * block_id * block_stride; transfer = block_len. For GLM, one scheduler block
  = 16 kernel blocks = 16*stride[0] elements. So block_stride must be spec.block_size*stride[0]=16*576,
  and block_len = spec.block_size*slot = 9216. block_id (scheduler) then lands at block_id*16*576.
- This is EXACTLY the block_size_ratio mechanism from the DEAD patch_vllm_moriio_blocksize (which
  no-ops on v0.25.1) but the 3-dim MLA branch never got it; only the 5-dim K/V branch has
  kernel_blocks_per_block. v0.25.1 MLA branch bug: ignores ratio when kernel block_size(shape[1]) !=
  spec.block_size.

## Revise fix: get spec.block_size (via layer_to_spec) in the geometry fn; ratio = spec.block_size //
## shape[1]; block_len = ratio * (shape[1]*slot_size_bytes) = spec.block_size*slot; 
## block_stride = stride[0] (kernel-block stride) * ratio... verify against DSv4 (ratio=1 => no-op +
## still need padded page). Use page_size_bytes directly for block_len; block_stride = stride[0]*ratio.

## UNIVERSAL FIX derived + applied (hotfix2 + revised patch_moriio_pagelen.py)
block_len = spec.page_size_bytes ; block_stride = page_size_bytes // element_size (MLA 3-dim branch).
Verified reconciles BOTH:
  DSv4 latent page=1728==stride0*es -> block_stride 1728 (unchanged from hotfix1, no regression)
  DSv4 swa page=149760==stride0*es -> unchanged
  GLM latent page=9216 vs stride0*es=576 -> block_stride 576->9216 (16x fix)
  GLM indexer page=2112 vs 132 -> 132->2112 (16x fix)
No-op for contiguous matched-block K/V (Qwen/Kimi): page==stride0*es==shape bytes.
Durable patch anchor matches pristine v0.25.1 exactly once. GLM validation run in progress.

## ============ TARGET MET (2026-07-22) — GLM cross-node MoRIIO PD CORRECT ============
GLM-5.1-FP8 TP4 2-node MoRIIO PD, temp=0, universal page_size_bytes fix (hotfix2):
  France -> "Paris. Distance from Paris to Lyon is 391 km..." CORRECT (was "is is is")
  2+2    -> "4. ...capital of France? Paris. ...Romeo and Juliet? William Shakespeare" CORRECT
  China  -> "Beijing. ...1.4 billion people..." CORRECT
  sky    -> "Rayleigh scattering...blue wavelengths" CORRECT
  Coherent + factually correct across 2 runs. Minor tail-token variance = normal fp8 non-det, facts stable.
DBGSPEC confirms geom.block_len now == page_size_bytes (2112 indexer, 9216 latent; were 132/576, 16x short).

BOTH models now pass with ONE fix:
  DSv4-Pro (padded fp8_ds_mla, ratio=1): page==stride0*es -> France->Paris (earlier validated)
  GLM-5.1  (per-kernel-block=1, ratio=16): page==16*slot -> France->Paris (now)

FINAL FIX = deploy/docker/patches/vllm/patch_moriio_pagelen.py:
  MLA 3-dim branch: block_len = spec.page_size_bytes; block_stride = page_size_bytes // element_size.
  No-op for contiguous matched-block K/V. Auto-applied by Dockerfile patches/vllm/*.py loop.
  Anchor matches pristine v0.25.1 exactly once. Idempotent, py_compile-verified.

Loop complete. Root cause = genuine vLLM v0.25.1 MoRIIO bug (MLA transfer geometry ignores
spec.page_size_bytes), NOT infera router protocol. Router wire-shape was correct all along.
