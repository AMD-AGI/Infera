# Fake-prefill D: TP4/EP4, ISL10000 / OSL500, c16 vs c32

## Result

Measured2026-09-10 on crsuse2-m2m-036, job130891, GPUs0-3 (4MI355X). User originally assigned249; it was unusable withD-stateflush_workqueueprocesses and failedstoprequests. Replacementexcluded056/249andwascoordinatedwithpeer;056untouched.

Both points use TP4EP4DP1, fake disaggregation decode,FP8KV,EAGLE5steps/6draft/topk1,simulatedaccept3.61(match-expected,real-draft-token),mem_fraction_static0.85,HiCacheoff,3previousfake-onlypatches. Maxrunning/graphmaxmatchpointconcurrency.16warmups+128measuredrequests each, exact10000input/500actualoutput,temperature0,ignoreEOStrue. No fakeinputthroughputisreportedasservedprefill.

| Metric | c16 | c32 |
|---|---:|---:|
| Measured successful requests |128/128|128/128|
| Errors |0|0|
| Duration(s) |35.493531|24.911468|
| Outputtokens |64000|64000|
| Outputtok/s,4GPUinstance |1803.145482|2569.097920|
| Outputtok/s/GPU |450.786370|642.274480|
| P50TPOT(ms) |8.775409|11.632978|
| P90TPOT(ms) |8.911237|13.871815|
| P50ITL(ms) |7.862275|10.387635|
| P90ITL(ms) |10.475096|13.830968|
| Measuredacceptlength |3.597335|3.618010|
| Clientlatency-integratedconcurrency |15.971752|31.905417|

c32vsc16:outputthroughput+42.48%;P50TPOT+32.56%;P90TPOT+55.67%. These are same-test short-run observations,notrepeat-run significanceorlong-durationstabilityclaims.

## Actual occupancy and backend changes

c16:114schedulerlogsallrunning16;allretractedqueue0. c32:56logsrunning32,1running16(warmup/runcontext);allretractedqueue0. Thesearesamplediterationcounts,nottimeweightedoccupancy. RequestlengtharraysandemptyerrorarraysverifiedfromrawJSON.

C32 graph capture explicitly logs `FlyDSL sparse MLA decode declined: seq 192, need 1..96` atbatch32,withsimilar180/168/etcdeclinesforotherlargeverifyshapes. The192=32*6target-verifyshapeexceedsoriginalkernelgate;fallbackremainsenabledasimplementedbythepinnedbackend. Draftdecode32andlowerrowshapescanstilluseFlyDSL. Do notsayc32entirelyFlyDSLorentirelyfallback. C16decodeengagedon4ranks. Exactper-callperformanceattributionnotprofiled.

EP4MoEshapeslack some tunedrows anduseheuristicFlyDSLfallback;Torchmissing`set_signal_pad_size`disabledmultimemall-gather. Theseareloggedimplementationchoices,notevidenceofcorrectnessfailure. Configurationwasnotchangedtosuppresswarnings.

## Scope / interpretation

SyntheticKVdoesnotencodevalidpromptcontext,andacceptancesimulationcommitsunverifieddrafttokens. ThisexperimentmeasuresDexecutionundercontrolledsyntheticconditions,notsemanticcorrectnessorproductionPDthroughput. Do notcompareoutputtok/sdirectlywitholdTP8EP1c16ISL289652/OSL774asatopologyonlyspeedup;bothworkloadandtopologychanged.

Onlyone35.5s/24.9smeasuredrunperpoint. ClientintegerIDsandrange_ratio1ensureexactlengths;retokenizedtextcountnotauthoritative. TPOTandITLareseparatestatistics. Nativeclientpeakconcurrencyusesintegersecondbucketsandmayexceedinstantaneousoccupancy;useschedulerlogsforactualbatch.

## Reproduction

Workspace W=/shared_nfs/yihou/playground/glm52_fake_tp4ep4_10k500_20260910-0452. CommandsrunthroughaCURRENTauthorizedSpurallocation:

```
bash $W/scripts/launch.sh sim c16 0.85 16
python3 $W/scripts/wait_ready.py $JOB c16
bash $W/scripts/bench.sh c16 128 16
# Preserve logs, gracefully stop, verify exit, rename stopped container first.
bash $W/scripts/launch.sh sim c32 0.85 32
python3 $W/scripts/wait_ready.py $JOB c32
bash $W/scripts/bench.sh c32 128 32
```

Launch/benchcommandsrunonhostDockercontextthroughspur exec;readinessmonitorrunsonlogin. Rerunsmustusenewroundnamesandworkspacecopytopreserveevidence. Scriptskeepport31816anduniquecontainerg52-fake-short-tp4ep4. ImageIDb9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d;SGLang402df1e1e453e1e85ec0f5ac4052d36598cc691a,AITER2c71811b32c8ce2e1266aedaec199df7d90f597d,threebind-mountedfake-onlypatchesinsource/. PriorimageJIT/Triton/Torchcacheswerecopiedintothisworkspacewithoutlockfiles;oldcacheunchanged.

RawJSON/logs/commands/containerinspect/readinessundereachround. results/summary.jsonmachine-readable. logs/final-environment.txtcapturedruntime. Noengine-sourcechangesbeyondcopiedprovenfakepatches;nouserrepochanges,nocommits.
