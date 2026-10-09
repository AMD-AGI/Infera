# Working process

# Leader status

UTC 2026-09-10T04:59:39.122646+00:00
TaskTP4EP4fakeD,10000/500,c16/c32. Workspaceprepared,exactimageb9a83742f631visibleon249. Userassigned249job130740ends11:52:45Z;056peerprotected. Peercoordinationconfirmed.
Authorizedstop9inspectedtraining/evalcontainersissuedwith300sgrace;notcompleteyet. OtherGPUprobeprocessesinsameGPUruntimeinspectedDstate/wchanflush_workqueue (daysold),all8VRAM90-91%,KFDnone. Firstunhealthynodeobservation~04:59Z;do notinfercauseorclaimkillworked. NoGPUserverlaunched. Informedpeerreplacementmaybeneededexcluding056/249. Noreset/reboot/daemonchanges.
Preparedscriptsthreefakepatches,TP4EP4GPUs0-3,mem0.85,sim3.61,eachgraphmaxequalsc16/c32. PossibleC32verify192rowfallback>FlyDSL96documented. NextCPU+GPUreadinesscheck,thenlaunchwhenidle.
Leadercron2539a084;short-scribependinginitialentry. Nofinalresults.

## 2026-09-10T05:02:05.067006+00:00
249unhealthyconfirmed:all9authorizedstopcommandsfailednoexiteventafter300s,processesD/flush_workqueue. No repeatedkills/reset. Peerowns056only;explicitlycoordinatedreplacementrequest1308916hburstexcluding056,249underuserauthorization. Original249holdnotcancelled,userprovided. Noourserverlaunched.

## 2026-09-10T05:03:39.670313+00:00
Replacement036/job130891allocated05:01:03until11:01:03Z,explicitexcludes056/249andcoordinatedpeer. Fixedimagealreadyresident. GPUhealthprobe8cards,3-5%memoryownedbyZebra130876PIDs113034-113041 mappedDockerTop;authorizedstop300sstarted,collector/exporterpreserved. Original249stuckDstateuntouchedfurther. Noserveryet.

## 2026-09-10T05:09:26.813427+00:00
036heavyworkloadstop exit0,onlycollectorsremain;all8GPUidleprobe passed. c16serverg52-fake-short-tp4ep4started,TP4EP4GPUs0-3,mem0.85,sim3.61,3fakepatches,graphmax16. CopiedexactimageJIT/Triton/TorchcachesintoNEWworkspacewithoutlockfiles;oldcacheunchanged. Readinessmonitorbackgroundupdatesstate/readiness.json20s. Noformalresults yet.

## 2026-09-10T05:23:20.215123+00:00
C16benchmarkPASS128/128,10000/500exact,35.49353s,1803.14548tok/snode451-ish/GPU,P50TPOT8.7754ms,P908.91124ms,accept3.59733.114logsactualrunning16/0retracted. C16containerstoppedexit0andpreservedrenamed. C32launchedsamesettingsTP4EP4mem0.85sim3.61butmaxrunning/graph32;readinesspending. NoDPA,GPUs0-3;no056access.

## 2026-09-10T05:31:32.580713+00:00 — COMPLETE
C16/C32both128success10000/500exact,0errors,actualbatch16/32logged,no retractedqueue. Output1803.145/2569.098tok/s4GPUs,P50TPOT8.775/11.633ms. C32verify192exceedsFlyDSL96gate;fallbackexplicitinlog. Reportsavedresults/REPORT.mdand summary.json.
C32containerstop exit0;all8GPUVRAM0verified05:30:59,onlycollector/exporterremain,KFDPID88564uses0GPUs0bytes(notourwork). Ownreplacement130891cancelledaftercleanup;userprovided249andpeer056untouched. Noadditionalpointsrequested.
