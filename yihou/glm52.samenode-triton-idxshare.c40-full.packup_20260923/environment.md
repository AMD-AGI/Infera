# Environment record — crsuse2-m2m-137, infera-sglang:v0519-yihou-0917-nextnfix-hicache, 2026-09-23

This run used **one physical node only**: `crsuse2-m2m-137` (`10.245.153.247`). Both legs of the 1P1D P4D4 deployment ran on that same host (`prefill-0`, `decode-0`, plus `router` and `etcd`), so the same-host GPU-IPC / co-location behavior is part of the environment, not a separate topology.

## Host summary

- Hostname: `crsuse2-m2m-137`
- Date captured: `2026-09-23 10:07:43 +0000`
- Kernel: `6.8.0-107-generic`
- OS: `Ubuntu 24.04.4 LTS`
- Uptime at capture: `27 days, 4:35`
- Load average at capture: `9.65 / 10.17 / 12.70`

## Hardware

### CPU / memory

- CPU model: `AMD EPYC 9575F 64-Core Processor`
- Socket count: `2`
- Cores per socket: `59`
- Threads per core: `2`
- Logical CPUs: `236`
- Total RAM: `2.7 TiB`
- NUMA nodes:
  - `node0`: CPUs `0-117`
  - `node1`: CPUs `118-235`

### GPUs

- GPU model: `AMD Instinct MI355X`
- GPU count: `8`
- ROCm GPU product names reported by `rocm-smi`: MI355X on `GPU[0]` through `GPU[7]`
- Driver version reported by `rocm-smi`: `6.14.14`
- Per-GPU VRAM total: `309220868096` bytes each (~`288 GiB`)
- Per-GPU VRAM used at capture time:
  - GPU0 `286716346368`
  - GPU1 `284141031424`
  - GPU2 `281699807232`
  - GPU3 `281576157184`
  - GPU4 `271260291072`
  - GPU5 `271214141440`
  - GPU6 `271679676416`
  - GPU7 `270891151360`

### NUMA / rail layout

This node exposes two NUMA domains and eight `ionic_*` RDMA rails, plus `mlx5_0` on `ens3`.

- GPUs `0-3` are on the `NUMA0` side and pair with `ionic_0`-`ionic_3`
- GPUs `4-7` are on the `NUMA1` side and pair with `ionic_4`-`ionic_7`
- `mlx5_0` is on `ens3` / `NUMA0`

No inactive rail was observed in the capture below, and no all-zero GID was observed.

## Drivers / ROCm / RDMA packages

- `uname -r`: `6.8.0-107-generic`
- `/sys/module/amdgpu/version`: `6.14.14`
- `amdgpu-dkms`: `1:6.14.14.30100100-2212064.24.04`
- `rocm-core`: `7.0.1.70001-42~24.04`
- `hsa-rocr`: `1.18.0.70001-42~24.04`
- `libionic1`: `54.0-149.g3304be71`
- `rdma-core`: `2410mlnx54-1.2410068`
- `ibverbs-providers`: `2410mlnx54-1.2410068`

## RDMA fabric

All rails below were reported `ACTIVE` / `LinkUp` at capture time.

| Device | Netdev | Port state | Rate | NUMA node | PCI BDF | GID index 1 |
|---|---|---|---|---|---|---|
| `mlx5_0` | `ens3` | `ACTIVE` | `200` | `0` | `0000:00:03.0` | `fe80:0000:0000:0000:8c9e:e2ff:feee:2387` |
| `ionic_0` | `enP2p0s9` | `ACTIVE` | `400` | `0` | `0002:00:09.0` | `fc01:0800:880d:2d5f:0690:81ff:fe44:4d69` |
| `ionic_1` | `enP2p0s10` | `ACTIVE` | `400` | `0` | `0002:00:0a.0` | `fc01:0700:870d:2d5f:0690:81ff:fe44:9f41` |
| `ionic_2` | `enP2p0s11` | `ACTIVE` | `400` | `0` | `0002:00:0b.0` | `fc01:0500:850d:2d5f:0690:81ff:fe43:c489` |
| `ionic_3` | `enP2p0s12` | `ACTIVE` | `400` | `0` | `0002:00:0c.0` | `fc01:0600:860d:2d5f:0690:81ff:fe44:d781` |
| `ionic_4` | `enP3p0s9` | `ACTIVE` | `400` | `1` | `0003:00:09.0` | `fc01:0400:840d:2d5f:0690:81ff:fe42:a771` |
| `ionic_5` | `enP3p0s10` | `ACTIVE` | `400` | `1` | `0003:00:0a.0` | `fc01:0300:830d:2d5f:0690:81ff:fe44:3719` |
| `ionic_6` | `enP3p0s11` | `ACTIVE` | `400` | `1` | `0003:00:0b.0` | `fc01:0100:810d:2d5f:0690:81ff:fe42:8731` |
| `ionic_7` | `enP3p0s12` | `ACTIVE` | `400` | `1` | `0003:00:0c.0` | `fc01:0200:820d:2d5f:0690:81ff:fe44:37c1` |

## Docker / image

- Docker server version on the node: `29.6.1`
- Image tag used: `infera-sglang:v0519-yihou-0917-nextnfix-hicache`
- Image ID: `sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35`
- RootFS DiffIDs:
  - `sha256:fbb9bbbaf4d2b027acd15252897d5043386eea7121e0e0433e697714bb14beac`
  - `sha256:5c4eabd140879cda77a89bf72a4dc6840a8059f2e450561604414557ab79e6f0`
  - `sha256:dd3085d4f3723f12e6e6efdbdd79d7803eefea14eca480b9e418be697c7b05c1`
  - `sha256:37de30e4cda523db57500931680eb5b90f0c5ff57d60601edaf4f6873744c5a8`
  - `sha256:8ca139a99936112756f816903d53e768231d7b3964843a65ceeedd5642717743`
  - `sha256:4b9ef101929866b45cec239df6580dd8ceb24adeb32453a4b9a4bc77fab0b481`
  - `sha256:76d0138cccf7f0671e2dd7959d0fbc6dae2e7b96d7a4dbb13d770e5238439d72`
  - `sha256:e1e3e472a1f785717338e9f0b7066adcc6b1799c11bb3e79d4f521b1b70f1e7c`
  - `sha256:fadb8107069714d1d1293d27933ca92f213e5c124b083c360a3bccbbb9a3cc82`
  - `sha256:a71eb0c36a83652317386dd2d1ee3ee902efd4410e89e1bf9aaf043efecf8a5d`
  - `sha256:f3c25a2a1b965487468593e634c75e8eb6780a171ae5862776d8f7e40486a224`
  - `sha256:97d17d2dbfca23b55bf3708c59fe3fff2f4e1d1d8e7c45763a39702ff9887bf7`
  - `sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef`
  - `sha256:d0059d506e9423a1e40b5974a8c0d13458e0312685bec7e84bb709f841ad0aaa`
  - `sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef`
  - `sha256:b1970ca5b184c5e93b5efbfda6759bbb8e7a6cc50c5a77598405bb9c6e2bda8d`
  - `sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef`
  - `sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef`
  - `sha256:7283aa2897fdd595382a7bb5db410add80acb23fdad218036375e0a5d5624cb5`
  - `sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef`
  - `sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef`
  - `sha256:8d9edb6313bbe67252c44e7e71edc0220cbc2eeffc269d51422cd5340c6840ad`
  - `sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef`
  - `sha256:7cdce616b6c9dfd0ced5a25fb423aa312deee2b1905f5e17ea619fc58e13d163`
  - `sha256:4876eda9c9ae24583d7f5b132243de7215175001035ab02c39b02437287f90ac`
  - `sha256:ce1e5eda04fb77cb7bbf6f1b955c5f26a53711995fdd2e8ee036c5649a0d1abb`
  - `sha256:4caa294e7b818a7c1a6d8f5d8fa1c2e9b91a5d0786686ba7cc1a7b6fe1d48f8b`
  - `sha256:de56ef4f3e9cbe20a402a5e47b24e759b7c589dc9b404b70d045bb08edc82cb4`
  - `sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef`
  - `sha256:a7c1a38301c21ef74954616e6d2012db03e1b1eb8bc7f5b6beddb0086edeecb7`
  - `sha256:f4a4e0b1af24b947cd81ee2f570c6049399e3df834359c07d8226ad3f36f517f`
  - `sha256:403c79afca686fd42ffe855eddc480e268694ed4969b8c2b78a49115fe4e97e7`
  - `sha256:e5d5f3f612f83212e9302a90a903dfff709411275cf5be52ba4bb35aacf90f6a`
  - `sha256:54cf52b4fb996d698e704e1afd9f3664819612f7569429a28d9fa3cf78a17319`
  - `sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef`
  - `sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef`
  - `sha256:2240d35609658cd7c187ca0e31aa71956a9b96d81a79ae40092079f52bbdeab6`
  - `sha256:43f889e343a6762ba445798f3c6ff688c2579c9b2aa1043c90a6fceb8e64fcc5`
  - `sha256:7c1854b714e79697db8e4c5a959a3004428ace5baf648815c5532004c288f85a`
  - `sha256:c7ff63a54dc9a374799da8cb8fb059a1e8c70afafa2bab2cbc32257bf08959f1`
  - `sha256:18db63cb07ecc06bfe5966499c7fab656c4e9961ebbe45a16057b523263a51c6`
  - `sha256:5a4b1f288c40a99d9d555063fa5a0f94584d6ec07845ae25f922fbe625f9138d`
  - `sha256:1a7f1af5201e33cc0b72d7c2ddcca825e5f4db7b51fc2d97de72371c9477074e`
  - `sha256:f5d16f0f7bb05d6ad2d189a3b7f817144cb648c3b2e9135933b4f8403a189c68`
  - `sha256:985135dc9e9151886d944932d440196ca2396ba688c3ec337aa47c5987803d28`
  - `sha256:7585a8aa97beb4eb165a4f589ab8f38c8d03759297894d1a1b74e592b1673544`
  - `sha256:9ec6acd06266547ecdff6c3fbb055905c9d110e0c37483aa8e39da1c54615342`
  - `sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef`
  - `sha256:b706a989387521439cb8b21224e190c771db3905a918e17dee86e3980e0bdd23`
  - `sha256:97d0901f9251b99971225df21da720378f596f320efde150214a5cb4d61d2788`
  - `sha256:626c1377584b3cbf6fe6a4567a1bbfd4c6f4717b3bd9a2c76bb6862b46745f8a`
  - `sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef`
  - `sha256:847721e8c9cb3f0b72beb0ef56e826c04ad83fe77961f98353759cd63c1cc8e7`
  - `sha256:5143d50f85aededac99de96df08ebb4cd43f58a46493dd06df6a8c83cb7e669a`
  - `sha256:adb8c7d1b94ff263bf2ccfad724f6c386d395f74d79e9c2dd5067f1f7f203f8e`
  - `sha256:0d0aabe962fbec4b0d1a96ac28fb689f28a46356bc20b6088fd406d80c4838c1`
  - `sha256:3d10a2acbeeb23a6a40bc74c17ee46a31b95ab7acfed52417bfa9c805dcaa54a`
  - `sha256:41450e97e3a96f5ac45e722b732a30e7c8b3a8c61204a3d5fe35bc65b4d1e590`
  - `sha256:c5c86458b0430a94d06769444de8880f9492159f9ac8fd07173de57832f793f6`
  - `sha256:cc10325ce755dc0c48cc9867518b47b112209fbb7564001f32e1cdb6a1b45509`
  - `sha256:b09c509c98a405d5fdeb71925bffcd8a2de4858258e46b1f90a36d984be823ae`
  - `sha256:5c38d69b417bcdfb162fc2712776b7f36d0b9c791b50eb7957068b3a9c4bbc9e`
  - `sha256:ef29f079e18bb02f0dffa77fbf1558ab93a135511db115ad9bf5950696e0dfeb`
  - `sha256:db205d38286dcb743d8ce9708827a5bc04f44b4a818c649fb9087b05dc6d17d2`
  - `sha256:69b5c6aabc82782db6d5a5861026acb2c30f54321673f5bb6421ca8dca97cbeb`
  - `sha256:345e02d9798df90f3b28b906ff81529bc653bee79eb888f8c6870d276b168f53`
  - `sha256:c2c5582919e3569a75e5248a51c3444799d03194d068e341fdf93659be4cce21`
  - `sha256:58f7b5d0b4a68032d83364f88509c42521e28bbf5917b04735b72efa848cac47`
  - `sha256:72bf032050de089f1aa2951f445e5ad8b5be99557058863b048722503b9df295`
  - `sha256:a9bff293ee6844bd30d9c3a0caab671dab4b1518f22cd8410b7da6c1ec291a27`
  - `sha256:ec7da4ec7f98c532bc542feac7101a804be6c6f1512f7f8b0f179c715bd7e1e9`
  - `sha256:6aeab0ce21820170af06c77495940a027eac7467d8e0871b4f722f73cb046da0`
- Base image: not captured from image metadata; `Config.Labels` only exposes `ubuntu:22.04` as the OS base hint

## Live container environment

Observed inside the live `glm52-pd-yihou-sn-p4d4-decode-0` container:

- `HOME`: `/root`
- `TRITON_CACHE_DIR`: unset
- `sglang.__version__`: `0.5.19.dev20260917+ga9fb1c3238`
- Triton cache path: `/root/.cache/sglang/triton`
- Triton cache subdirs present: many hashed subdirectories under `/root/.cache/sglang/triton/`
- Compiled `.hsaco` kernels found: `181`

This is the evidence that the Triton cache is container-local and cold-compiled on bring-up.

## Repo state

Repository used for this run: `/home/yihou/dev/git/infera.glm52.pd`

- Branch: `dev/pd_opt/glm_5.2_agentx`
- HEAD: `2e5a2d89f839e69729c56a6cb2786050136d45cb`
- Dirty tree: `yes`

## Storage

- `/mnt/m2m_nobackup`: `28T` total, `20T` used, `8.6T` available
- `/shared_nfs`: `410T` total, `408T` used, `2.7T` available
- Model path exists: `yes`
- Model path: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`
- Model size: `408G`

## Notes

- Both legs were on this one node, which is the key reproduction condition for this experiment.
- No environment value was invented here; anything not directly captured was marked as not captured in the underlying collection step.
