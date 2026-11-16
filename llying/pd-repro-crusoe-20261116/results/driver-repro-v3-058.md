| case | driver | headroom before | headroom after GEM pin | headroom after reg | reported free after reg | real free after reg | MRs | reg s | new stream probe | error | last line |
|---|---|---|---|---|---|---|---|---|---|---|---|
| crsuse2-m2m-v3-058-g0-kfd-dmabuf-40 | 6.16.13 | 287.25 |  | 207.25 | 207.38 | 40.45 | 40 | 0.2 | memset ok; memset ok; memset ok; memset ok; hipMalloc 2 MiB ok |  | exit 0 |
| crsuse2-m2m-v3-058-g1-kfd-peermem-40 | 6.16.13 | 287.25 |  |  |  |  |  |  |  | peermem MR ionic_1 off 0 GiB 14 | exit 1 |
| crsuse2-m2m-v3-058-g2-kfd-dmabuf-150 | 6.16.13 | 287.25 |  | 0.0 | 0.0 | 137.7 | 150 | 0.5 |  |  | exit 139 |
| crsuse2-m2m-v3-058-g3-kfd-peermem-150 | 6.16.13 | 287.25 |  |  |  |  |  |  |  | peermem MR ionic_3 off 0 GiB 14 | exit 1 |
| crsuse2-m2m-v3-058-g4-gem-dmabuf-40 | 6.16.13 | 287.25 | 247.25 | 0.5 | 247.23 | 246.81 | 40 | 0.2 | memset ok; memset ok; memset ok; memset ok; hipMalloc 2 MiB ok |  | exit 0 |
| crsuse2-m2m-v3-058-g5-gem-dmabuf-150 | 6.16.13 | 287.25 | 137.25 | 0.5 | 137.23 | 136.81 | 150 | 0.6 | memset ok; memset ok; memset ok; memset ok; hipMalloc 2 MiB ok |  | exit 0 |
| crsuse2-m2m-v3-058-g6-kfd-peermem-40 | 6.16.13 | 287.25 |  |  |  |  |  |  |  | peermem MR ionic_0 off 0 GiB 14 | exit 1 |
| crsuse2-m2m-v3-058-g7-kfd-none-40 | 6.16.13 | 287.25 |  | 247.25 | 247.38 | 0.45 | 0 | 0.0 | memset ok; memset ok; memset ok; memset ok; hipMalloc 2 MiB ok |  | exit 0 |
