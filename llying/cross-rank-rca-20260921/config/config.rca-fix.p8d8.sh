#!/usr/bin/env bash
# Fix run: the RCA contract with only the image replaced by fix/Dockerfile's
# build (base fd7220a5 + Mooncake destination-local pinned rail and RC ACK
# timeout, enabled by image ENV).
RCA_IMAGE="infera-sglang:v0519-yihou-0917-nextnfix-hicache-mcdestpin-ibto18"
RCA_IMAGE_ID="sha256:88286215b79d43306dce88dcdf299a13d939c6b708691dc192ca7c7c33c518a5"
source "$(dirname "${BASH_SOURCE[0]}")/config.rca.p8d8.sh"
