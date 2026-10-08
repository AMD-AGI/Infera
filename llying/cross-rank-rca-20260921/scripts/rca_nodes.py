"""Node pair and image used by the RCA harness.

The default is the current 137->136 contract (the 2026-09-21 reproduction ran
from 138). RCA_PREFILL_NODE / RCA_DECODE_NODE select another pair; IPs are
derived from KNOWN_IPS unless given explicitly. RCA_IMAGE / RCA_IMAGE_ID select
the fix image.
"""

from __future__ import annotations

import os


KNOWN_IPS = {
    "crsuse2-m2m-136": "10.245.154.168",
    "crsuse2-m2m-137": "10.245.153.247",
    "crsuse2-m2m-138": "10.245.157.237",
}
BASE_IMAGE = "infera-sglang:v0519-yihou-0917-nextnfix-hicache"
BASE_IMAGE_ID = (
    "sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35"
)


def _node(role_env: str, default: str) -> tuple[str, str]:
    node = os.environ.get(role_env, default)
    ip = os.environ.get(f"{role_env}_IP") or KNOWN_IPS.get(node)
    if not ip:
        raise SystemExit(f"{role_env}={node} has no known IP; set {role_env}_IP")
    return node, ip


SOURCE_NODE, SOURCE_IP = _node("RCA_PREFILL_NODE", "crsuse2-m2m-137")
DESTINATION_NODE, DESTINATION_IP = _node("RCA_DECODE_NODE", "crsuse2-m2m-136")
NODES = (SOURCE_NODE, DESTINATION_NODE)
IMAGE = os.environ.get("RCA_IMAGE", BASE_IMAGE)
IMAGE_ID = os.environ.get("RCA_IMAGE_ID", BASE_IMAGE_ID)
