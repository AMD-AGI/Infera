#!/usr/bin/env python3
"""Plan a ratio preserving the exact page-aligned baseline HiCache capacity."""
import argparse
import json
import math
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--device-tokens',type=int,required=True,help='Actual new device pool capacity, not a predicted value')
p.add_argument('--host-tokens',type=int,default=4715200)
p.add_argument('--page-size',type=int,default=64)
p.add_argument('--bytes-per-token',type=int,default=44928)
a=p.parse_args()
if min(a.device_tokens,a.host_tokens,a.page_size,a.bytes_per_token)<=0 or a.host_tokens%a.page_size:
 p.error('positive capacities and page-aligned host tokens required')
# The image uses floor(size/page)+1, including when already aligned.
raw_target=a.host_tokens-a.page_size/2
ratio=raw_target/a.device_tokens
allocated=(int(a.device_tokens*ratio)//a.page_size+1)*a.page_size
assert allocated==a.host_tokens
size_gb=a.host_tokens*a.bytes_per_token/1e9
options={}
for gb in {math.floor(size_gb),math.ceil(size_gb)}:
 tokens=(int(gb*1e9//a.bytes_per_token)//a.page_size+1)*a.page_size
 options[str(gb)]={'allocated_tokens':tokens,'delta_tokens':tokens-a.host_tokens}
print(json.dumps({'actual_device_tokens':a.device_tokens,'target_host_tokens':a.host_tokens,
 'hicache_ratio':ratio,'verified_allocated_host_tokens':allocated,'host_bytes':a.host_tokens*a.bytes_per_token,
 'host_decimal_gb':size_gb,'integer_hicache_size_alternatives':options,
 'status':'PLAN_ONLY_VERIFY_LIVE_CAPACITY_BEFORE_LOAD'},indent=2))
