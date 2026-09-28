"""Explicit per-worker placement for shared-node TP4 and multi-node comparisons."""
import ipaddress,json,re
from pathlib import Path


def load(path):
    source=json.loads(Path(path).read_text())
    if not isinstance(source,list) or not source:
        raise ValueError('placement must be a nonempty worker list')
    result=[];counts={'prefill':0,'decode':0};node_ips={};ip_nodes={};gpus={};ports={};engine_ports={};containers=set()
    for index,row in enumerate(source):
        role=row['role'];node=row['node'];ip=str(ipaddress.IPv4Address(row['data_ip']))
        if role not in counts or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',node):
            raise ValueError('invalid role or node')
        if node_ips.setdefault(node,ip)!=ip or ip_nodes.setdefault(ip,node)!=node:
            raise ValueError('node/IP mapping is not one-to-one')
        ids=row['gpu_ids'];tp=row['tp'];dp=row['dp']
        if not isinstance(ids,list) or any(type(x)!=int or not 0<=x<8 for x in ids) or len(set(ids))!=len(ids):
            raise ValueError('invalid GPU list')
        if type(tp)!=int or type(dp)!=int or tp!=len(ids) or dp!=tp:
            raise ValueError('campaign workers require TP=DP=GPU count')
        used=gpus.setdefault(node,set())
        if used.intersection(ids):raise ValueError('GPU assignments overlap')
        used.update(ids)
        assigned=[]
        for key in ['engine_port','bootstrap_port','kv_port','snapshot_port']:
            value=row[key]
            if type(value)!=int or not 1024<=value<=65535:raise ValueError('invalid port')
            assigned.append(value)
        used_ports=ports.setdefault(node,set())
        if len(set(assigned))!=len(assigned) or used_ports.intersection(assigned):
            raise ValueError('worker ports overlap')
        used_ports.update(assigned)
        engine=row['engine_port'];existing=engine_ports.setdefault(node,[])
        if any(abs(engine-p)<256 for p in existing):raise ValueError('same-node engine ports need separate 256-port ranges')
        existing.append(engine)
        container=row['container']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',container) or (node,container) in containers:
            raise ValueError('invalid or duplicate container')
        containers.add((node,container))
        instance=f'{role}-{counts[role]}';counts[role]+=1
        result.append(dict(row,index=str(index),instance=instance,ip=ip,url=f'http://{ip}:{engine}'))
    for a in result:
        if a['engine_port']+255>65535:raise ValueError('engine port range exceeds TCP limit')
        for b in result:
            if a is b or a['node']!=b['node']:continue
            for key in ['engine_port','bootstrap_port','kv_port','snapshot_port']:
                if a['engine_port']<=b[key]<=a['engine_port']+255:
                    raise ValueError('worker control port overlaps another engine range')
    if not all(counts.values()):raise ValueError('placement needs P and D')
    return result
