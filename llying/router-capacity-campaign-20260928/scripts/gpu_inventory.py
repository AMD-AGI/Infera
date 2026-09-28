"""Map HIP device ordinals to ROCm-SMI cards through PCI addresses."""
import json,uuid

HIP_PROBE="""import ctypes,json
h=ctypes.CDLL('libamdhip64.so')
assert h.hipInit(0)==0
n=ctypes.c_int();assert h.hipGetDeviceCount(ctypes.byref(n))==0
result={}
for i in range(n.value):
    bus=ctypes.create_string_buffer(64)
    assert h.hipDeviceGetPCIBusId(bus,64,i)==0
    result[i]=bus.value.decode()
print(json.dumps(result))
"""
SMI_PROBE="""import json,subprocess
from pathlib import Path
cards=json.loads(subprocess.check_output(['rocm-smi','--showbus','--showmeminfo','vram','--json'],text=True))
print(json.dumps({'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'cards':cards}))
"""


def match_cards(hip_pci,cards):
    pci_to_card={v['PCI Bus'].lower():k for k,v in cards.items() if k.startswith('card') and 'PCI Bus' in v}
    assert len(pci_to_card)==sum(k.startswith('card') and 'PCI Bus' in v for k,v in cards.items()),'duplicate SMI PCI address'
    mapping={int(gpu):pci_to_card[bus.lower()] for gpu,bus in hip_pci.items()}
    assert len(set(mapping.values()))==len(mapping),'HIP devices alias one card'
    return mapping


def probe_hip(remote,node,image,job):
    name=f'llying-campaign-gpu-map-{job}-{uuid.uuid4().hex[:8]}'
    args=['docker','run','--rm','--name',name,'--label','infera.allocation-job='+str(job),'--device=/dev/kfd','--device=/dev/dri','--group-add','video','-e','HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7',image,'python3','-c',HIP_PROBE]
    try:return json.loads(remote(node,args,timeout=60))
    except BaseException:
        try:remote(node,['docker','rm','-f',name],timeout=30)
        except Exception:pass
        raise


class Inventory:
    def __init__(self):self.cache={}

    def memory(self,remote,node,image,job,snapshot_dir):
        observed=json.loads(remote(node,['python3','-c',SMI_PROBE],timeout=30))
        previous=self.cache.get(node)
        if previous is None or previous['boot_id']!=observed['boot_id']:
            previous={'boot_id':observed['boot_id'],'hip_pci':probe_hip(remote,node,image,job)}
            self.cache[node]=previous
        mapping=match_cards(previous['hip_pci'],observed['cards'])
        assert set(mapping)==set(range(8)),(node,mapping)
        snapshot_dir.mkdir(parents=True,exist_ok=True)
        (snapshot_dir/f'gpu-map-{node}.json').write_text(json.dumps({**previous,'hip_to_smi':mapping},indent=2)+'\n')
        return {gpu:observed['cards'][card] for gpu,card in mapping.items()}
