import importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('placement',Path(__file__).parent/'bench-harness/tools/campaign_topology.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def worker(role,node,ip,gpus,port,name):
    return dict(role=role,node=node,data_ip=ip,gpu_ids=gpus,tp=len(gpus),dp=len(gpus),engine_port=port,bootstrap_port=28000+(port-29000)//256,kv_port=27000+(port-29000)//256,snapshot_port=26000+(port-29000)//256,container=name)

class PlacementTest(unittest.TestCase):
    def setUp(self):
        self.rows=[worker('prefill','p','10.0.0.1',[0,1,2,3],29001,'p0'),worker('prefill','p','10.0.0.1',[4,5,6,7],29257,'p1'),worker('decode','d','10.0.0.2',list(range(8)),29002,'d0')]
    def load(self,rows):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'topology.json';p.write_text(json.dumps(rows));return m.load(p)
    def test_two_p_workers_share_node_without_gpu_or_port_overlap(self):
        rows=self.load(self.rows)
        self.assertEqual([r['instance'] for r in rows],['prefill-0','prefill-1','decode-0'])
        self.assertEqual(rows[1]['gpu_ids'],[4,5,6,7])
        self.assertEqual(rows[1]['url'],'http://10.0.0.1:29257')
    def test_gpu_overlap_is_rejected(self):
        self.rows[1]['gpu_ids']=[3,4,5,6]
        with self.assertRaisesRegex(ValueError,'GPU assignments overlap'):self.load(self.rows)
    def test_internal_port_ranges_are_separate(self):
        self.rows[1]['engine_port']=29003
        with self.assertRaisesRegex(ValueError,'256-port'):self.load(self.rows)
    def test_same_node_requires_same_ip(self):
        self.rows[1]['data_ip']='10.0.0.3'
        with self.assertRaisesRegex(ValueError,'one-to-one'):self.load(self.rows)
    def test_parallelism_matches_physical_gpus(self):
        self.rows[0]['tp']=8
        with self.assertRaisesRegex(ValueError,'TP=DP'):self.load(self.rows)
    def test_control_port_cannot_land_in_another_engine_range(self):
        self.rows[1]['bootstrap_port']=29100
        with self.assertRaisesRegex(ValueError,'engine range'):self.load(self.rows)
    def test_control_port_collision_is_rejected(self):
        self.rows[1]['kv_port']=self.rows[0]['bootstrap_port']
        with self.assertRaisesRegex(ValueError,'ports overlap'):self.load(self.rows)

if __name__=='__main__':unittest.main()
