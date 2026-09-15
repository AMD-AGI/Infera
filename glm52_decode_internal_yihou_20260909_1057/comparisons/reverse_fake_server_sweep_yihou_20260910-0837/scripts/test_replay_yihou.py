import copy
import importlib.util
from pathlib import Path
import unittest
from typing import Any

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReplayTests(unittest.TestCase):
    def test_off_commands(self):
        m = load('run_point_yihou')
        for c in [4, 8, 16, 20, 24]:
            launch, client, server = m.commands('off', c, 'yihou-test', m.W / 'rounds/test_yihou')
            self.assertNotIn('--rm', launch)
            self.assertEqual(server[server.index('--max-running-requests') + 1], str(c))
            self.assertEqual(server[server.index('--cuda-graph-max-bs') + 1], str(c))
            self.assertEqual(client[client.index('--random-input-len') + 1], '70000')
            self.assertEqual(client[client.index('--random-output-len') + 1], '10000')
            self.assertEqual(client[client.index('--num-prompts') + 1], '128')
            self.assertEqual(client[client.index('--warmup-requests') + 1], '16')
            self.assertIn('--fake-prefill', client)
            self.assertNotIn('--disable-overlap-schedule', server)

    def test_dp_global_admission_local_graph(self):
        m = load('run_point_yihou')
        for c in [4, 8, 16, 20, 24]:
            _, _, server = m.commands('on', c, 'yihou-test', m.W / 'rounds/test_yihou')
            self.assertEqual(server[server.index('--max-running-requests') + 1], str(c))
            self.assertEqual(server[server.index('--cuda-graph-max-bs') + 1], str(c // 4))
            self.assertEqual(server[server.index('--dp-size') + 1], '4')
            self.assertIn('--enable-dp-attention', server)
            self.assertNotIn('--enable-two-batch-overlap', server)

    def fixture(self) -> dict[str, Any]:
        return dict(completed=128,total_input_tokens=8960000,total_output_tokens=1280000,
            random_input_len=70000,random_output_len=10000,max_concurrency=16,
            input_lens=[70000]*128,output_lens=[10000]*128,duration=800,output_throughput=1600,
            server_info=dict(tp_size=4,ep_size=4,dp_size=1,enable_dp_attention=False,
                max_running_requests=16,disaggregation_mode='decode',disaggregation_transfer_backend='fake',
                speculative_algorithm='EAGLE',speculative_num_steps=5,speculative_eagle_topk=1,
                speculative_num_draft_tokens=6,kv_cache_dtype='fp8_e4m3',disable_overlap_schedule=False))

    def test_validate(self):
        m = load('collect_yihou')
        self.assertTrue(m.validate(self.fixture(),'off',16)['topology_verified'])

    def test_reject_wrong_lengths_counts_topology(self):
        m = load('collect_yihou')
        for key,value in [('completed',127),('total_output_tokens',1270000),
                ('input_lens',[70000]*127),('output_lens',[9999]*128),('output_throughput',1700)]:
            d = copy.deepcopy(self.fixture());d[key]=value
            with self.assertRaises(ValueError):m.validate(d,'off',16)
        d = self.fixture();d['server_info']['dp_size']=4
        with self.assertRaises(ValueError):m.validate(d,'off',16)


if __name__ == '__main__':
    unittest.main()
