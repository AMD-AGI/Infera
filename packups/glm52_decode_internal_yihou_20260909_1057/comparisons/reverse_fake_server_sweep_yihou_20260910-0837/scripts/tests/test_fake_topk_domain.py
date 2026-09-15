"""Fake PD must keep the physical-slot index domain consistently."""
import ast
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

source=ast.parse(Path(sys.argv.pop(1)).read_text())
fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='should_use_dsa_fused_topk')


class DomainTest(unittest.TestCase):
    def test_consistent_fake_domain_and_unchanged_other_modes(self):
        for mode,backend,seed,enabled,remap in itertools.product(['null','decode','prefill'],['fake','mooncake'],[False,True],[False,True],[False,True]):
            with self.subTest(mode=mode,backend=backend,seed=seed,enabled=enabled,remap=remap):
                env={'get_disagg':lambda:SimpleNamespace(disaggregation_mode=mode,disaggregation_transfer_backend=backend),
                     'envs':SimpleNamespace(SGLANG_DSA_FUSE_TOPK=SimpleNamespace(get=lambda:enabled)),
                     'should_remap_pd_dsa_seed_to_local_slots':lambda:remap}
                exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-domain-guard>','exec'),env)
                expected=enabled and (mode=='null' or not seed or remap or (mode=='decode' and backend=='fake'))
                self.assertEqual(env['should_use_dsa_fused_topk'](seed),expected)


unittest.main()
