"""Exercise the actual configurator guard without importing GPU runtime."""
import ast
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

source = ast.parse(Path(sys.argv.pop(1)).read_text())
fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == '_should_elide_dsa_index_k')


class ElisionTest(unittest.TestCase):
    def test_guard_truth_table(self):
        for mode, backend, draft, hicache, hisparse in itertools.product(
            ['null', 'prefill', 'decode'], ['fake', 'mooncake', 'nixl'], [False,True], [False,True], [False,True]
        ):
            with self.subTest(mode=mode, backend=backend, draft=draft, hicache=hicache, hisparse=hisparse):
                env = {'get_memory':lambda:SimpleNamespace(enable_hierarchical_cache=hicache,enable_hisparse=hisparse),
                       'get_disagg':lambda:SimpleNamespace(disaggregation_mode=mode,disaggregation_transfer_backend=backend)}
                exec(compile(ast.Module(body=[fn],type_ignores=[]), '<actual-guard>', 'exec'),env)
                expected = not draft and not hicache and not hisparse and (mode=='null' or (mode=='decode' and backend=='fake'))
                self.assertEqual(env['_should_elide_dsa_index_k'](is_draft_worker=draft),expected)


unittest.main()
