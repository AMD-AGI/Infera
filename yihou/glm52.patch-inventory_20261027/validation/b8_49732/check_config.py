import argparse
import ast
import importlib.util
import subprocess
import types
import unittest
from pathlib import Path
from unittest import mock

parser = argparse.ArgumentParser(
    description="Run the registered absorb-BMM config class without GPU-heavy imports"
)
parser.add_argument("--checkout", type=Path, required=True)
parser.add_argument("--negative-base")
args = parser.parse_args()
W = args.checkout.resolve()
source = "python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py"
spec = importlib.util.spec_from_file_location(
    "takeover_real_environ", W / "python/sglang/srt/environ.py"
)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
text = (
    (W / source).read_text()
    if args.negative_base is None
    else subprocess.check_output(
        ["git", "show", args.negative_base + ":" + source], cwd=W, text=True
    )
)
tree = ast.parse(text)
selected = [
    n
    for n in tree.body
    if (isinstance(n, ast.FunctionDef) and n.name == "_absorb_bmm_config")
    or (
        isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in ["_ABSORB_BMM_DECODE_MAX_M", "_absorb_bmm_num_cu"]
            for t in n.targets
        )
    )
]
m = types.ModuleType("actual_config")
m.__dict__.update(
    envs=e.envs,
    cdiv=lambda x, y: (x + y - 1) // y,
    Optional=__import__("typing").Optional,
    get_device_core_count=lambda: (_ for _ in ()).throw(
        AssertionError("unexpected device discovery")
    ),
)
exec(compile(ast.Module(body=selected, type_ignores=[]), str(W / source), "exec"), m.__dict__)
tests = ast.parse(
    (W / "test/registered/unit/models/test_rocm_mla_absorb_fusion_gate.py").read_text()
)
klass = next(
    n for n in tests.body if isinstance(n, ast.ClassDef) and n.name == "TestAbsorbBmmConfig"
)
ns = {
    "CustomTestCase": unittest.TestCase,
    "forward_mla_rocm": m,
    "_ABSORB_CONFIG": m._absorb_bmm_config,
    "envs": e.envs,
    "mock": mock,
}
exec(
    compile(
        ast.Module(body=[klass], type_ignores=[]), "actual_registered_config_test_class", "exec"
    ),
    ns,
)


class CountingResult(unittest.TextTestResult):
    subtests = 0

    def addSubTest(self, *args, **kwargs):
        self.subtests += 1
        return super().addSubTest(*args, **kwargs)


r = unittest.TextTestRunner(verbosity=2, resultclass=CountingResult).run(
    unittest.defaultTestLoader.loadTestsFromTestCase(ns["TestAbsorbBmmConfig"])
)
print("subtests", r.subtests)
if args.negative_base is None:
    assert r.wasSuccessful()
else:
    assert not r.wasSuccessful(), "The unpatched negative control must expose the regression"
