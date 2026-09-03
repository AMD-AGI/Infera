import json, sys, importlib.util
class _B:
    def find_module(self, name, path=None):
        if name == "aiter" or name.startswith("aiter."): return self
    def load_module(self, name): raise ImportError("blocked")
sys.meta_path.insert(0, _B())
spec = importlib.util.spec_from_file_location(
    "quark_utils", "/probe/python/sglang/srt/layers/quantization/quark/utils.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
should_ignore_layer = m.should_ignore_layer
cfg = json.load(open("/model/config.json"))
ex = cfg["quantization_config"]["exclude"]
print("cfg packed_modules_mapping:", cfg["quantization_config"].get("packed_modules_mapping"))
for pmm in ({}, {"experts": ["gate_proj","up_proj","down_proj"]}):
    print("=== fused_mapping:", pmm)
    for p in [f"model.layers.{L}.mlp.experts" for L in (3,5,6,45,10)] + \
             ["model.layers.45.mlp.experts.0.down_proj",
              "model.layers.3.mlp.experts.0.down_proj",
              "model.layers.45.mlp.experts.gate_proj"]:
        try: r = should_ignore_layer(p, ignore=ex, fused_mapping=pmm)
        except Exception as e: r = f"EXC {type(e).__name__}: {e}"
        print(f"  {p} -> {r}")
