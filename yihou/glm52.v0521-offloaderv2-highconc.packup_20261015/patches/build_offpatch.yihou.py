import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
old = '''    def create_device_tensor(self):
        return self._param.to("cuda", non_blocking=True)


class _ShmCpuParamOffloader'''
assert s.count(old) == 1
new = '''    def create_device_tensor(self):
        # YIHOU-OFFLOAD-ATTRS (task v0521cap2): the .to() copy is a plain tensor and drops Python attributes that kernels
        # read off the weight; aiter fused_moe selects its preshuffled MXFP4 kernels by getattr(w, "is_shuffled") and quark
        # apply_weights re-tags only if hasattr(layer.w13_weight, "is_shuffled"). Carry the flag over from the host param.
        t = self._param.to("cuda", non_blocking=True)
        if getattr(self._param, "is_shuffled", False):
            t.is_shuffled = True
        return t


class _ShmCpuParamOffloader'''
s = s.replace(old, new)
compile(s, dst, "exec")
open(dst, "w").write(s)
print("ok")
