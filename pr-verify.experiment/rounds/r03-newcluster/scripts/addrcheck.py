import ctypes, torch, sys
lib = ctypes.CDLL("libamdhip64.so")
lib.hipHostRegister.restype = ctypes.c_int
lib.hipHostGetDevicePointer.restype = ctypes.c_int
LAYERS, PAGES, ITEM = 4, 64, 512
bufs = []
for i in range(LAYERS):
    t = torch.empty((PAGES, ITEM), dtype=torch.uint8)
    nb = t.numel()*t.element_size()
    rc = lib.hipHostRegister(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(nb), ctypes.c_uint(0x3))
    out = ctypes.c_void_p()
    lib.hipHostGetDevicePointer(ctypes.byref(out), ctypes.c_void_p(t.data_ptr()), ctypes.c_uint(0))
    print(f"layer{i}: hostVA=0x{t.data_ptr():x}..0x{t.data_ptr()+nb:x}  devPtr=0x{out.value:x}  size={nb}")
    bufs.append(t)
