// t_local.yihou.c -- dlopen a HIP-using library RTLD_LOCAL (like a python extension) to test symbol resolution.
#include <dlfcn.h>
#include <stdio.h>
int main(void) {
    void* h = dlopen("./libfakemc.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }
    int (*run)(void) = (int (*)(void))dlsym(h, "fakemc_run");
    return run();
}
