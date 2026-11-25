/* A fake libamdhip64 for testing instr/umbp_vmm_pool.c without a GPU: a "VMM chunk" is a memfd, mapped at the
 * reserved "device" address by hipMemMap and exported as a dup of the memfd, so data written through the CPU address
 * shows up at the device address. hipHostRegister / hipHostGetDevicePointer fail: the shim must answer them itself.
 * test_umbp_vmm_pool.c sets fail_at to make the Nth hipMemCreate fail. */
#define _GNU_SOURCE
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
typedef struct { int type, id; } Loc;
typedef struct { int type, req; Loc loc; void *w; unsigned f; } Prop;
typedef struct { Loc loc; int flags; } Acc;
int fail_at = -1, created = 0, released = 0, last_kind = -1, access_calls = 0, nodes_seen[16];
int hipMemGetAllocationGranularity(size_t *g, const Prop *p, int o) { (void)p; (void)o; *g = 2 << 20; return 0; }
int hipMemAddressReserve(void **a, size_t len, size_t al, void *h, unsigned long long f)
{
    (void)al; (void)h; (void)f;
    *a = mmap(NULL, len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return *a == MAP_FAILED;
}
int hipMemAddressFree(void *a, size_t len) { return munmap(a, len); }
int hipMemCreate(void **h, size_t len, const Prop *p, unsigned long long f)
{
    (void)f;
    if (created == fail_at)
        return 2;
    if (p->type != 1 || p->req != 1 || p->loc.type != 3)
        return 1;
    int fd = memfd_create("chunk", 0);
    if (fd < 0 || ftruncate(fd, (off_t)len))
        return 3;
    if (created < 16)
        nodes_seen[created] = p->loc.id;
    created++;
    *h = (void *)(long)(fd + 1);
    return 0;
}
int hipMemRelease(void *h) { close((int)(long)h - 1); released++; return 0; }
int hipMemMap(void *a, size_t len, size_t off, void *h, unsigned long long f)
{
    (void)off; (void)f;
    return mmap(a, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, (int)(long)h - 1, 0) != a;
}
int hipMemUnmap(void *a, size_t len)
{
    return mmap(a, len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED;
}
int hipMemExportToShareableHandle(void *fd, void *h, int t, unsigned long long f)
{
    (void)t; (void)f;
    *(int *)fd = dup((int)(long)h - 1);
    return 0;
}
int hipMemSetAccess(void *a, size_t len, const Acc *d, size_t n)
{
    (void)a; (void)len; (void)n;
    access_calls++;
    return d->loc.type != 1 || d->flags != 3;
}
int hipGetDeviceCount(int *n) { *n = 2; return 0; }
int hipGetLastError(void) { return 0; }
int hipDeviceGetPCIBusId(char *b, int l, int d) { (void)b; (void)l; (void)d; return 1; }
int hipHostRegister(void *p, size_t l, unsigned f) { (void)p; (void)l; (void)f; return 99; }
int hipHostUnregister(void *p) { (void)p; return 99; }
int hipHostGetDevicePointer(void **d, void *h, unsigned f) { (void)d; (void)h; (void)f; return 98; }
int hipMemcpy(void *d, const void *s, size_t n, int k) { last_kind = k; memcpy(d, s, n); return 0; }
int hipMemcpyAsync(void *d, const void *s, size_t n, int k, void *st) { (void)st; return hipMemcpy(d, s, n, k); }
