/* Tests instr/umbp_vmm_pool.c against instr/test/fakehip.c (run by instr/test/run_shim_tests.sh, preloading the shim
 * with UMBP_VMM_MIN_GB=1). No argument: the takeover path; "fail": the second hipMemCreate fails. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

extern int fail_at, created, released, last_kind, access_calls;
int hipHostRegister(void *, size_t, unsigned);
int hipHostUnregister(void *);
int hipHostGetDevicePointer(void **, void *, unsigned);
int hipMemcpy(void *, const void *, size_t, int);

int main(int argc, char **argv)
{
    (void)argv;
    size_t len = 6UL << 30;
    if (argc > 1)
        fail_at = 1;
    char *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(p != MAP_FAILED);
    if (argc > 1) { /* fell back to an anonymous mapping; what was created is released */
        assert(created == 1 && released == 1);
        p[len - 1] = 7;
        assert(hipHostRegister(p, len, 3) == 99); /* not the pool: reaches the real (fake) HIP */
        printf("fallback ok: created %d released %d\n", created, released);
        return 0;
    }
    assert(created == 3 && access_calls == 2);
    assert(hipHostRegister(p, len, 3) == 0 && hipHostUnregister(p) == 0);
    void *dev = NULL;
    assert(hipHostGetDevicePointer(&dev, p, 0) == 0 && dev && dev != p);
    strcpy(p + (4UL << 30) + 5, "hello"); /* CPU write into the third chunk */
    assert(strcmp((char *)dev + (4UL << 30) + 5, "hello") == 0); /* the same pages at the device address */
    char buf[8] = {0};
    assert(hipMemcpy(buf, p + (4UL << 30) + 5, 6, 2) == 0 && last_kind == 3 && !strcmp(buf, "hello"));
    assert(hipMemcpy(p + 100, "abc", 4, 1) == 0 && last_kind == 3 && !strcmp(p + 100, "abc"));
    assert(hipMemcpy(buf, p + 100, 4, 0) == 0 && last_kind == 0); /* host-to-host left alone */
    assert(madvise(p, len, MADV_DONTNEED) == 0 && !strcmp(p + 100, "abc")); /* a no-op on the pool */
    char *small = mmap(NULL, 1 << 20, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(small != MAP_FAILED && created == 3); /* below the threshold: untouched */
    assert(munmap(p, len) == 0 && released == 3);
    printf("takeover ok: device %p, CPU %p, created %d released %d\n", dev, (void *)p, created, released);
    return 0;
}
