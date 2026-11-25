/* Tests instr/hostalloc_log.c against instr/test/fakehip.c (run by run_shim_tests.sh with HOSTALLOC_LOG_DIR set):
 * a 2 MiB hipHostMalloc (logged with a backtrace) and two hipHostRegister calls, which the fake fails (failures are
 * logged whatever their size), then the allocation freed; the script checks the log. */
#include <assert.h>
#include <stdlib.h>
int hipHostMalloc(void **, size_t, unsigned);
int hipHostFree(void *);
int hipHostRegister(void *, size_t, unsigned);
int hipHostUnregister(void *);

int main(void)
{
    void *a = NULL;
    assert(hipHostMalloc(&a, 2 << 20, 0) == 0 && a);
    void *b = malloc(128 << 10), *c = malloc(4096);
    assert(hipHostRegister(b, 128 << 10, 2) == 99); /* the fake fails registration; still logged with err 99 */
    hipHostRegister(c, 4096, 2);
    assert(hipHostFree(a) == 0);
    return 0;
}
