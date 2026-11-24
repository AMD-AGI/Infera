/* LD_PRELOAD shim for the UMBP standalone server: mlockall(MCL_CURRENT | MCL_FUTURE) at start-up, so its host pool
 * (memfd shmem registered for GPU access, tracked by KFD as userptr) sits on the unevictable LRU. Only useful where
 * vm.compact_unevictable_allowed=0: compaction then never migrates the pool's pages, and KFD never evicts the server's
 * GPU queues to revalidate 418 GiB (EXPERIMENT 4.6, 4.8). Needs RLIMIT_MEMLOCK unlimited (the container has it).
 *   gcc -shared -fPIC -O2 -o mlockall.so mlockall.c
 */
#include <stdio.h>
#include <sys/mman.h>

__attribute__((constructor)) static void lock_all(void)
{
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
        perror("mlockall.so: mlockall");
}
