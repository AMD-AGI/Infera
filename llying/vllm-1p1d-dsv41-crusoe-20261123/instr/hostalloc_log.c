/* LD_PRELOAD logger for the engine processes: which host memory do they pin or register for the GPU?
 *
 * hipHostMalloc / hipHostAlloc / hipMallocHost memory and hipHostRegister'd memory are both KFD userptr (EXPERIMENT 4.11,
 * kfd_migrate_probe.py): when the kernel migrates or write-protects any page of it, KFD evicts every GPU queue of the
 * process until the range is revalidated. After the Engram tables and the UMBP pool left userptr, the engine workers
 * are still evicted for 1-8 s under compaction (R9b, hangE), so something in them still holds userptr memory. This
 * shim changes nothing; it records, per process, in $HOSTALLOC_LOG_DIR/hostalloc-<pid>.log:
 *   every call of at least HOSTALLOC_LOG_MIN bytes (65536): op, pointer, size, flags, live userptr bytes and count;
 *   for calls of at least HOSTALLOC_BT_MIN bytes (1 MiB), a native backtrace (C/C++ frames; Python frames show as the
 *   interpreter);
 *   every 60 s (on the next call) and at exit, the totals: live bytes / count, and calls and bytes by op.
 * Smaller calls are counted but not logged one by one.
 *   gcc -shared -fPIC -O2 -o hostalloc_log.so hostalloc_log.c -ldl -lpthread
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef int hipError_t;
enum { kHipSuccess = 0 };
enum { OP_MALLOC, OP_ALLOC, OP_MALLOCHOST, OP_REGISTER, OP_FREE, OP_UNREGISTER, OP_N };
static const char *op_name[OP_N] = {"hipHostMalloc", "hipHostAlloc", "hipMallocHost", "hipHostRegister", "hipHostFree",
                                    "hipHostUnregister"};

#define LIVE_MAX 65536
static struct { uintptr_t p; size_t n; } live[LIVE_MAX];
static size_t live_bytes, live_count, live_peak;
static unsigned long op_calls[OP_N];
static unsigned long long op_bytes[OP_N];
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *out;
static time_t last_summary;
static __thread int busy; /* our own work (fopen, backtrace) must not recurse into the hooks */

static size_t env_size(const char *name, size_t fallback)
{
    const char *s = getenv(name);
    return s && *s ? strtoull(s, NULL, 10) : fallback;
}

static FILE *log_file(void) /* caller holds lock */
{
    if (!out) {
        const char *dir = getenv("HOSTALLOC_LOG_DIR");
        char path[512];
        snprintf(path, sizeof path, "%s/hostalloc-%d.log", dir && *dir ? dir : "/tmp", (int)getpid());
        out = fopen(path, "a");
        if (out)
            setvbuf(out, NULL, _IOLBF, 0);
    }
    return out;
}

static void stamp(FILE *f)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    gmtime_r(&ts.tv_sec, &tm);
    fprintf(f, "%02d:%02d:%02d.%03ld ", tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000);
}

static void summary(FILE *f, const char *why) /* caller holds lock */
{
    stamp(f);
    fprintf(f, "SUMMARY(%s) live %zu bytes in %zu ranges, peak %zu;", why, live_bytes, live_count, live_peak);
    for (int i = 0; i < OP_N; i++)
        if (op_calls[i])
            fprintf(f, " %s %lu calls %llu bytes;", op_name[i], op_calls[i], op_bytes[i]);
    fputc('\n', f);
}

static void track_add(void *p, size_t n) /* caller holds lock */
{
    for (size_t i = 0; i < LIVE_MAX; i++)
        if (!live[i].p) {
            live[i].p = (uintptr_t)p;
            live[i].n = n;
            break;
        }
    live_bytes += n;
    live_count++;
    if (live_bytes > live_peak)
        live_peak = live_bytes;
}

static size_t track_del(void *p) /* caller holds lock; returns the size, 0 if unknown */
{
    for (size_t i = 0; i < LIVE_MAX; i++)
        if (live[i].p == (uintptr_t)p) {
            size_t n = live[i].n;
            live[i].p = 0;
            live_bytes -= n;
            live_count--;
            return n;
        }
    return 0;
}

static void record(int op, void *p, size_t n, unsigned flags, hipError_t err)
{
    if (busy)
        return;
    busy = 1;
    pthread_mutex_lock(&lock);
    if (err == kHipSuccess) {
        if (op <= OP_REGISTER)
            track_add(p, n);
        else
            n = track_del(p);
    }
    op_calls[op]++;
    op_bytes[op] += n;
    FILE *f = log_file();
    if (f) {
        if (n >= env_size("HOSTALLOC_LOG_MIN", 65536) || err != kHipSuccess) {
            stamp(f);
            fprintf(f, "%s %p %zu flags 0x%x err %d live %zu/%zu\n", op_name[op], p, n, flags, err, live_bytes,
                    live_count);
            if (op <= OP_REGISTER && n >= env_size("HOSTALLOC_BT_MIN", 1 << 20)) {
                void *frames[24];
                int k = backtrace(frames, 24);
                fflush(f);
                backtrace_symbols_fd(frames + 2 < frames + k ? frames + 2 : frames, k > 2 ? k - 2 : k, fileno(f));
                fputs("--\n", f);
            }
        }
        time_t now = time(NULL);
        if (now - last_summary >= 60) {
            last_summary = now;
            summary(f, "periodic");
        }
    }
    pthread_mutex_unlock(&lock);
    busy = 0;
}

static void at_exit(void)
{
    busy = 1;
    pthread_mutex_lock(&lock);
    FILE *f = log_file();
    if (f)
        summary(f, "exit");
    pthread_mutex_unlock(&lock);
}

__attribute__((constructor)) static void init(void)
{
    atexit(at_exit);
}

#define REAL(name, type) \
    static type real; \
    if (!real) \
        real = (type)dlsym(RTLD_NEXT, name)

hipError_t hipHostMalloc(void **p, size_t n, unsigned flags)
{
    REAL("hipHostMalloc", hipError_t(*)(void **, size_t, unsigned));
    hipError_t e = real(p, n, flags);
    record(OP_MALLOC, p ? *p : NULL, n, flags, e);
    return e;
}

hipError_t hipHostAlloc(void **p, size_t n, unsigned flags)
{
    REAL("hipHostAlloc", hipError_t(*)(void **, size_t, unsigned));
    hipError_t e = real(p, n, flags);
    record(OP_ALLOC, p ? *p : NULL, n, flags, e);
    return e;
}

hipError_t hipMallocHost(void **p, size_t n)
{
    REAL("hipMallocHost", hipError_t(*)(void **, size_t));
    hipError_t e = real(p, n);
    record(OP_MALLOCHOST, p ? *p : NULL, n, 0, e);
    return e;
}

hipError_t hipHostRegister(void *p, size_t n, unsigned flags)
{
    REAL("hipHostRegister", hipError_t(*)(void *, size_t, unsigned));
    hipError_t e = real(p, n, flags);
    record(OP_REGISTER, p, n, flags, e);
    return e;
}

hipError_t hipHostFree(void *p)
{
    REAL("hipHostFree", hipError_t(*)(void *));
    hipError_t e = real(p);
    if (p)
        record(OP_FREE, p, 0, 0, e);
    return e;
}

hipError_t hipHostUnregister(void *p)
{
    REAL("hipHostUnregister", hipError_t(*)(void *));
    hipError_t e = real(p);
    record(OP_UNREGISTER, p, 0, 0, e);
    return e;
}
