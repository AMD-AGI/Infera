// DIAG (AUS): LD_PRELOAD shim for the decode worker's RDMA registration ENOMEM. Logs every failed madvise (libibverbs'
// MADV_DONTFORK on each ibv_reg_mr range after ibv_fork_init) with the process's map count and the /proc/self/maps
// lines covering the range, and every ioctl/write that fails with ENOMEM with the file it targets.
// Output: $REGDIAG_DIR/regdiag-<pid>.log
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

static int (*real_madvise)(void *, size_t, int);
static int (*real_ioctl)(int, unsigned long, ...);
static ssize_t (*real_write)(int, const void *, size_t);

static FILE *log_file(void) {
    static FILE *f;
    if (!f) {
        char path[512];
        const char *dir = getenv("REGDIAG_DIR");
        snprintf(path, sizeof path, "%s/regdiag-%d.log", dir ? dir : "/tmp", getpid());
        f = fopen(path, "a");
        if (f) setvbuf(f, NULL, _IOLBF, 0);
    }
    return f;
}

static double now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

static int map_count(void) {
    FILE *m = fopen("/proc/self/maps", "r");
    char line[512];
    int n = 0;
    if (!m) return -1;
    while (fgets(line, sizeof line, m)) n++;
    fclose(m);
    return n;
}

static void maps_covering(FILE *f, unsigned long start, unsigned long len) {
    FILE *m = fopen("/proc/self/maps", "r");
    char line[512];
    unsigned long s, e, covered = 0;
    if (!m) return;
    while (fgets(line, sizeof line, m)) {
        if (sscanf(line, "%lx-%lx", &s, &e) == 2 && e > start && s < start + len) {
            fprintf(f, "    %s", line);
            covered += (e < start + len ? e : start + len) - (s > start ? s : start);
        }
    }
    fclose(m);
    fprintf(f, "    covered %lu of %lu bytes\n", covered, len);
}

static void fd_target(int fd, char *out, size_t n) {
    char path[64];
    snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    ssize_t k = readlink(path, out, n - 1);
    out[k > 0 ? k : 0] = 0;
}

int madvise(void *addr, size_t len, int advice) {
    if (!real_madvise) real_madvise = dlsym(RTLD_NEXT, "madvise");
    int r = real_madvise(addr, len, advice);
    if (r != 0) {
        int e = errno;
        FILE *f = log_file();
        if (f) {
            fprintf(f, "%.6f tid %ld madvise(%p, %zu, %d) = -1 errno %d (%s) maps %d\n", now(), (long)gettid(), addr, len,
                    advice, e, strerror(e), map_count());
            maps_covering(f, (unsigned long)addr, len);
        }
        errno = e;
    }
    return r;
}

int ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);
    if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    int r = real_ioctl(fd, request, arg);
    if (r != 0 && errno == ENOMEM) {
        int e = errno;
        FILE *f = log_file();
        if (f) {
            char target[256];
            fd_target(fd, target, sizeof target);
            fprintf(f, "%.6f tid %ld ioctl(%d -> %s, 0x%lx) = -1 ENOMEM maps %d\n", now(), (long)gettid(), fd, target,
                    request, map_count());
        }
        errno = e;
    }
    return r;
}

ssize_t write(int fd, const void *buf, size_t count) {
    if (!real_write) real_write = dlsym(RTLD_NEXT, "write");
    ssize_t r = real_write(fd, buf, count);
    if (r < 0 && errno == ENOMEM) {
        int e = errno;
        char target[256];
        fd_target(fd, target, sizeof target);
        if (strstr(target, "infiniband")) {
            FILE *f = log_file();
            if (f) fprintf(f, "%.6f tid %ld write(%d -> %s, %zu) = -1 ENOMEM\n", now(), (long)gettid(), fd, target, count);
        }
        errno = e;
    }
    return r;
}
