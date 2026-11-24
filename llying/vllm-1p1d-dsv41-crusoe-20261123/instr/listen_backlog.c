/* LD_PRELOAD shim: raise every listen() backlog to at least LISTEN_BACKLOG_MIN (4096; the kernel caps it at
 * net.core.somaxconn). The infera router listens with a backlog of 128; when AgentX's warmup switches phase, ~1,000
 * new connections arrive within a second, the SYN and accept queues overflow, the kernel falls back to SYN cookies and
 * resets the connections whose cookies fail (SyncookiesFailed equalled the resets, EXPERIMENT 4.2).
 *   gcc -shared -fPIC -O2 -o listen_backlog.so listen_backlog.c -ldl
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/socket.h>

int listen(int fd, int backlog)
{
    static int (*real_listen)(int, int);
    if (!real_listen)
        real_listen = (int (*)(int, int))dlsym(RTLD_NEXT, "listen");
    /* No atoi: glibc >= 2.38 headers turn it into __isoc23_strtol, which the image's glibc 2.35 lacks. */
    const char *env = getenv("LISTEN_BACKLOG_MIN");
    int min = 4096;
    if (env && *env) {
        min = 0;
        for (; *env >= '0' && *env <= '9'; env++)
            min = min * 10 + (*env - '0');
    }
    if (backlog >= 0 && backlog < min)
        backlog = min;
    return real_listen(fd, backlog);
}
