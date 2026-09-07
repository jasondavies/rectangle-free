#ifndef RECTANGLE_DURABLE_FILE_H
#define RECTANGLE_DURABLE_FILE_H
// POSIX publication of an already serialized payload. Parent directories must
// exist durably. Success means data AND the directory entry have been fsynced.
// Errors after link/rename may leave a valid published file: callers must not
// acknowledge success, nor delete that file to attempt recovery.
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Compile-time fault injection for the standalone regression test only.
#ifndef RECT_FILE_WRITE
#define RECT_FILE_WRITE write
#endif
#ifndef RECT_FILE_FSYNC
#define RECT_FILE_FSYNC fsync
#endif
#ifndef RECT_FILE_CLOSE
#define RECT_FILE_CLOSE close
#endif
#ifndef RECT_FILE_LINK
#define RECT_FILE_LINK link
#endif
#ifndef RECT_FILE_RENAME
#define RECT_FILE_RENAME rename
#endif

typedef struct { const void* data; size_t size; } RectFilePart;

static inline int rect_file_sync(int fd) {
    int status;
    do { status = RECT_FILE_FSYNC(fd); } while (status < 0 && errno == EINTR);
    return status;
}

// replace=0 publishes immutably; replace=1 atomically replaces a checkpoint.
static inline int rect_publish_file(const char* path, const RectFilePart* parts,
                                    size_t count, int replace) {
    int fd = -1, directory = -1, failed = 0, saved = 0;
    size_t length = strlen(path);
    char* temporary = (char*)malloc(length + sizeof(".tmp.XXXXXX"));
    char* parent = (char*)malloc(length + 2);
    if (!temporary || !parent) {
        free(temporary); free(parent); errno = ENOMEM; return -1;
    }
    memcpy(temporary, path, length);
    memcpy(temporary + length, ".tmp.XXXXXX", sizeof(".tmp.XXXXXX"));
    memcpy(parent, path, length + 1);
    char* slash = strrchr(parent, '/');
    if (!slash) strcpy(parent, ".");
    else if (slash == parent) slash[1] = '\0';
    else *slash = '\0';
    directory = open(parent, O_RDONLY | O_DIRECTORY);
    if (directory < 0) failed = 1;
    if (!failed) { fd = mkstemp(temporary); if (fd < 0) failed = 1; }
    for (size_t i = 0; !failed && i < count; ++i) {
        const char* data = (const char*)parts[i].data;
        size_t remaining = parts[i].size;
        while (remaining) {
            // Bounded writes also avoid implementation-defined >SSIZE_MAX sizes.
            size_t chunk = remaining < (1U << 20) ? remaining : (1U << 20);
            ssize_t written = RECT_FILE_WRITE(fd, data, chunk);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) { if (!written) errno = EIO; failed = 1; break; }
            data += written; remaining -= (size_t)written;
        }
    }
    if (!failed && rect_file_sync(fd)) failed = 1;
    if (failed) saved = errno;
    if (fd >= 0 && RECT_FILE_CLOSE(fd) && !failed) { failed = 1; saved = errno; }
    // Do not retry close on EINTR: the descriptor may already have been closed.
    if (!failed) {
        int status = replace ? RECT_FILE_RENAME(temporary, path)
                             : RECT_FILE_LINK(temporary, path);
        if (status) { failed = 1; saved = errno; }
    }
    // After rename there is no temporary pathname. After link, remove only
    // our private name, never the destination (including on later sync errors).
    if (fd >= 0 && unlink(temporary) && errno != ENOENT && !failed) {
        failed = 1; saved = errno;
    }
    if (!failed && rect_file_sync(directory)) { failed = 1; saved = errno; }
    if (directory >= 0 && RECT_FILE_CLOSE(directory) && !failed) {
        failed = 1; saved = errno;
    }
    free(temporary); free(parent);
    if (failed) { errno = saved; return -1; }
    return 0;
}
#endif
