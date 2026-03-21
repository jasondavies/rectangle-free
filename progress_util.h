#ifndef RECTANGLE_FREE_PROGRESS_UTIL_H
#define RECTANGLE_FREE_PROGRESS_UTIL_H

#include <stdio.h>
#include <unistd.h>

typedef struct {
    FILE* stream;
    int is_tty;
    long long last_reported;
    int line_active;
} ProgressReporter;

static inline void progress_format_duration(double seconds, char* out, size_t out_size) {
    if (seconds < 0.0) seconds = 0.0;
    long long rounded = (long long)(seconds + 0.5);
    long long hours = rounded / 3600;
    long long minutes = (rounded % 3600) / 60;
    long long secs = rounded % 60;
    snprintf(out, out_size, "%lld:%02lld:%02lld", hours, minutes, secs);
}

static inline void progress_reporter_init(ProgressReporter* reporter, FILE* stream) {
    reporter->stream = stream;
    reporter->is_tty = isatty(fileno(stream));
    reporter->last_reported = -1;
    reporter->line_active = 0;
}

static inline void progress_reporter_print_initial(ProgressReporter* reporter, long long total_tasks) {
    if (reporter->is_tty) {
        fprintf(reporter->stream, "\rProgress: 0/%lld (0.00%%)", total_tasks);
        reporter->line_active = 1;
    } else {
        fprintf(reporter->stream, "Progress: 0/%lld (0.00%%)\n", total_tasks);
    }
    fflush(reporter->stream);
    reporter->last_reported = 0;
}

static inline void progress_reporter_maybe_report(ProgressReporter* reporter, long long done,
                                                  long long total_tasks, long long report_step,
                                                  double start_time, double current_time) {
    if (done != total_tasks && (report_step <= 0 || (done % report_step) != 0)) return;

    #pragma omp critical(progress_output)
    {
        if (done > reporter->last_reported) {
            reporter->last_reported = done;

            double elapsed = current_time - start_time;
            double pct = total_tasks > 0 ? (100.0 * (double)done / (double)total_tasks) : 100.0;
            double rate = elapsed > 0.0 ? ((double)done / elapsed) : 0.0;
            double eta = (rate > 0.0 && total_tasks > done) ? ((double)(total_tasks - done) / rate) : 0.0;
            char elapsed_str[32];
            char eta_str[32];
            progress_format_duration(elapsed, elapsed_str, sizeof(elapsed_str));
            progress_format_duration(eta, eta_str, sizeof(eta_str));

            fprintf(reporter->stream,
                    reporter->is_tty
                        ? "\rProgress: %lld/%lld (%.2f%%, %.1f tasks/s, elapsed %s, ETA %s)"
                        : "Progress: %lld/%lld (%.2f%%, %.1f tasks/s, elapsed %s, ETA %s)\n",
                    done, total_tasks, pct, rate, elapsed_str, eta_str);
            fflush(reporter->stream);
            reporter->line_active = reporter->is_tty;
        }
    }
}

static inline void progress_reporter_finish(ProgressReporter* reporter) {
    if (reporter->is_tty && reporter->line_active) {
        fputc('\n', reporter->stream);
        fflush(reporter->stream);
        reporter->line_active = 0;
    }
}

#endif
