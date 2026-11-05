/* Sanitizer driver: exercises a valid artifact and mutations of its bytes. */
#include "meft_rows.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static int check_bounded_compiler(void) {
    uint64_t rows[3] = {0, 1, 2}, counts[3] = {1, 1, 1}, offsets[2] = {0, 3};
    uint64_t mapping[3] = {0, 1, 0}, sizes[2] = {2, 1};
    double norms[3] = {1, 0, 1}, totals[2] = {2, 0}, defects[2] = {0, 0};
    uint64_t ids[3], plan_offsets[2]; float weights[3]; double bound[1]; unsigned char exact[1];
    size_t used; char error[256];
#define COMPILE_BOUND(capacity, bag_capacity, row_input) meft_rows_compile_bounded_bags( \
    row_input, counts, offsets, 3, 1, mapping, 3, sizes, 2, norms, totals, defects, \
    ids, weights, plan_offsets, bound, exact, capacity, bag_capacity, &used, error, sizeof(error))
    if (COMPILE_BOUND(3, 1, rows) || used != 2 || ids[0] != 0 || ids[1] != 1 ||
        weights[0] != 2 || weights[1] != 1 || !exact[0] || bound[0] > 1e-12) return 1;
    if (!COMPILE_BOUND(2, 1, rows) || !COMPILE_BOUND(3, 0, rows) || !COMPILE_BOUND(3, 1, NULL)) return 1;
    rows[1] = 0;
    if (!COMPILE_BOUND(3, 1, rows)) return 1;
    rows[1] = 1; mapping[0] = 2;
    if (!COMPILE_BOUND(3, 1, rows)) return 1;
    mapping[0] = 0; counts[0] = 1u << 24;
    if (!COMPILE_BOUND(3, 1, rows)) return 1;
    counts[0] = 1; norms[0] = -1;
    if (!COMPILE_BOUND(3, 1, rows)) return 1;
    norms[0] = 1; offsets[1] = 4;
    if (!COMPILE_BOUND(3, 1, rows)) return 1;
#undef COMPILE_BOUND
    return 0;
}
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    if (check_bounded_compiler()) return 1;
    char error[256];
    meft_rows *s = meft_rows_open(argv[1], error, sizeof(error));
    if (!s) return 0; /* malformed files are expected to be rejected */
    uint64_t n = meft_rows_count(s), width = meft_rows_width(s);
    if (width > 1048576) { meft_rows_close(s); return 0; }
    unsigned char *out = malloc((size_t)width * 4);
    if (!out) { meft_rows_close(s); return 1; }
    uint64_t ids[4] = {0, n ? n - 1 : 0, 0, n};
    (void)meft_rows_set_cache_bytes(s, 65536);
    (void)meft_rows_gather(s, ids, 3, out, (size_t)width * 4, error, sizeof(error));
    (void)meft_rows_gather(s, ids, 3, out, (size_t)width * 4, error, sizeof(error));
    (void)meft_rows_gather(s, ids, 4, out, (size_t)width * 4, error, sizeof(error));
    /* Exercise lazy hash growth, LRU unlink/eviction and budget resets without
     * allocating a full query/table. Malformed reads may legitimately fail. */
    const size_t budgets[] = {65536, 8192, 0, 65536};
    for (int policy = 0; policy <= 1; ++policy) {
        (void)meft_rows_set_cache_policy(s, policy);
        for (size_t budget = 0; budget < sizeof(budgets) / sizeof(budgets[0]); ++budget) {
            (void)meft_rows_set_cache_bytes(s, budgets[budget]);
            uint64_t limit = n < 300 ? n : 300;
            for (uint64_t row = 0; row < limit; ++row)
                (void)meft_rows_gather(s, &row, 1, out, (size_t)width * 4, error, sizeof(error));
            for (uint64_t row = limit; row;) {
                --row;
                (void)meft_rows_gather(s, &row, 1, out, (size_t)width * 4, error, sizeof(error));
            }
        }
    }
    (void)meft_rows_set_cache_bytes(s, 1);
    (void)meft_rows_gather(s, ids, 3, out, (size_t)width * 4, error, sizeof(error));
    (void)meft_rows_set_cache_bytes(s, 0);
    (void)meft_rows_gather_chunked(s, ids, 3, out, (size_t)width * 4, 1, error, sizeof(error));
    (void)meft_rows_gather_chunked(s, ids, 4, out, (size_t)width * 4, 2, error, sizeof(error));
    (void)meft_rows_gather_chunked(s, ids, 3, out, 0, 2, error, sizeof(error));
    (void)meft_rows_gather_chunked(s, ids, 3, out, (size_t)width * 4, 0, error, sizeof(error));
    (void)meft_rows_gather_chunked(s, NULL, 3, out, (size_t)width * 4, 2, error, sizeof(error));
    (void)meft_rows_gather_chunked(NULL, ids, 3, out, (size_t)width * 4, 2, error, sizeof(error));
    uint64_t offsets[3] = {0, 2, 3}, bad_offsets[3] = {0, 3, 2};
    (void)meft_rows_set_cache_bytes(s, 65536);
    unsigned char weights[12] = {0}; /* Enough LE FP32/FP16/BF16 scalar bytes. */
    for (int dtype = 1; dtype <= 3; ++dtype) {
        (void)meft_rows_embedding_bag(s, ids, 3, offsets, 2, dtype, 1, UINT64_MAX, 2,
                                     (float *)out, (size_t)width, error, sizeof(error));
        (void)meft_rows_embedding_bag(s, ids, 4, offsets, 2, dtype, 0, UINT64_MAX, 1,
                                     (float *)out, (size_t)width, error, sizeof(error));
        (void)meft_rows_embedding_bag(s, ids, 3, bad_offsets, 2, dtype, 0, UINT64_MAX, 1,
                                     (float *)out, (size_t)width, error, sizeof(error));
        (void)meft_rows_embedding_bag(s, ids, 3, offsets, 2, dtype, 0, UINT64_MAX, 0,
                                     (float *)out, (size_t)width, error, sizeof(error));
        (void)meft_rows_embedding_bag(s, ids, 3, offsets, 2, dtype, 0, UINT64_MAX, 1,
                                     (float *)out, 0, error, sizeof(error));
        (void)meft_rows_embedding_bag_weighted(s, ids, 3, offsets, 2, dtype,
                                              weights, sizeof(weights), UINT64_MAX, 1,
                                              (float *)out, (size_t)width, error, sizeof(error));
        (void)meft_rows_embedding_bag_weighted(s, ids, 3, offsets, 2, dtype,
                                              weights, 1, UINT64_MAX, 1,
                                              (float *)out, (size_t)width, error, sizeof(error));
        (void)meft_rows_embedding_bag_weighted(s, ids, 3, offsets, 2, dtype,
                                              NULL, sizeof(weights), UINT64_MAX, 1,
                                              (float *)out, (size_t)width, error, sizeof(error));
        (void)meft_rows_embedding_bag_weighted(s, ids, 3, bad_offsets, 2, dtype,
                                              weights, sizeof(weights), UINT64_MAX, 2,
                                              (float *)out, (size_t)width, error, sizeof(error));
    }
    free(out); meft_rows_close(s); return 0;
}
