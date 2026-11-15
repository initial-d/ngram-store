#!/usr/bin/env python3
"""Distinct rows in a token-level pooled n-gram memory.

Proposition 1: with pooled reads, row-local optimizers and equal initialization,
two rows stay byte-identical iff their (window, multiplicity) incidence sets are
equal. Distinct rows therefore follow from the token stream alone -- no training.

Grouping uses lexsort over (key hash, window id); the two are never packed into
one uint64, which would silently truncate the key. Token ids are int32 so that an
exact distinct-count can be taken by viewing the gram matrix as bytes.

Usage:
  count.py --tokens 16000000 --window 512 --stride 512 --orders 1,2,3,4 --epochs 1

The two WikiText-103 raw training shards must be in --data; download them from
Salesforce/wikitext (wikitext-103-raw-v1). The tokenizer path defaults to the
cached Qwen3-0.6B copy and can be overridden with --tokenizer.
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

QWEN = os.path.join(os.path.expanduser('~'), '.cache/huggingface/hub',
                    'models--Qwen--Qwen3-0.6B/snapshots')
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64 = np.uint64


def mix(x):
    """splitmix64 finalizer."""
    x = np.ascontiguousarray(x, dtype=np.uint64)
    with np.errstate(over='ignore'):
        for c in (0xff51afd7ed558ccd, 0xc4ceb9fe1a85ec53):
            x ^= x >> U64(33)
            x *= U64(c)
        x ^= x >> U64(33)
    return x


def gram_matrix(grid, n):
    """(W * span, n) int32 grams of order n."""
    n_win, window = grid.shape
    span = window - n + 1
    return np.stack([grid[:, j:j + span] for j in range(n)], axis=-1).reshape(-1, n)


def gram_hashes(grams, n):
    h = np.full(len(grams), U64(FNV_OFFSET), dtype=np.uint64)
    with np.errstate(over='ignore'):
        h += U64(n) * U64(1000003)          # the order belongs to the key
        for col in grams.T:
            h ^= col.astype(np.uint64)
            h *= U64(FNV_PRIME)
    return mix(h)


def analyse(grid, n, epochs, window, stride):
    n_win = len(grid)
    span = grid.shape[1] - n + 1
    grams = gram_matrix(grid, n)
    h = gram_hashes(grams, n)
    win = np.repeat(np.arange(n_win, dtype=np.int64), span)

    # sanity: hashing must not merge distinct keys. Compare on a prefix.
    m = min(len(grams), 4_000_000)
    exact = len(np.unique(grams[:m].view(np.dtype(f'V{4 * n}')).ravel()))
    hashed = len(np.unique(h[:m]))
    del grams

    order = np.lexsort((win, h))
    h, win = h[order], win[order]
    del order
    run = np.flatnonzero(np.r_[True, (h[1:] != h[:-1]) | (win[1:] != win[:-1])])
    key_at_run = h[run]
    mult = np.diff(np.r_[run, len(h)]).astype(np.uint64)
    win_at_run = win[run].astype(np.uint64)
    del h, win, run
    with np.errstate(over='ignore'):
        terms = mix(win_at_run * U64(1 << 22) + mult * U64(epochs))
    del win_at_run, mult

    key_start = np.flatnonzero(np.r_[True, key_at_run[1:] != key_at_run[:-1]])
    windows_per_key = np.diff(np.r_[key_start, len(key_at_run)])
    sig = np.add.reduceat(mix(terms), key_start)
    del terms, key_at_run

    groups, by_sig = np.unique(sig, return_inverse=True)
    n_keys = len(key_start)
    distinct_rows = len(groups)
    single = windows_per_key == 1
    rows_single = len(np.unique(by_sig[single]))
    rows_multi = len(np.unique(by_sig[~single]))
    mincount = {}
    for k in (2, 3, 5, 10, 20, 50, 100, 1000):
        keep = windows_per_key >= k
        nk = int(keep.sum())
        if nk:
            mincount[str(k)] = {'keys': nk, 'rows': int(len(np.unique(by_sig[keep]))),
                                'dedup': round(1 - len(np.unique(by_sig[keep])) / nk, 4)}
    return {
        'order': n, 'window': window, 'stride': stride, 'epochs': epochs,
        'windows': int(n_win), 'grams': int(len(sig)),
        'distinct_keys': int(n_keys), 'distinct_rows': int(distinct_rows),
        'duplicate_fraction': round((n_keys - distinct_rows) / n_keys, 4),
        'keys_in_one_window': int(single.sum()), 'rows_for_one_window_keys': int(rows_single),
        'dedup_among_one_window_keys': round(1 - rows_single / max(1, int(single.sum())), 4),
        'keys_in_several_windows': int((~single).sum()), 'rows_for_multi_window_keys': int(rows_multi),
        'dedup_among_multi_window_keys': round(1 - rows_multi / max(1, int((~single).sum())), 4),
        'rows_per_window': round(distinct_rows / n_win, 1),
        'keys_per_window': round(n_keys / n_win, 1),
        'hash_keys_vs_exact': [int(hashed), int(exact)],
        'min_windows': mincount,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=16_000_000)
    ap.add_argument('--window', type=int, default=512)
    ap.add_argument('--stride', type=int, default=512)
    ap.add_argument('--orders', default='1,2,3,4')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--out', default='')
    ap.add_argument('--data', default='data', help='directory holding the two WikiText parquet shards')
    ap.add_argument('--tokenizer', default=os.path.join(QWEN, 'tokenizer.json'))
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    files = [f'{args.data}/train-00000-of-00002.parquet',
             f'{args.data}/train-00001-of-00002.parquet']
    chunk, total = [], 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=20000, columns=['text']):
            for enc in tok.encode_batch(batch.column('text').to_pylist()):
                if enc.ids:
                    chunk.append(np.asarray(enc.ids, dtype=np.int32) + np.int32(1))
                    total += len(enc.ids)
            if total >= args.tokens:
                break
        if total >= args.tokens:
            break
    ids = np.concatenate(chunk)[:args.tokens]
    starts = np.arange(0, len(ids) - args.window + 1, args.stride, dtype=np.int64)
    grid = ids[starts[:, None] + np.arange(args.window, dtype=np.int64)]
    print(f'tokens {len(ids)} windows {len(grid)} window {args.window} stride {args.stride}',
          flush=True)

    rows = []
    for n in [int(x) for x in args.orders.split(',')]:
        t0 = time.perf_counter()
        st = analyse(grid, n, args.epochs, args.window, args.stride)
        st['seconds'] = round(time.perf_counter() - t0, 1)
        rows.append(st)
        print(json.dumps(st), flush=True)
        if st['hash_keys_vs_exact'][0] != st['hash_keys_vs_exact'][1]:
            print('  WARNING: hashed distinct keys != exact distinct keys', flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
