"""Do byte-identical rows coincide with identical bag-incidence signatures?

AG News prepared with tools.large_ngram_application prepare; zero-init model
trained by evidence/host_query_profile_20261112/scripts/train_cpu.py.
Usage: check.py DATA_DIR MODEL_DIR > result.json
"""
import json, sys
import numpy as np
data, model = sys.argv[1], sys.argv[2]
ids = np.load(f'{data}/train-ids.npy'); ptr = np.load(f'{data}/train-offsets.npy'); counts = np.load(f'{data}/counts.npy')
rows = len(counts) + 1
doc = np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))
pair, mult = np.unique(ids.astype(np.int64) * (1 << 20) + doc, return_counts=True)
key, d = pair >> 20, pair & ((1 << 20) - 1)
def mix(x):
    x = x.astype(np.uint64)
    with np.errstate(over='ignore'):
        for c in (0xff51afd7ed558ccd, 0xc4ceb9fe1a85ec53):
            x ^= x >> np.uint64(33); x *= np.uint64(c)
        x ^= x >> np.uint64(33)
    return x
sig = np.zeros(rows, dtype=np.uint64)            # order-free hash of the (document, multiplicity) set
with np.errstate(over='ignore'):
    np.add.at(sig, key, mix(d * 4096 + mult))
table = np.load(f'{model}/rows.npy', mmap_mode='r')
_, by_sig = np.unique(sig[1:], return_inverse=True)
_, by_row = np.unique(np.ascontiguousarray(table[1:]).view('V512').ravel(), return_inverse=True)
joint = len(np.unique(np.stack([by_sig, by_row], 1), axis=0))
print(json.dumps({'rows_excluding_padding': rows - 1, 'incidence_groups': int(by_sig.max() + 1),
    'byte_groups': int(by_row.max() + 1), 'joint_groups': joint,
    'partitions_identical': bool(joint == by_sig.max() + 1 == by_row.max() + 1),
    'keys_occurring_once_fraction': float((counts == 1).mean())}, indent=1))
