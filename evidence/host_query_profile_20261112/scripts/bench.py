"""Per-document host row lookup on the full AG News test set (CPU only).
Mirrors StagedClassifier: per document, drop OOV id 0, fetch rows, FP64 sum."""
import sys, time, json, ctypes
from pathlib import Path
import numpy as np
sys.path.insert(0, '/Users/duyimin/paper_work/ngram-store')
from tools.native_row_store import NativeRows
import os; M = Path(os.environ.get('MODEL', '/tmp/agnews/model-zero-lr10')); D = Path('/tmp/agnews/data')
ids = np.load(D / 'test-ids.npy'); ptr = np.load(D / 'test-offsets.npy')
docs = [ids[ptr[i]:ptr[i + 1]] for i in range(len(ptr) - 1)]
docs = [d[d != 0].astype('<u8') for d in docs]
R = np.load(M / 'rows.npy', mmap_mode='r'); DIM = 128

def raw_doc(d): return np.ascontiguousarray(R[d.astype(np.int64)])
def make_binding(reader):      # what the experiment did (ApplicationRows.rows)
    def f(d): return np.frombuffer(reader.rows(d.tolist()), dtype='<f4').reshape(len(d), DIM)
    return f
def make_direct(reader):       # numpy buffers straight to C, no Python list/ctypes array
    def f(d):
        d = np.ascontiguousarray(d, dtype='<u8'); out = np.empty((len(d), DIM), dtype='<f4')
        reader.gather_into(d.ctypes.data, len(d), out.ctypes.data, out.nbytes); return out
    return f

def run(fetch, batch=None, passes=2):
    for p in range(passes):
        times = []
        if batch is None:
            for d in docs:
                t = time.perf_counter_ns(); v = fetch(d); v.astype(np.float64).sum(0); times.append(time.perf_counter_ns() - t)
        else:   # one call per batch of documents, split sums afterwards
            for b in range(0, len(docs), batch):
                group = docs[b:b + batch]; cat = np.concatenate(group); cuts = np.cumsum([len(g) for g in group])[:-1]
                t = time.perf_counter_ns(); v = fetch(cat).astype(np.float64)
                [x.sum(0) for x in np.split(v, cuts)]; times.append(time.perf_counter_ns() - t)
    a = np.array(times) / 1e3
    return {'p50_us': round(float(np.median(a)), 1), 'p95_us': round(float(np.percentile(a, 95)), 1), 'total_s': round(a.sum() / 1e6, 3)}

results = {}
results['raw-mmap per-doc'] = run(raw_doc)
results['raw-mmap batch32'] = run(raw_doc, batch=32)
for name in sys.argv[1:]:
    with NativeRows(M / 'bundles' / name / 'rows.meft') as r:
        results[name + ' binding per-doc'] = run(make_binding(r))
        results[name + ' direct per-doc'] = run(make_direct(r))
        results[name + ' direct batch32'] = run(make_direct(r), batch=32)
        results[name + ' gather_array per-doc'] = run(lambda d: r.gather_array(d).view('<f4'))
        r.gather_into  # keep handle
base_doc = results['raw-mmap per-doc']['p50_us']; base_b = results['raw-mmap batch32']['p50_us']
for k, v in results.items():
    v['x_raw_p50'] = round(v['p50_us'] / (base_b if 'batch32' in k else base_doc), 1)
    print(f"{k:38s} p50 {v['p50_us']:>9} us  p95 {v['p95_us']:>9} us  total {v['total_s']:>7} s  {v['x_raw_p50']}x raw")
import os, platform
sizes = {n: (M / 'bundles' / n / 'rows.meft').stat().st_size for n in sys.argv[1:]}
json.dump({'model': str(M.name), 'documents': len(docs), 'ids_per_doc_mean': float(np.mean([len(d) for d in docs])),
           'source_fp32_bytes': int(R.nbytes), 'artifact_bytes': sizes, 'timings': results,
           'environment': {'platform': platform.platform(), 'python': platform.python_version(), 'numpy': np.__version__}},
          open(os.environ['OUT'], 'w'), indent=1)
