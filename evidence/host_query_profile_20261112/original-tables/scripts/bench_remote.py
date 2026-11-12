"""Per-document host row lookup on original trained classifier tables (CPU only).
Same per-document operation as StagedClassifier before GPU transfer: drop OOV id 0,
fetch rows, FP64 sum. Usage: bench_remote.py MODEL_NAME OUT.json"""
import sys, time, json, os, platform, hashlib
from pathlib import Path
import numpy as np
REPO = Path('/llm-align/duyimin/ngram-host-profile-20261112/repo'); sys.path.insert(0, str(REPO))
from tools.native_row_store import NativeRows, pack_native
W = Path('/llm-align/duyimin/ngram-controlled-deployment-20261111')
NEW_LIB = REPO / 'build/librows-zstd.so'; OLD_LIB = W / 'runtime-7fe4b6f/build/librows-allcodecs.so'
name, out = sys.argv[1], Path(sys.argv[2]); dataset = name.split('-')[0]
M = W / 'application-frontier/models' / name; D = W / 'data' / dataset
src = np.load(W / 'training-registered' / name / 'rows.npy', mmap_mode='r'); dim = src.shape[1]
selected = 'byte-dedup-zstd' if '-zero-' in name else 'generic-zstd'
own = Path('/llm-align/duyimin/ngram-host-profile-20261112/bundles') / name / 'dedup-raw' / 'rows.meft'
if not own.exists():
    own.parent.mkdir(parents=True, exist_ok=True); t = time.perf_counter()
    pack_native(M / 'source.raw', own, dim * 4, 256, deduplicate=True, compress=False)
    build_seconds = time.perf_counter() - t
else: build_seconds = None
ids = np.load(D / 'test-ids.npy', mmap_mode='r'); ptr = np.load(D / 'test-offsets.npy')
pick = np.sort(np.random.default_rng(20261112).choice(len(ptr) - 1, size=min(5000, len(ptr) - 1), replace=False))
docs = [np.asarray(ids[ptr[i]:ptr[i + 1]]) for i in pick]; docs = [d[d != 0].astype('<u8') for d in docs]

def run(fetch, passes=2):
    for _ in range(passes):
        times = []
        for d in docs:
            t = time.perf_counter_ns(); v = fetch(d); v.astype(np.float64).sum(0); times.append(time.perf_counter_ns() - t)
    a = np.array(times) / 1e3
    return {'p50_us': float(np.median(a)), 'p95_us': float(np.percentile(a, 95)), 'total_s': float(a.sum() / 1e6)}
def binding(r): return lambda d: np.frombuffer(r.rows(d.tolist()), dtype='<f4').reshape(len(d), dim)
def array(r): return lambda d: r.gather_array(d).view('<f4')

res = {'raw-mmap': run(lambda d: np.ascontiguousarray(src[d.astype(np.int64)]))}
with NativeRows(own, NEW_LIB) as r:
    res['dedup-raw new-lib gather_array'] = run(array(r))
    check = all(r.gather_array(d).tobytes() == np.ascontiguousarray(src[d.astype(np.int64)]).tobytes() for d in docs[:500])
for v in res.values(): v['x_raw_p50'] = v['p50_us'] / res['raw-mmap']['p50_us']
report = {'model': name, 'selected_layout': selected, 'documents': len(docs), 'doc_sample_seed': 20261112,
    'ids_per_doc_mean': float(np.mean([len(d) for d in docs])), 'rows': int(src.shape[0]), 'source_fp32_bytes': int(src.nbytes),
    'artifact_bytes': {selected: (M / 'bundles' / selected / 'rows.meft').stat().st_size, 'dedup-raw': own.stat().st_size},
    'dedup_raw_build_seconds': build_seconds, 'dedup_raw_bytes_match_source_on_500_docs': bool(check), 'timings': res,
    'affinity': sorted(os.sched_getaffinity(0)), 'environment': {'platform': platform.platform(), 'python': platform.python_version(),
    'numpy': np.__version__, 'cpu': open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': ')}}
out.write_text(json.dumps(report, indent=1))
for k, v in res.items(): print(f"{name:34s} {k:42s} p50 {v['p50_us']:9.1f} us  {v['x_raw_p50']:7.1f}x", flush=True)
