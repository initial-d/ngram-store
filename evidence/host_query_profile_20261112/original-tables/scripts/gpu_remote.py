"""End-to-end host path of the controlled grid (tools.benchmark_large_application),
same StagedClassifier, request set (first 256 test documents), warmup 2, repeat 3,
with three readers on one GPU. Usage: gpu_remote.py MODEL OUT.json"""
import sys, time, json, os, subprocess
from pathlib import Path
import numpy as np, torch
REPO = Path('/llm-align/duyimin/ngram-host-profile-20261112/repo'); sys.path.insert(0, str(REPO))
from tools.native_row_store import NativeRows
from tools.benchmark_large_application import StagedClassifier, SourceRows, document_batches
W = Path('/llm-align/duyimin/ngram-controlled-deployment-20261111')
NEW_LIB = REPO / 'build/librows-zstd.so'; OLD_LIB = W / 'runtime-7fe4b6f/build/librows-allcodecs.so'
name, out = sys.argv[1], Path(sys.argv[2]); dataset = name.split('-')[0]
M = W / 'application-frontier/models' / name; D = W / 'data' / dataset; S = W / 'training-registered' / name
selected = 'byte-dedup-zstd' if '-zero-' in name else 'generic-zstd'
own = Path('/llm-align/duyimin/ngram-host-profile-20261112/bundles') / name / 'dedup-raw' / 'rows.meft'
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
class Grid:      # = ApplicationRows native path used by the grid
    def __init__(s, path, lib, dim): s.r = NativeRows(path, lib); s.dim = dim
    def rows(s, ids): return np.frombuffer(s.r.rows(ids.tolist()), dtype='<f4').reshape(len(ids), s.dim)
    def close(s): s.r.close()
class Array:
    def __init__(s, path, lib): s.r = NativeRows(path, lib)
    def rows(s, ids): return s.r.gather_array(ids).view('<f4')
    def close(s): s.r.close()
ids = np.load(D / 'test-ids.npy', mmap_mode='r'); ptr = np.load(D / 'test-offsets.npy', mmap_mode='r')
labels = np.load(D / 'test-labels.npy')[:256]; classifier = np.load(S / 'classifier.npz')
dim = np.load(S / 'rows.npy', mmap_mode='r').shape[1]
readers = {'raw-mmap (SourceRows)': lambda: SourceRows(S),
           'dedup-raw gather_array': lambda: Array(own, NEW_LIB)}
report = {'model': name, 'gpu': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
          'affinity': sorted(os.sched_getaffinity(0)), 'results': {}}
logits_ref = None
for batch in (1, 32):
    requests = list(document_batches(ids, ptr, labels, batch))
    for label, make in readers.items():
        reader = make(); engine = StagedClassifier(reader, classifier, 'cuda', chunk_rows=4096)
        with torch.no_grad():
            logits = np.concatenate([engine(d).cpu().numpy() for _, d in requests])
            if logits_ref is None: logits_ref = logits
            same = bool(np.array_equal(logits, logits_ref))
            for _ in range(2):
                for _, d in requests: engine(d)
            torch.cuda.synchronize(); samples = []
            for _ in range(3):
                for _, d in requests:
                    torch.cuda.synchronize(); t = time.perf_counter_ns(); engine(d); torch.cuda.synchronize()
                    samples.append((time.perf_counter_ns() - t) / 1e6)
        reader.close()
        report['results'][f'{label} b{batch}'] = {'p50_ms': float(np.median(samples)), 'p95_ms': float(np.quantile(samples, .95)),
            'logits_equal_raw': same, 'samples': len(samples)}
        print(f'{name} b{batch:<2} {label:32s} p50 {np.median(samples):8.3f} ms  logits_equal={same}', flush=True)
    for k in [k for k in report['results'] if k.endswith(f'b{batch}')]:
        report['results'][k]['x_raw_p50'] = report['results'][k]['p50_ms'] / report['results'][f'raw-mmap (SourceRows) b{batch}']['p50_ms']
out.write_text(json.dumps(report, indent=1))
