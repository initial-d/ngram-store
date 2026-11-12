import sys, time, os, json
from pathlib import Path
import numpy as np
sys.path.insert(0, '/Users/duyimin/paper_work/ngram-store')
from tools.native_row_store import pack_native
import os; M = Path(os.environ.get('MODEL', '/tmp/agnews/model-zero-lr10')); B = M / 'bundles'; B.mkdir(exist_ok=True)
R = np.load(M / 'rows.npy', mmap_mode='r'); raw = M / 'source.raw'
if not raw.exists():
    with raw.open('wb') as f:
        for b in range(0, len(R), 8192): R[b:b + 8192].tofile(f)
layouts = {  # name: kwargs   (experiment used block_rows=256, no max_block_bytes)
    'generic-zstd-256': dict(deduplicate=False, codec='zstd'),
    'dedup-zstd-256':   dict(deduplicate=True, codec='zstd'),
    'dedup-zstd-16':    dict(deduplicate=True, codec='zstd', max_block_bytes=8192),
    'dedup-zstd-4':     dict(deduplicate=True, codec='zstd', max_block_bytes=2048),
    'dedup-raw':        dict(deduplicate=True, compress=False),
    'plain-raw':        dict(deduplicate=False, compress=False),
}
for name in sys.argv[1:] or layouts:
    out = B / name / 'rows.meft'
    if out.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); t = time.perf_counter()
    info = pack_native(raw, out, 512, 256, predictor='none', **layouts[name])
    print(json.dumps({'layout': name, 'seconds': round(time.perf_counter() - t, 1), 'bytes': out.stat().st_size,
                      'saving_vs_fp32': round(1 - out.stat().st_size / raw.stat().st_size, 4)}), flush=True)
