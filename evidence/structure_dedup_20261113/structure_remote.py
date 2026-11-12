"""Joint .99/.99 prefix/suffix sharing re-stored as dedup + uncompressed.
Usage: structure_remote.py OUT.json"""
import sys, json, gzip, time, os
from pathlib import Path
import numpy as np
REPO = Path('/llm-align/duyimin/ngram-host-profile-20261112/repo'); sys.path.insert(0, str(REPO))
from tools.native_row_store import NativeRows, pack_native
LIB = REPO / 'build/librows-zstd.so'
T = Path('/llm-align/duyimin/ngram-tree-node-quality-20261109/mechanism-analysis/points')
WORK = Path('/llm-align/duyimin/ngram-host-profile-20261112/structure'); WORK.mkdir(exist_ok=True)
rng = np.random.default_rng(20261113)

def decode(path, n):
    with NativeRows(path, LIB) as r: return r.gather_array(np.arange(n)).copy()
def timing(fetch, bags, passes=2):
    for _ in range(passes):
        t = []
        for b in bags:
            s = time.perf_counter_ns(); fetch(b).view('<f4').astype(np.float64).sum(0); t.append(time.perf_counter_ns() - s)
    return float(np.median(t)) / 1e3

out = []
for seed in (3, 7, 11):
    for step in (2000, 8000, 32000):
        name = f'order-12-seed-{seed}-average-init-1-step-{step}'; P = T / name
        codec = json.load(open(P / 'datrie/codec.json')); n, dim = codec['shape']; rb = dim * 4
        src = decode(P / 'identity/table.men', n); approx = decode(P / 'datrie/table.men', n)
        W = WORK / name; W.mkdir(exist_ok=True); (W / 'approx.f32').write_bytes(approx.tobytes()); (W / 'source.f32').write_bytes(src.tobytes())
        for f in ('joint-dedup-raw.men', 'source-dedup-raw.men'):
            if (W / f).exists(): (W / f).unlink()
        pack_native(W / 'approx.f32', W / 'joint-dedup-raw.men', rb, 256, deduplicate=True, compress=False)
        pack_native(W / 'source.f32', W / 'source-dedup-raw.men', rb, 256, deduplicate=True, compress=False)
        bags = [rng.integers(0, n, 40).astype('<u8') for _ in range(5000)]
        res = {'raw (in-memory array)': timing(lambda b: src.view(np.uint8).reshape(n, rb)[b.astype(np.int64)], bags)}
        for label, path in (('identity xor+zlib (original)', P / 'identity/table.men'), ('joint xor+zlib (original)', P / 'datrie/table.men'),
                            ('joint dedup uncompressed', W / 'joint-dedup-raw.men')):
            with NativeRows(path, LIB) as r:
                res[label] = timing(r.gather_array, bags)
                assert r.gather_array(np.arange(n)).tobytes() == (approx if 'joint' in label else src).tobytes()
        q = {}
        for part in ('test', 'confirmation'):
            iv = json.load(gzip.open(P / f'datrie-{part}-intervals.json.gz'))
            q[part] = {k: iv[k]['loss_relative_change_95_percent_interval'] for k in ('all', 'oov_nonempty')}
        size = lambda p: os.path.getsize(p)
        row = {'point': name, 'rows': n, 'dim': dim, 'raw_bytes': n * rb,
               'unique_rows_joint': int(len(np.unique(approx.view(f'V{rb}')))), 'unique_rows_source': int(len(np.unique(src.view(f'V{rb}')))),
               'bytes': {'identity xor+zlib (original)': size(P / 'identity/table.men'), 'joint xor+zlib (original)': size(P / 'datrie/table.men'),
                         'joint dedup uncompressed': size(W / 'joint-dedup-raw.men'), 'source dedup uncompressed': size(W / 'source-dedup-raw.men')},
               'lookup_p50_us': res, 'joint_loss_relative_change_95ci': q}
        out.append(row)
        b = row['bytes']; print(f"{name:42s} rows {n:6d} uniq-joint {row['unique_rows_joint']:6d} | saving: joint-dedup-raw {1-b['joint dedup uncompressed']/row['raw_bytes']:.3f} joint-zlib {1-b['joint xor+zlib (original)']/row['raw_bytes']:.3f} id-zlib {1-b['identity xor+zlib (original)']/row['raw_bytes']:.3f} | p50 us " + ' '.join(f'{v:.1f}' for v in res.values()), flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
