"""Matched old/new certificate and native-plan compilation on real trained bags.

Uses existing SGNS checkpoints and source-pinned prior Python constructor.
All native plan arrays are byte-compared; no new vectors or GPU work.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .cotraining_groups import BagPartition, CertifiedBagPlan


def main():
    import torch
    from .benchmark_heldout_compression import split_tokens
    from .train_ngram_similarity import ngrams
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-root', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--reference-module', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=15)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU-only: empty CUDA_VISIBLE_DEVICES required')
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('tools._old_bag_certificates', cfg.reference_module)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    split = split_tokens(cfg.corpus.read_text())
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'reference_module_sha256': hashlib.sha256(cfg.reference_module.read_bytes()).hexdigest(),
              'current_module_sha256': hashlib.sha256(Path(__file__).with_name('cotraining_groups.py').read_bytes()).hexdigest(),
              'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'os': platform.platform(),
              'limitations': ['Real cached SGNS bags, no new trained vectors or GPU work.',
                'Certificate/plan compilation only, not native reducer or full-model latency.',
                'Paired interleaved timings on shared warm CPU, no exclusive-speed claim.'], 'runs': []}
    for order in (4, 8, 12):
        for seed in (3, 7, 11):
            name = f'order-{order}-seed-{seed}'
            checkpoint = torch.load(cfg.checkpoint_root / name / 'reference.pt', map_location='cpu', weights_only=True)
            words, grams = checkpoint['words'], checkpoint['grams']
            offset = len(words) + 1
            bags = [[int(i) for i in bag if i != 0] for bag in checkpoint['bags'].numpy()]
            partition = BagPartition.fit(bags, len(checkpoint['input_rows']), protected=range(offset))
            lookup = {g: i + offset for i, g in enumerate(grams)}
            ids = {w: i for i, w in enumerate(words)}
            rng = np.random.default_rng(42)
            known = sorted(set(split['test']) & set(words))
            oov = sorted(set(split['test']) - set(words))
            known = [known[i] for i in rng.choice(len(known), min(512, len(known)), replace=False)]
            oov = [oov[i] for i in rng.choice(len(oov), min(512, len(oov)), replace=False)]
            known_bags = [bags[ids[w]] for w in known]
            mixed = known_bags + [[lookup[g] for g in ngrams(w, order) if g in lookup] for w in oov]
            expected = np.asarray([partition.certifies(b) for b in mixed])
            np.testing.assert_array_equal(partition.certifies_many(mixed), expected)
            a, b = old.CertifiedBagPlan(partition, known_bags, padding_idx=0), CertifiedBagPlan(partition, known_bags, padding_idx=0)
            fields = ('ids', 'weights', 'offsets', 'denominators')
            if any(getattr(a, f).tobytes() != getattr(b, f).tobytes() for f in fields):
                raise ValueError('native compiled plan changed')
            timings = {key: [] for key in ('scalar-certificate', 'batch-certificate', 'old-native-plan', 'batch-native-plan')}
            functions = {'scalar-certificate': lambda: [partition.certifies(x) for x in mixed],
                'batch-certificate': lambda: partition.certifies_many(mixed),
                'old-native-plan': lambda: old.CertifiedBagPlan(partition, known_bags, padding_idx=0),
                'batch-native-plan': lambda: CertifiedBagPlan(partition, known_bags, padding_idx=0)}
            for _ in range(2):
                for f in functions.values():
                    f()
            for _ in range(cfg.repeat):
                for key in rng.permutation(list(functions)):
                    start = time.perf_counter()
                    functions[key]()
                    timings[key].append(time.perf_counter() - start)
            report['runs'].append({'order': order, 'seed': seed, 'known_bags': len(known_bags),
                'mixed_bags': len(mixed), 'mixed_accepted': int(expected.sum()),
                'plan_arrays_byte_identical': True,
                'plan_sha256': hashlib.sha256(b''.join(getattr(b, f).tobytes() for f in fields)).hexdigest(),
                'timing': {key: {'samples_seconds': samples, 'p50_ms': float(np.median(samples) * 1000),
                                    'p95_ms': float(np.quantile(samples, .95) * 1000)} for key, samples in timings.items()}})
    cfg.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
