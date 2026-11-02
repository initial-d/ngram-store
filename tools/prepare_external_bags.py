"""Prepare matched CSR from cached actual n-gram keys and registered words.

Preparation uses resident JSON registries and is measured separately from
bounded construction. No corpus download, new training or manufactured rows.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time

import numpy as np
from .external_cotraining import U64, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-dir', type=Path, required=True)
    p.add_argument('--words', type=Path, required=True)
    p.add_argument('--reference-report', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU-only: empty CUDA_VISIBLE_DEVICES required')
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    training = json.loads((cfg.training_dir / 'training.json').read_text())
    reference = json.loads(cfg.reference_report.read_text())
    for name, key in (('trained.f32', 'trained_sha256'), ('keys.json', 'keys_sha256'),
                      ('trained-source-ids.u64', 'source_row_ids_sha256')):
        if file_hash(cfg.training_dir / name) != training[key]:
            raise ValueError('cached source hash mismatch: ' + name)
    words = json.loads(cfg.words.read_text())
    keys = json.loads((cfg.training_dir / 'keys.json').read_text())
    if len(words) != reference['partition']['domain_bags'] or len(keys) != reference['partition']['original_rows']:
        raise ValueError('cached registry shape mismatch')
    lookup = {key: i for i, key in enumerate(keys)}
    order, dim = training['settings']['order'], training['dim']
    limits = (250000, 1000000, len(keys))
    states = []
    for limit in limits:
        out = cfg.output_dir / str(limit)
        out.mkdir()
        states.append({'rows': limit, 'dir': out, 'ids': (out / 'ids.u64').open('wb'),
                       'offsets': (out / 'offsets.u64').open('wb'), 'occurrences': 0,
                       'max_bag': 0, 'empty_bags': 0, 'digest': hashlib.sha256()})
        states[-1]['offsets'].write(U64.pack(0))
    try:
        for word in words:
            bounded = '<' + word + '>'
            # Same complete occurrence list as train_ngram_similarity.ngrams.
            bag = [lookup[g] for i in range(len(bounded)) for n in range(2, order + 1)
                   if i + n <= len(bounded) for g in (bounded[i:i + n],) if g in lookup]
            for state in states:
                selected = [row for row in bag if row < state['rows']]
                np.asarray(selected, dtype='<u8').tofile(state['ids'])
                state['occurrences'] += len(selected)
                state['max_bag'] = max(state['max_bag'], len(selected))
                state['empty_bags'] += not selected
                state['offsets'].write(U64.pack(state['occurrences']))
                pairs = sorted(Counter(selected).items())
                state['digest'].update(U64.pack(len(pairs)))
                state['digest'].update(np.asarray(pairs, dtype='<u8').tobytes())
    finally:
        for state in states:
            state['ids'].close()
            state['offsets'].close()
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'command': shlex.join([sys.executable, *sys.argv]), 'python': sys.version,
              'numpy': np.__version__, 'os': platform.platform(), 'device': 'cpu',
              'source_table_sha256': training['trained_sha256'],
              'source_keys_sha256': training['keys_sha256'],
              'source_training_ids_sha256': training['source_row_ids_sha256'],
              'words_sha256': file_hash(cfg.words), 'reference_report_sha256': file_hash(cfg.reference_report),
              'source_files_sha256': {name: file_hash(Path(__file__).with_name(name)) for name in
                                     ('prepare_external_bags.py', 'external_cotraining.py')},
              'sets': [], 'limitations': ['Cached JSON preprocessing is not a bounded-memory tokenizer/key registry.',
                  'Actual original-row prefixes; no independent training, new quality or GPU claim.']}
    for state in states:
        out = state['dir']
        if state['rows'] == len(keys):
            source = cfg.training_dir / 'trained.f32'
            if state['digest'].hexdigest() != reference['partition']['domain_sha256']:
                raise ValueError('full CSR changes previous registered domain')
        else:
            source = out / 'source.f32'
            with (cfg.training_dir / 'trained.f32').open('rb') as original, source.open('wb') as target:
                remaining = state['rows'] * dim * 4
                while remaining:
                    block = original.read(min(65536, remaining))
                    if not block:
                        raise ValueError('truncated trained source')
                    target.write(block)
                    remaining -= len(block)
        item = {'rows': state['rows'], 'dim': dim, 'bags': len(words), 'occurrences': state['occurrences'],
                'maximum_bag_occurrences': state['max_bag'], 'empty_bags': state['empty_bags'],
                'domain_sha256': state['digest'].hexdigest(),
                'source': str(source), 'source_sha256': file_hash(source),
                'ids': str(out / 'ids.u64'), 'offsets': str(out / 'offsets.u64'),
                'ids_sha256': file_hash(out / 'ids.u64'), 'offsets_sha256': file_hash(out / 'offsets.u64')}
        (out / 'input.json').write_text(json.dumps(item, indent=2) + '\n')
        report['sets'].append(item)
    report.update(seconds=time.perf_counter() - started, status='completed',
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                  (1 if sys.platform == 'darwin' else 1024))
    (cfg.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'prepared': [s['rows'] for s in report['sets']], 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
