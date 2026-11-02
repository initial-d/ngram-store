"""Independent-corpus paired SGNS checkpoints with train-only incidence groups.

Reads a cached public benchmark zip without fetching data. A deterministic token
prefix and all settings are recorded before training. Reports finite-training
and held-out SGNS evidence, not convergence, semantic or LLM quality.
"""
import argparse
import codecs
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import zipfile

import numpy as np
import torch

from .benchmark_cotraining import train_dynamics, vector_metrics, write_json
from .benchmark_heldout_compression import compose, deltas, evaluation_set, score, split_tokens, summarize
from .benchmark_faiss_cotraining import digest_file


def corpus_prefix(path, member, tokens, prefix_bytes):
    """Fixed raw benchmark prefix; preserve regex-token order, including markup."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as stream:
            raw = stream.read(prefix_bytes)
    # Delay only an incomplete UTF-8 suffix; malformed interior bytes fail.
    text = codecs.getincrementaldecoder('utf-8')().decode(raw, final=False)
    selected = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    if len(selected) < tokens:
        raise ValueError('fixed byte prefix has fewer than the requested tokens')
    return ' '.join(selected[:tokens]), hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus-zip', type=Path, required=True)
    p.add_argument('--member', default='enwik8')
    p.add_argument('--corpus-source', required=True)
    p.add_argument('--expected-zip-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=200000)
    p.add_argument('--prefix-bytes', type=int, default=4 * 1024**2)
    p.add_argument('--max-orders', default='4,8,12')
    p.add_argument('--seeds', default='3,7,11')
    p.add_argument('--conventions', default='legacy,average')
    p.add_argument('--checkpoints', default='0,100,500,2000')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--vocab-limit', type=int, default=1200)
    p.add_argument('--min-count', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--negatives', type=int, default=5)
    p.add_argument('--eval-pairs', type=int, default=8192)
    p.add_argument('--lr', type=float, default=.05)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU only: CUDA_VISIBLE_DEVICES must be empty')
    if min(cfg.tokens, cfg.prefix_bytes, cfg.steps, cfg.dim, cfg.vocab_limit,
           cfg.min_count, cfg.batch_size, cfg.negatives, cfg.eval_pairs) < 1 or cfg.lr <= 0:
        p.error('positive training sizes and learning rate required')
    orders, seeds = [int(v) for v in cfg.max_orders.split(',')], [int(v) for v in cfg.seeds.split(',')]
    conventions = cfg.conventions.split(',')
    if min(orders) < 2 or any(v not in ('legacy', 'average') for v in conventions):
        p.error('order >= 2 and legacy/average conventions required')
    zip_hash = digest_file(cfg.corpus_zip)
    if zip_hash != cfg.expected_zip_sha256:
        raise ValueError('cached corpus zip hash differs from the predeclared source')
    text, raw_hash = corpus_prefix(cfg.corpus_zip, cfg.member, cfg.tokens, cfg.prefix_bytes)
    cfg.output_dir = cfg.output_dir.resolve()
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    corpus = cfg.output_dir / 'corpus.txt'
    corpus.write_text(text, encoding='utf-8')
    split = split_tokens(text)
    counts = Counter(split['train'])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = {'category': 'independent public-text finite paired CPU SGNS training',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_files_sha256': {name: digest_file(Path(__file__).with_name(name)) for name in
            ('train_corpus_checkpoints.py', 'benchmark_cotraining.py', 'benchmark_heldout_compression.py',
             'train_ngram_similarity.py', 'cotraining_groups.py')},
        'command': shlex.join([sys.executable, *sys.argv]),
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()},
        'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'os': platform.platform(), 'cpu': subprocess.check_output(['lscpu', '-J'], text=True),
        'device': 'cpu', 'threads': 1, 'interop_threads': 1,
        'corpus_zip_sha256': zip_hash, 'raw_prefix_sha256': raw_hash,
        'tokenized_corpus_sha256': digest_file(corpus), 'split_tokens': {k: len(v) for k, v in split.items()},
        'limitations': ['Deterministic prefix of raw Wikipedia-derived enwik8; markup/metadata tokens remain.',
            'Single additional corpus, training-only 1200-word vocabulary; OOV coverage is reported.',
            'Fixed finite SGD steps and samples, not convergence or large-model retraining evidence.',
            'Paired shared-parameter training uses identical initial means, samples and negatives.',
            'Held-out SGNS loss/discrimination is not semantic quality, perplexity or LLM memory.',
            'Only source hashes and metrics may be published here; text and checkpoints are not redistributed.',
            'Shared CPU, no exclusive latency, monetary/energy metering or GPU work.'],
        'runs': [], 'failures': []}
    write_json(cfg.output_dir / 'protocol.json', report)
    for convention in conventions:
        parent = cfg.output_dir / convention
        parent.mkdir()
        for order in orders:
            for seed in seeds:
                directory = parent / f'order-{order}-seed-{seed}'
                try:
                    checkpoint, entry = train_dynamics(split, cfg, order, seed, convention, directory)
                    tied = torch.load(directory / 'tied-reference.pt', weights_only=True, map_location='cpu')
                    hidden = compose(checkpoint['input_rows'], checkpoint['bags'], checkpoint['lengths'])
                    tied_hidden = compose(tied['input_rows'], checkpoint['bags'], checkpoint['lengths'])
                    entry['matched_final_aggregate_fidelity'] = vector_metrics(hidden.numpy(), tied_hidden.numpy())
                    entry['heldout'] = {}
                    for name in ('validation', 'test'):
                        evaluation = evaluation_set(split[name], checkpoint['words'], counts,
                                                    cfg.eval_pairs, cfg.negatives, 42 + (name == 'test'))
                        baseline = score(hidden, checkpoint['output_rows'], evaluation)
                        candidate = score(tied_hidden, tied['output_rows'], evaluation)
                        entry['heldout'][name] = {'baseline': summarize(baseline), 'tied': summarize(candidate),
                            'delta': deltas(candidate, baseline),
                            'evaluation_sha256': evaluation['sha256'], 'token_coverage': evaluation['token_coverage']}
                    write_json(directory / 'dynamics.json', entry)
                    report['runs'].append(entry)
                except Exception as exc:
                    report['failures'].append({'convention': convention, 'order': order, 'seed': seed,
                                               'error': repr(exc)})
                    write_json(cfg.output_dir / 'report.json', report)
                    raise
                write_json(cfg.output_dir / 'report.json', report)
                print(json.dumps({'completed': [convention, order, seed], 'test': entry['heldout']['test']}), flush=True)
    report['status'] = 'completed'
    write_json(cfg.output_dir / 'report.json', report)


if __name__ == '__main__':
    main()
