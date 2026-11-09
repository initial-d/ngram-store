"""CPU factorial with shared schedules and exact replay of saved SGNS rows.

Train only; quality/structural analysis is a separate frozen-checkpoint phase.
Initialization changes are controls, not synthetic-vector performance evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def environment():
    import torch
    return {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_sha256': digest(__file__), 'command': shlex.join([sys.executable, *sys.argv]),
            'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
            'os': platform.platform(), 'cpu': platform.processor(), 'device': 'cpu',
            'gpu': 'unused', 'threads': 1, 'interop_threads': 1,
            'loadavg': list(os.getloadavg())}


def prepare_schedules(cfg, order, seed, out):
    import torch
    from torch import nn
    from .train_ngram_similarity import prepare
    from .benchmark_heldout_compression import split_tokens
    original = torch.load(cfg.training_root / 'legacy' / f'order-{order}-seed-{seed}' / 'reference.pt',
                          weights_only=True, map_location='cpu')
    settings = json.loads((cfg.training_root / 'report.json').read_text())['settings']
    split = split_tokens((cfg.training_root / 'corpus.txt').read_text())
    words, grams, _, pairs, bags, lengths, counts = prepare(' '.join(split['train']), order,
        settings['vocab_limit'], settings['min_count'])
    assert words == original['words'] and grams == original['grams']
    assert torch.equal(bags, original['bags']) and torch.equal(lengths, original['lengths'])
    torch.manual_seed(seed)
    model = nn.EmbeddingBag(len(original['initial_rows']), settings['dim'], mode='sum', sparse=True, padding_idx=0)
    nn.Embedding(len(words), settings['dim'], sparse=True)
    with torch.no_grad():
        model.weight.uniform_(-1 / settings['dim'], 1 / settings['dim'])
        model.weight[0].zero_()
    assert torch.equal(model.weight, original['initial_rows']), 'original initialization cannot be reproduced'
    generator = torch.Generator().set_state(torch.get_rng_state())
    probability = torch.tensor([counts[w] ** .5 for w in words], dtype=torch.float32)
    probability /= probability.sum()
    rng = np.random.default_rng(seed)
    maximum = max(cfg.steps)
    sampled = np.lib.format.open_memmap(out / 'pairs.npy', mode='w+', dtype='<i8',
        shape=(maximum, settings['batch_size'], 2))
    negatives = np.lib.format.open_memmap(out / 'negatives.npy', mode='w+', dtype='<i8',
        shape=(maximum, settings['batch_size'], settings['negatives']))
    start = time.perf_counter()
    for step in range(maximum):
        sampled[step] = pairs[rng.integers(len(pairs), size=settings['batch_size'])]
        negatives[step] = torch.multinomial(probability, settings['batch_size'] * settings['negatives'],
            replacement=True, generator=generator).reshape(settings['batch_size'], settings['negatives']).numpy()
    sampled.flush(); negatives.flush()
    del sampled, negatives
    report = {'order': order, 'seed': seed, 'maximum_steps': maximum,
        'pair_schedule_sha256': digest(out / 'pairs.npy'),
        'negative_schedule_sha256': digest(out / 'negatives.npy'),
        'initial_row_sha256': array_digest(original['initial_rows'].numpy()),
        'training_pool_sha256': array_digest(pairs.astype('<i8')),
        'source_training_report_sha256': digest(cfg.training_root / 'report.json'),
        'corpus_sha256': digest(cfg.training_root / 'corpus.txt'),
        'preparation_seconds': time.perf_counter() - start, 'environment': environment()}
    write(out / 'schedule.json', report)


def trajectory(cfg, order, seed, convention, scale, schedules, out):
    import torch
    from torch.nn import functional as F
    source_path = cfg.training_root / convention / f'order-{order}-seed-{seed}' / 'reference.pt'
    old = torch.load(source_path, map_location='cpu', weights_only=True)
    settings = json.loads((cfg.training_root / 'report.json').read_text())['settings']
    initial = old['initial_rows'].clone()
    offset = 1 + len(old['words'])
    initial[offset:] *= scale
    weight = torch.nn.Parameter(initial.clone())
    output = torch.nn.Parameter(torch.zeros_like(old['output_rows']))
    optimizer = torch.optim.SGD([weight, output], lr=settings['lr'])
    sampled = np.load(schedules / 'pairs.npy', mmap_mode='r')
    negatives = np.load(schedules / 'negatives.npy', mmap_mode='r')
    bags, lengths = old['bags'], old['lengths']
    events = np.zeros(len(old['words']), dtype='<i8')
    report = {'status': 'running', 'order': order, 'seed': seed, 'convention': convention,
        'ngram_initialization_scale': scale, 'word_initialization_unchanged': True,
        'source_checkpoint_sha256': digest(source_path),
        'schedule_sha256': digest(schedules / 'schedule.json'),
        'environment': environment(), 'settings': settings, 'checkpoints': [], 'loss_samples': []}
    write(out / 'report.json', report)
    started = time.perf_counter()
    for step in range(max(cfg.steps)):
        pair = np.array(sampled[step])
        center = torch.from_numpy(pair[:, 0].copy())
        target = torch.from_numpy(pair[:, 1].copy())
        negative = torch.from_numpy(np.array(negatives[step]))
        np.add.at(events, pair[:, 0], 1)
        summed = F.embedding_bag(bags[center], weight, mode='sum', sparse=True, padding_idx=0)
        averaged = summed / lengths[center, None]
        hidden = summed + (averaged - summed).detach() if convention == 'legacy' else averaged
        positive = (hidden * F.embedding(target, output, sparse=True)).sum(1)
        logits = (hidden[:, None] * F.embedding(negative, output, sparse=True)).sum(2)
        loss = (-F.logsigmoid(positive) - F.logsigmoid(-logits).sum(1)).sum()
        if not torch.isfinite(loss):
            report.update(status='failed', failure=f'nonfinite loss at step {step}')
            write(out / 'report.json', report)
            raise ValueError(report['failure'])
        optimizer.zero_grad(set_to_none=True)
        loss.backward(); optimizer.step()
        if step % 1000 == 0 or step + 1 in cfg.steps:
            report['loss_samples'].append({'step': step + 1, 'loss_per_pair': float(loss.detach()) / len(center),
                                          'elapsed_seconds': time.perf_counter() - started})
        if step + 1 in cfg.steps:
            duration = step + 1
            exact_input = torch.equal(weight.detach(), old['input_rows']) if duration == 2000 and scale == 1 else None
            exact_output = torch.equal(output.detach(), old['output_rows']) if duration == 2000 and scale == 1 else None
            if duration == 2000 and scale == 1 and not (exact_input and exact_output):
                report.update(status='failed', failure='2000-step exact source replay failed',
                    input_max_difference=float((weight.detach() - old['input_rows']).abs().max()),
                    output_max_difference=float((output.detach() - old['output_rows']).abs().max()))
                write(out / 'report.json', report)
                raise ValueError(report['failure'])
            directory = out / f'step-{duration}'
            directory.mkdir()
            checkpoint = {'input_rows': weight.detach().clone(), 'output_rows': output.detach().clone(),
                'initial_rows': initial, 'words': old['words'], 'grams': old['grams'],
                'bags': bags, 'lengths': lengths, 'order': order, 'seed': seed}
            torch.save(checkpoint, directory / 'reference.pt')
            np.save(directory / 'center-events.npy', events)
            point = {'steps': duration, 'checkpoint_sha256': digest(directory / 'reference.pt'),
                'input_row_sha256': array_digest(checkpoint['input_rows'].numpy()),
                'output_row_sha256': array_digest(checkpoint['output_rows'].numpy()),
                'sampled_center_events_sha256': array_digest(events.astype('<u8')),
                'sampled_pairs': int(events.sum()), 'original_input_bytes_replayed': exact_input,
                'original_output_bytes_replayed': exact_output,
                'training_seconds': time.perf_counter() - started}
            report['checkpoints'].append(point)
            write(out / 'report.json', report)
            print(json.dumps({'completed': out.name, **point}), flush=True)
    report.update(status='complete', peak_rss_bytes=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss * 1024)
    write(out / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', default='2000,8000,32000')
    parser.add_argument('--jobs', type=int, default=9)
    parser.add_argument('--worker', nargs=2, type=int)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU-only execution requires empty CUDA_VISIBLE_DEVICES')
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    args.steps = sorted(set(map(int, args.steps.split(','))))
    args.output = args.output.resolve()
    if args.worker:
        order, seed = args.worker
        parent = args.output / f'order-{order}-seed-{seed}'
        parent.mkdir()
        schedules = parent / 'schedule'; schedules.mkdir()
        prepare_schedules(args, order, seed, schedules)
        for convention in ('legacy', 'average'):
            for scale in (1., .1):
                out = parent / f'{convention}-init-{scale:g}'; out.mkdir()
                trajectory(args, order, seed, convention, scale, schedules, out)
        return
    if args.jobs < 1 or args.steps != [2000, 8000, 32000]:
        parser.error('predeclared positive jobs and complete duration grid required')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'environment': environment(), 'steps': args.steps,
        'jobs': args.jobs, 'training_root_report_sha256': digest(args.training_root / 'report.json'),
        'trajectories': [], 'failures': []}
    write(args.output / 'report.json', report)
    def execute(order, seed):
        log_path = args.output / f'order-{order}-seed-{seed}.log'
        command = [sys.executable, '-m', 'tools.train_tree_node_mechanism', '--training-root', str(args.training_root),
            '--output', str(args.output), '--steps', ','.join(map(str, args.steps)), '--worker', str(order), str(seed)]
        with log_path.open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'order {order} seed {seed}: worker exit {result.returncode}; preserve {log_path.name}')
        return [{'name': f'order-{order}-seed-{seed}/{convention}-init-{scale:g}',
                 'report_sha256': digest(args.output / f'order-{order}-seed-{seed}' / f'{convention}-init-{scale:g}' / 'report.json')}
                for convention in ('legacy', 'average') for scale in (1., .1)]
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        pending = {executor.submit(execute, order, seed): (order, seed)
                   for order in (4, 8, 12) for seed in (3, 7, 11)}
        for future in as_completed(pending):
            try:
                report['trajectories'].extend(future.result())
            except Exception as error:
                report['failures'].append({'pair': pending[future], 'error': repr(error)})
            write(args.output / 'report.json', report)
            print(json.dumps({'complete_trajectories': len(report['trajectories']), 'failures': report['failures']}), flush=True)
    report['status'] = 'complete' if not report['failures'] else 'failed'
    write(args.output / 'report.json', report)
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
