"""Complete public-corpus GPU word-n-gram classifiers with training-only keys.

This is a standalone n-gram classification task, not Engram pretraining. Full
official train/test data are used; calibration comes only from official train.
Large tables contain actual updated rows, never tiled or synthetic embeddings.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from .tree_node_quality import array_digest, digest, environment, load, write

DATASETS = {'ag_news': (120000, 7600, 4), 'dbpedia_14': (560000, 70000, 14)}


def word_tokens(text):
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def document_keys(words, order):
    """Positive training word IDs; zero suffix distinguishes n-gram orders."""
    words = np.asarray(words, dtype='<u4')
    parts = []
    for n in range(1, min(order, len(words)) + 1):
        part = np.zeros((len(words) - n + 1, order), dtype='<u4')
        for j in range(n):
            part[:, j] = words[j:j + len(part)]
        parts.append(part)
    return np.concatenate(parts) if parts else np.empty((0, order), dtype='<u4')


def key_view(keys):
    keys = np.ascontiguousarray(keys, dtype='<u4')
    return keys.view(np.dtype('V' + str(4 * keys.shape[1]))).reshape(-1)


def exact_ids(keys, registry):
    query = key_view(keys)
    positions = np.searchsorted(registry, query)
    valid = positions < len(registry)
    candidates = np.minimum(positions, len(registry) - 1)
    valid &= registry[candidates] == query
    return np.where(valid, positions + 1, 0).astype('<i8')


def hash_ids(keys, buckets):
    """Fixed uint64 FNV-1a over all canonical fields, then one modulo."""
    values = np.full(len(keys), np.uint64(14695981039346656037))
    for field in np.asarray(keys, dtype='<u4').T:
        for shift in (0, 8, 16, 24):
            values ^= ((field >> shift) & 255).astype(np.uint64)
            values *= np.uint64(1099511628211)
    return (values % np.uint64(buckets) + 1).astype('<i8')


def prepare(args):
    from datasets import load_dataset
    expected_train, expected_test, classes = DATASETS[args.dataset]
    output = args.output
    if (output / 'prepared.json').exists():
        report = load(output / 'prepared.json')
        for name, sha in report['payload_sha256'].items():
            if digest(output / name) != sha:
                raise ValueError('prepared corpus changed: ' + name)
        return
    if output.exists():
        raise ValueError('partial preparation exists; preserve it and use a fresh output')
    output.mkdir(parents=True)
    started = time.perf_counter()
    dataset = load_dataset('fancyzhx/' + args.dataset, cache_dir=str(args.cache))
    if len(dataset['train']) != expected_train or len(dataset['test']) != expected_test:
        raise ValueError('complete official corpus required')
    label = np.asarray(dataset['train']['label'], dtype='<i8')
    rng = np.random.default_rng(42)
    validation = np.sort(np.concatenate([rng.permutation(np.flatnonzero(label == k))[:expected_train // classes // 10]
                                         for k in range(classes)]))
    training = np.setdiff1d(np.arange(len(label)), validation)
    def text(row):
        return row['text'] if args.dataset == 'ag_news' else row['title'] + ' ' + row['content']
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        parsed = list(pool.map(lambda row: word_tokens(text(row)), dataset['train']))
    vocabulary = sorted({word for i in training for word in parsed[i]})
    lookup = {word: i + 1 for i, word in enumerate(vocabulary)}
    write(output / 'word-vocabulary.json', vocabulary)
    def encoded(document):
        # UINT32_MAX is an OOV word, distinct from zero suffix padding. Using
        # zero here would silently turn a known-word/OOV bigram into a unigram.
        return document_keys([lookup.get(w, (1 << 32) - 1) for w in document], args.order)
    lengths = np.array([sum(max(0, len(parsed[i]) - n + 1) for n in range(1, args.order + 1))
                        for i in training], dtype='<i8')
    offsets = np.r_[0, np.cumsum(lengths)]
    keys = np.lib.format.open_memmap(output / 'train-keys.npy', mode='w+', dtype='<u4',
                                    shape=(int(offsets[-1]), args.order))
    for j, i in enumerate(training):
        keys[offsets[j]:offsets[j + 1]] = encoded(parsed[i])
    keys.flush()
    print(json.dumps({'phase': 'unique-training-keys', 'occurrences': len(keys), 'seconds': time.perf_counter() - started}), flush=True)
    registry, inverse, counts = np.unique(key_view(keys), return_inverse=True, return_counts=True)
    np.save(output / 'registry.npy', registry)
    np.save(output / 'counts.npy', counts.astype('<i8'))
    np.save(output / 'train-ids.npy', inverse.astype('<i8') + 1)
    np.save(output / 'train-offsets.npy', offsets)
    np.save(output / 'train-labels.npy', label[training])
    np.savez(output / 'split.npz', train=training, calibration=validation)
    del inverse
    for split, docs, labels in (
            ('calibration', [parsed[i] for i in validation], label[validation]),
            ('test', [word_tokens(text(row)) for row in dataset['test']], np.asarray(dataset['test']['label'], dtype='<i8'))):
        lengths = np.array([sum(max(0, len(doc) - n + 1) for n in range(1, args.order + 1)) for doc in docs], dtype='<i8')
        ptr = np.r_[0, np.cumsum(lengths)]
        split_keys = np.lib.format.open_memmap(output / (split + '-keys.npy'), mode='w+', dtype='<u4',
                                              shape=(int(ptr[-1]), args.order))
        ids = np.lib.format.open_memmap(output / (split + '-ids.npy'), mode='w+', dtype='<i8', shape=(int(ptr[-1]),))
        for i, doc in enumerate(docs):
            k = encoded(doc)
            split_keys[ptr[i]:ptr[i + 1]] = k
            ids[ptr[i]:ptr[i + 1]] = exact_ids(k, registry)
        ids.flush(); split_keys.flush()
        np.save(output / (split + '-offsets.npy'), ptr)
        np.save(output / (split + '-labels.npy'), labels)
    files = sorted(p.name for p in output.iterdir() if p.suffix in ('.npy', '.npz', '.json'))
    report = {'status': 'complete', 'dataset': args.dataset, 'classes': classes,
        'official_train_documents': expected_train, 'official_test_documents': expected_test,
        'training_documents': len(training), 'calibration_documents': len(validation),
        'ngram_order': args.order, 'rows': len(registry) + 1, 'training_occurrences': len(keys),
        'word_vocabulary': len(vocabulary), 'training_fingerprint': dataset['train']._fingerprint,
        'test_fingerprint': dataset['test']._fingerprint, 'environment': environment(),
        'source_sha256': digest(__file__), 'seconds': time.perf_counter() - started,
        'payload_sha256': {name: digest(output / name) for name in files},
        'contract': 'Training-only canonical word n-grams. Complete official documents, no truncation. Row 0 is zero for unseen keys.',
        'limitations': ['Standalone public classification, not general language modeling or Engram pretraining.',
                         'Train/calibration is a fixed stratified 90/10 split of official training data.',
                         'Text and trained weights are private artifacts; only aggregates/hashes are published.']}
    write(output / 'prepared.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'rows', 'training_documents', 'seconds')}), flush=True)


def batch_csr(ids, ptr, documents, mapping=None):
    parts = [ids[ptr[i]:ptr[i + 1]] for i in documents]
    flat = np.concatenate(parts).astype(np.int64, copy=False)
    if mapping is not None:
        flat = mapping[flat]
    offsets = np.r_[0, np.cumsum([len(x) for x in parts])].astype(np.int64)
    return flat, offsets


def make_hash_map(data, buckets, oov):
    registry = np.load(data / 'registry.npy', mmap_mode='r')
    order = load(data / 'prepared.json')['ngram_order']
    mapping = np.zeros(len(registry) + 1, dtype='<i8')
    for begin in range(0, len(registry), 65536):
        keys = registry[begin:begin + 65536].view('<u4').reshape(-1, order)
        mapping[begin + 1:begin + 1 + len(keys)] = hash_ids(keys, buckets)
    # Primary matched-domain comparison: unseen training keys use zero for both.
    if oov != 'zero':
        raise ValueError('only the preregistered matched zero-OOV contract is supported')
    return mapping


def train(args):
    import torch
    from torch import nn
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = load(args.data / 'prepared.json')
    if args.output.exists():
        raise ValueError('fresh training output required')
    args.output.mkdir(parents=True)
    device = torch.device('cuda')
    rows = metadata['rows'] if args.addressing == 'exact' else args.buckets + 1
    if args.addressing == 'hash-matched':
        rows = metadata['rows']; args.buckets = rows - 1
    mapping = None if args.addressing == 'exact' else make_hash_map(args.data, rows - 1, 'zero')
    ids = np.load(args.data / 'train-ids.npy', mmap_mode='r')
    ptr = np.load(args.data / 'train-offsets.npy', mmap_mode='r')
    labels = np.load(args.data / 'train-labels.npy')
    embedding = nn.EmbeddingBag(rows, args.dim, mode='mean', sparse=True, include_last_offset=True,
                                 padding_idx=0, device=device)
    classifier = nn.Linear(args.dim, metadata['classes'], device=device)
    with torch.no_grad():
        if args.initialization == 'zero':
            embedding.weight.zero_()
        else:
            generator = torch.Generator(device=device).manual_seed(args.seed + 100003)
            nn.init.uniform_(embedding.weight, -1 / args.dim, 1 / args.dim, generator=generator)
            embedding.weight[0].zero_()
        # Separate RNG makes the classifier identical even when table sizes
        # differ; default Embedding construction consumes size-dependent draws.
        classifier_generator = torch.Generator(device=device).manual_seed(args.seed)
        nn.init.normal_(classifier.weight, std=.1, generator=classifier_generator)
        classifier.bias.zero_()
    initial = None if args.initialization == 'zero' else embedding.weight.detach().clone()
    optimizer = torch.optim.SGD([*embedding.parameters(), *classifier.parameters()], lr=args.lr)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    report = {'status': 'training', 'environment': {**environment(), 'device': 'cuda', 'gpu': torch.cuda.get_device_name()},
        'data_sha256': digest(args.data / 'prepared.json'), 'source_sha256': digest(__file__),
        'seed': args.seed, 'addressing': args.addressing, 'rows': rows, 'dim': args.dim,
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'table_bytes': rows * args.dim * 4, 'epochs': [], 'cuda_selector': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'mapping_sha256': None if mapping is None else array_digest(mapping),
        'oov_contract': 'Unseen canonical training keys map to zero in both full and hash controls.',
        'initialization': args.initialization, 'classifier_initial_sha256': array_digest(classifier.weight.detach().cpu().numpy()),
        'initialization_contract': 'Identical independently seeded classifier for every table size. Uniform rows are real training initializations, never benchmark stand-ins.'}
    write(args.output / 'training.json', report)
    if mapping is not None:
        np.save(args.output / 'canonical-to-bucket.npy', mapping)
    batches = (len(labels) + args.batch_size - 1) // args.batch_size
    for epoch in range(args.epochs):
        permutation = rng.permutation(len(labels)); losses = []
        for step, begin in enumerate(range(0, len(labels), args.batch_size)):
            docs = permutation[begin:begin + args.batch_size]
            flat, offsets = batch_csr(ids, ptr, docs, mapping)
            x = torch.from_numpy(flat).to(device); off = torch.from_numpy(offsets).to(device)
            target = torch.from_numpy(labels[docs]).to(device)
            optimizer.param_groups[0]['lr'] = args.lr * (1 - (epoch * batches + step) / (args.epochs * batches))
            loss = nn.functional.cross_entropy(classifier(embedding(x, off)), target)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(json.dumps({'epoch': epoch, 'batch': step, 'loss': losses[-1], 'seconds': time.perf_counter() - started}), flush=True)
        report['epochs'].append({'epoch': epoch + 1, 'mean_loss': float(np.mean(losses)), 'documents': len(labels)})
        write(args.output / 'training.json', report)
    table = np.lib.format.open_memmap(args.output / 'rows.npy', mode='w+', dtype='<f4', shape=(rows, args.dim))
    changed = 0; nonzero = 0
    for begin in range(0, rows, 65536):
        part = embedding.weight[begin:begin + 65536].detach().cpu().numpy()
        if not np.isfinite(part).all():
            raise ValueError('nonfinite trained table')
        table[begin:begin + len(part)] = part
        nonzero += int(np.any(part != 0, axis=1).sum())
        previous = 0 if initial is None else initial[begin:begin + len(part)].cpu().numpy()
        changed += int(np.any(part != previous, axis=1).sum())
    table.flush()
    np.savez(args.output / 'classifier.npz', weight=classifier.weight.detach().cpu().numpy(), bias=classifier.bias.detach().cpu().numpy())
    if args.addressing == 'exact' and changed != rows - 1:
        report.update(status='incomplete-row-coverage', changed_rows=changed)
        write(args.output / 'training.json', report)
        raise ValueError('every claimed full-memory row must have changed from its declared initialization')
    calibration_ids = np.load(args.data / 'calibration-ids.npy', mmap_mode='r')
    calibration_ptr = np.load(args.data / 'calibration-offsets.npy', mmap_mode='r')
    calibration_labels = np.load(args.data / 'calibration-labels.npy')
    predictions = []
    with torch.no_grad():
        for begin in range(0, len(calibration_labels), 512):
            docs = np.arange(begin, min(begin + 512, len(calibration_labels)))
            flat, offsets = batch_csr(calibration_ids, calibration_ptr, docs, mapping)
            logits = classifier(embedding(torch.from_numpy(flat).to(device), torch.from_numpy(offsets).to(device)))
            predictions.append(logits.cpu().numpy())
    logits = np.concatenate(predictions)
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    nll = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(logits)), calibration_labels]
    np.savez(args.output / 'calibration.npz', logits=logits, labels=calibration_labels, nll=nll)
    report['calibration'] = {'documents': len(logits), 'accuracy': float(np.mean(logits.argmax(1) == calibration_labels)),
        'mean_nll': float(np.mean(nll)), 'sha256': digest(args.output / 'calibration.npz')}
    report.update(status='complete', changed_rows=changed, zero_rows=rows - nonzero,
        seconds=time.perf_counter() - started, rows_sha256=digest(args.output / 'rows.npy'),
        classifier_sha256=digest(args.output / 'classifier.npz'),
        cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved())
    write(args.output / 'training.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'changed_rows', 'seconds')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--order', type=int, choices=(2, 3, 4), default=4)
    p.add_argument('--workers', type=int, default=8)
    p = sub.add_parser('train')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--addressing', choices=('exact', 'hash-matched', 'hash-budget'), default='exact')
    p.add_argument('--buckets', type=int, default=1 << 20)
    p.add_argument('--seed', type=int, choices=(3, 7, 11), default=3)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=10.)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--initialization', choices=('zero', 'uniform'), default='zero')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args)
    else:
        if min(args.dim, args.epochs, args.batch_size, args.threads, args.buckets) < 1 or args.lr <= 0:
            parser.error('positive training settings required')
        train(args)


if __name__ == '__main__':
    main()
