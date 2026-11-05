"""Full AG News supervised CPU training; no test-based model selection."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .external_cotraining import file_hash
from .text_classifier import tokenize, ngrams


CLASSES = ['World', 'Sports', 'Business', 'Sci/Tech']


def read_csv(path):
    texts, labels = [], []
    with Path(path).open(newline='', encoding='utf-8') as stream:
        for row in csv.reader(stream):
            if len(row) != 3 or row[0] not in ('1', '2', '3', '4'):
                raise ValueError('AG News requires label/title/description columns')
            texts.append(row[1] + ' ' + row[2])
            labels.append(int(row[0]) - 1)
    return texts, np.asarray(labels, dtype=np.int64)


def split_training(labels):
    rng = np.random.default_rng(42)
    validation = np.concatenate([rng.permutation(np.flatnonzero(labels == k))[:3000] for k in range(4)])
    validation = np.sort(validation)
    return np.setdiff1d(np.arange(len(labels)), validation), validation


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare(data, output):
    texts, labels = read_csv(data / 'train.csv')
    if len(texts) != 120000 or not np.array_equal(np.bincount(labels), [30000]*4):
        raise ValueError('the complete official balanced training split is required')
    train, validation = split_training(labels)
    parsed = [tokenize(text)[0] for text in texts]
    if any(not words for words in parsed):
        raise ValueError('empty training document after preprocessing')
    counts = Counter(w for i in train for w in parsed[i])
    tokens = sorted(counts)
    whole = sorted(w for w, count in counts.items() if count >= 2)
    whole_ids = {w: i for i, w in enumerate(whole)}
    grams = sorted({g for word in tokens for g in ngrams(word, 8)})
    gram_ids = {g: len(whole) + i for i, g in enumerate(grams)}
    lexicon = {'tokens': tokens, 'whole_words': whole, 'whole_word_rows': len(whole),
               'grams': grams, 'max_order': 8, 'max_tokens': 128}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'lexicon.json', lexicon)
    token_ids = {w: i for i, w in enumerate(tokens)}
    word_bags = [[whole_ids[word]] if word in whole_ids else [] for word in tokens]
    for word, bag in zip(tokens, word_bags):
        bag.extend(gram_ids[g] for g in ngrams(word, 8))
    offsets = np.r_[0, np.cumsum([len(bag) for bag in word_bags], dtype=np.uint64)].astype('<u8')
    np.concatenate(word_bags).astype('<u8').tofile(output / 'word-ids.u64')
    offsets.tofile(output / 'word-offsets.u64')
    documents = [[token_ids[w] for w in parsed[i]] for i in train]
    np.savez(output / 'split.npz', train=train, validation=validation)
    report = {'category': 'AG News real supervised CPU news routing', 'source_urls': {
        name: 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/555590db4219b1243abb1918effd6a7425a2d75f/data/ag_news_csv/'+name
        for name in ('train.csv', 'test.csv', 'readme.txt')},
        'train_csv_sha256': file_hash(data / 'train.csv'), 'test_csv_sha256': file_hash(data / 'test.csv'),
        'research_use_notice_sha256': file_hash(data / 'readme.txt'),
        'train_documents': len(train), 'validation_documents': len(validation), 'official_train_documents': len(texts),
        'vocabulary_tokens': len(tokens), 'whole_word_rows': len(whole), 'ngram_rows': len(grams),
        'rows': len(whole) + len(grams), 'dim': 64,
        'split_sha256': hashlib.sha256(train.astype('<u8').tobytes()+validation.astype('<u8').tobytes()).hexdigest(),
        'registry_sha256': {name: file_hash(output/name) for name in ('lexicon.json','word-ids.u64','word-offsets.u64')}}
    write_json(output / 'data.json', report)
    return texts, labels, train, validation, lexicon, word_bags, documents, report


def torch_word_features(embedding, bags, torch, chunk=1024):
    result = []
    with torch.no_grad():
        for start in range(0, len(bags), chunk):
            batch = bags[start:start+chunk]
            offsets = np.r_[0, np.cumsum([len(b) for b in batch])]
            flat = np.concatenate(batch) if offsets[-1] else np.empty(0, np.int64)
            values = embedding(torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long))
            result.append(values / torch.tensor(np.maximum(np.diff(offsets), 1), dtype=torch.float32)[:, None])
    return torch.cat(result)


def validation_features(texts, lexicon, word_bags):
    parsed = [tokenize(text)[0] for text in texts]
    unique = list(dict.fromkeys(w for words in parsed for w in words))
    token_ids = {w: i for i, w in enumerate(lexicon['tokens'])}
    gram_ids = {g: lexicon['whole_word_rows']+i for i, g in enumerate(lexicon['grams'])}
    lookup = {w: i for i, w in enumerate(unique)}
    bags = [word_bags[token_ids[w]] if w in token_ids else [gram_ids[g] for g in ngrams(w, 8) if g in gram_ids] for w in unique]
    docs = [[lookup[w] for w in words] for words in parsed]
    return bags, docs


def train_one(root, common, seed, cfg, texts, labels, train, validation, lexicon, word_bags, documents, data_report):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    output = root / f'seed-{seed}'
    output.mkdir()
    embedding = nn.EmbeddingBag(data_report['rows'], 64, mode='sum', sparse=True, include_last_offset=True)
    head = nn.Linear(64, 4)
    with torch.no_grad():
        embedding.weight.uniform_(-1/64, 1/64)
        head.weight.normal_(0, .1)
        head.bias.zero_()
    initial_sha = hashlib.sha256(embedding.weight.detach().numpy().tobytes()).hexdigest()
    optimizer = torch.optim.SGD([{'params': embedding.parameters(), 'lr': 1.}, {'params': head.parameters(), 'lr': 1.}])
    val_bags, val_docs = validation_features([texts[i] for i in validation], lexicon, word_bags)
    report = {'source_commit': cfg.source_commit, 'seed': seed, 'status': 'running',
              'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'os': platform.platform(),
              'device': 'cpu', 'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
              'data': data_report, 'initial_rows_sha256': initial_sha, 'epochs': [],
              'command_argv': sys.orig_argv, 'settings': {'epochs':8,'batch':128,'dim':64,'embedding_lr':1.,'head_lr':1.,
                 'backward':'unscaled word-bag; ordinary document mean/head derivative', 'optimizer':'sparse+dense SGD',
                 'activation':cfg.activation},
              'source_files_sha256': {name:file_hash(Path(__file__).with_name(name)) for name in ('train_news_classifier.py','text_classifier.py')}}
    best = float('inf')
    started = time.perf_counter()
    for epoch in range(8):
        before = time.perf_counter()
        for group in optimizer.param_groups:
            group['lr'] = 1. * (1 - epoch/8)
        permutation = rng.permutation(len(documents))
        losses = []
        for start in range(0, len(documents), 128):
            chosen = permutation[start:start+128]
            docs = [documents[int(i)] for i in chosen]
            doc_offsets = np.r_[0, np.cumsum([len(d) for d in docs])]
            flat_words = np.concatenate(docs)
            unique, inverse = np.unique(flat_words, return_inverse=True)
            bags = [word_bags[int(i)] for i in unique]
            bag_offsets = np.r_[0, np.cumsum([len(b) for b in bags])]
            ids = torch.tensor(np.concatenate(bags), dtype=torch.long)
            summed = embedding(ids, torch.tensor(bag_offsets, dtype=torch.long))
            mean = summed / torch.tensor(np.diff(bag_offsets), dtype=torch.float32)[:, None]
            features = summed + (mean-summed).detach()
            hidden = F.embedding_bag(torch.tensor(inverse, dtype=torch.long), features,
                                     torch.tensor(doc_offsets, dtype=torch.long), mode='mean', include_last_offset=True)
            loss = F.cross_entropy(head(hidden.relu() if cfg.activation=='relu' else hidden),
                                   torch.tensor(labels[train[chosen]], dtype=torch.long))
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            vectors = torch_word_features(embedding, val_bags, torch)
            val_logits = []
            for start in range(0, len(val_docs), 512):
                docs = val_docs[start:start+512]
                offsets = np.r_[0, np.cumsum([len(d) for d in docs])]
                hidden = F.embedding_bag(torch.tensor(np.concatenate(docs), dtype=torch.long), vectors,
                                        torch.tensor(offsets, dtype=torch.long), mode='mean', include_last_offset=True)
                val_logits.append(head(hidden.relu() if cfg.activation=='relu' else hidden))
            logits = torch.cat(val_logits)
            ce = float(F.cross_entropy(logits, torch.tensor(labels[validation])))
            accuracy = float((logits.argmax(1).numpy() == labels[validation]).mean())
        item = {'epoch':epoch+1, 'train_loss':float(np.mean(losses)), 'validation_ce':ce,
                'validation_accuracy':accuracy, 'seconds':time.perf_counter()-before}
        report['epochs'].append(item)
        if ce < best:
            best = ce
            embedding.weight.detach().numpy().astype('<f4').tofile(output / 'source.f32')
            np.savez(output / 'head.npz', weight=head.weight.detach().numpy().astype('<f4'), bias=head.bias.detach().numpy().astype('<f4'))
            report['selected_epoch'] = epoch+1
        write_json(output / 'training.json', report)
        print(json.dumps({'seed':seed, **item}), flush=True)
    report.update(status='passed',seconds=time.perf_counter()-started,
                  source_sha256=file_hash(output/'source.f32'),head_sha256=file_hash(output/'head.npz'))
    if report['source_sha256'] == initial_sha:
        raise AssertionError('source embeddings were not trained')
    write_json(output / 'training.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--seeds',default='3,7,11')
    p.add_argument('--activation',choices=('identity','relu'),default='identity')
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU only: CUDA_VISIBLE_DEVICES must be empty')
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg.output.mkdir(parents=True,exist_ok=False)
    common = cfg.output / 'common'
    arguments = prepare(cfg.data, common)
    texts, labels, train, validation, lexicon, word_bags, documents, report = arguments
    for seed in map(int,cfg.seeds.split(',')):
        train_one(cfg.output, common, seed, cfg, texts, labels, train, validation, lexicon, word_bags, documents, report)


if __name__ == '__main__':
    main()
