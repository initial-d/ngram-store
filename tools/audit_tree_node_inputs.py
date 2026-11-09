"""Reconstruct every evaluation input and full known-word source cache on CPU."""
import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np

from .tree_node_quality import (SPLITS, array_digest, context_evaluation, digest,
    environment, load, plan_for, string_ngrams, write)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frontier', 'training-root', 'corpus-zip', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU only; empty CUDA_VISIBLE_DEVICES required')
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from .benchmark_faiss_cotraining import evaluate, objective
    from .benchmark_heldout_compression import split_tokens
    from .train_corpus_checkpoints import corpus_prefix
    from .structural_native import StructuralIndex
    prepared = load(args.frontier / 'prepare.json')
    original = load(args.training_root / 'report.json')
    assert digest(args.corpus_zip) == prepared['corpus_zip_sha256'] == original['corpus_zip_sha256']
    text, _ = corpus_prefix(args.corpus_zip, 'enwik8', 300000, 4 * 1024**2)
    assert ' '.join(text.split()[:200000]) == (args.training_root / 'corpus.txt').read_text()
    splits = split_tokens((args.training_root / 'corpus.txt').read_text())
    splits['confirmation'] = text.split()[200000:300000]
    assert array_digest(np.frombuffer(' '.join(splits['confirmation']).encode(), dtype='u1')) == prepared['confirmation_tokens_sha256']
    counts = Counter(splits['train'])
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'environment': environment(), 'phase_source_sha256': digest(__file__),
        'source_training_report_sha256': digest(args.training_root / 'report.json'),
        'source_prepare_report_sha256': digest(args.frontier / 'prepare.json'), 'models': []}
    for model in prepared['models']:
        common = args.frontier / model['name'] / 'common'
        metadata = load(common / 'metadata.json')
        checkpoint = args.training_root / metadata['convention'] / f'order-{metadata["order"]}-seed-{metadata["seed"]}' / 'reference.pt'
        assert digest(checkpoint) == metadata['source_checkpoint_sha256']
        source = torch.load(checkpoint, map_location='cpu', weights_only=True)
        words, grams = source['words'], source['grams']
        assert load(common / 'word-keys.json') == words and load(common / 'gram-keys.json') == grams
        rows = source['input_rows'].numpy()[len(words) + 1:]
        word_rows = source['input_rows'].numpy()[1:len(words) + 1]
        output_rows = source['output_rows'].numpy()
        for name, expected in (('source-rows.npy', rows), ('word-rows.npy', word_rows), ('output-rows.npy', output_rows)):
            assert np.load(common / name).tobytes() == expected.tobytes()
        for name, recorded in metadata['common_serving_files'].items():
            assert digest(common / name) == recorded['sha256']
        assert array_digest(rows) == metadata['source_row_sha256']
        keys = words + sorted(set(w for s in SPLITS for w in splits[s]) - set(words))
        lexical_ids = {word: i for i, word in enumerate(keys)}
        gram_ids = {gram: i for i, gram in enumerate(grams)}
        bags = [[gram_ids[g] for g in string_ngrams(word, metadata['order']) if g in gram_ids] for word in keys]
        ids = np.asarray([i if i < len(words) else -1 for i in range(len(keys))], dtype='<i8')
        assert bags == load(common / 'lexical-bags.json')
        assert ids.tobytes() == np.load(common / 'lexical-word-ids.npy').tobytes()
        assert np.array_equal([bool(b) for b in bags], np.load(common / 'lexical-nonempty.npy'))
        occurrence = np.zeros(len(rows), dtype=np.int64)
        for word, bag in zip(words, bags[:len(words)]):
            np.add.at(occurrence, bag, counts[word])
        assert occurrence.tobytes() == np.load(common / 'source-occurrence-counts.npy').tobytes()
        fit_ids = np.random.default_rng(42).choice(len(rows), min(16384, len(rows)), replace=False)
        assert fit_ids.tobytes() == np.load(common / 'fit-original-ids.npy').tobytes()
        assert array_digest(fit_ids.astype('<u8')) == metadata['fit_original_ids_sha256']
        class Oracle:
            mapping = None
            def __init__(self):
                self.metadata = {'rows': len(rows), 'dim': rows.shape[1]}
            def rows(self, found):
                return rows[found]
        reader = Oracle()
        hidden = evaluate(reader, plan_for(bags, ids, reader), word_rows, batch_bags=64)
        assert hidden.tobytes() == np.load(common / 'lexical-source-hidden.npy').tobytes()
        assert hidden[:len(words)].tobytes() == np.load(common / 'cached-word-rows.npy').tobytes()
        job = load(model['jobs'][0])
        structural_library = os.environ.get('MEFT_QUERY_STRUCTURAL_LIBRARY', job['structural_library'])
        for kind, name in (('datrie', 'keys.datrie'), ('hash', 'keys.flat')):
            with StructuralIndex(common / name, structural_library) as index:
                assert index.lookup(grams, index=kind) == list(range(len(grams)))
        evaluations = {}
        for i, split in enumerate(SPLITS):
            expected, info = context_evaluation(splits[split], words, lexical_ids, counts, 42 + i)
            actual = dict(np.load(common / (split + '-evaluation.npz')))
            assert all(expected[k].tobytes() == actual[k].tobytes() for k in expected)
            assert info == metadata['evaluations'][split]
            expected_scores = objective(hidden, output_rows, expected)
            actual_scores = dict(np.load(common / (split + '-source-scores.npz')))
            assert all(expected_scores[k].tobytes() == actual_scores[k].tobytes() for k in expected_scores)
            evaluations[split] = {'pairs': info['pairs'], 'evaluation_sha256': info['sha256'],
                'source_scores_sha256': digest(common / (split + '-source-scores.npz')),
                'arrays_and_source_scores_byte_exact': True}
        query = load(common / 'query-strings.json'); rng = np.random.default_rng(42)
        for name, types in (('known', sorted(set(splits['confirmation']) & set(words))),
                            ('oov', sorted(set(splits['confirmation']) - set(words)))):
            picked = [types[i] for i in sorted(rng.choice(len(types), min(512, len(types)), replace=False))]
            assert query[name] == picked
            assert hidden[[lexical_ids[w] for w in picked]].tobytes() == np.load(common / (name + '-query-reference.npy')).tobytes()
        report['models'].append({'name': model['name'], 'known_words_verified': len(words),
            'canonical_cached_words_byte_exact': True, 'structural_library_sha256': digest(structural_library),
            'both_indexes_all_ids_verified': len(grams),
            'training_occurrence_weights_verified': True, 'fit_original_ids_sha256': metadata['fit_original_ids_sha256'],
            'checkpoint_sha256': digest(checkpoint), 'metadata_sha256': digest(common / 'metadata.json'),
            'cached_word_serving_file': {'bytes': (common / 'cached-word-rows.npy').stat().st_size,
                                       'sha256': digest(common / 'cached-word-rows.npy')},
            'evaluations': evaluations, 'query_type_samples_verified': {k: len(v) for k, v in query.items()}})
        write(args.output / 'report.json', report)
        print(model['name'] + ': full cache, inputs, scores and both key indexes verified', flush=True)
    report['status'] = 'complete'
    write(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
