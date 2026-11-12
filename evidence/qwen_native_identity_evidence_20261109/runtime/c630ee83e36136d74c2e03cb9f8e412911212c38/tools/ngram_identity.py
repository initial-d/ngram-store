"""Training-only canonical n-gram vocabulary and exact/bucket identity controls.

Keep Engram's token normalization, padding, six heads and downstream operations.
Only the tying of distinct canonical keys changes. No evaluation keys are added.
"""
from collections import Counter, OrderedDict
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def array_sha(array):
    return hashlib.sha256(np.asarray(array, dtype='<i8').tobytes()).hexdigest()


def windows(tokens, order, pad):
    """Oldest-to-current tuples with the same left padding as Engram."""
    x = np.asarray(tokens, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError('batched token IDs required')
    b, t = x.shape
    left = np.pad(x, ((0, 0), (order - 1, 0)), constant_values=pad)
    return np.stack([left[:, k:k+t] for k in range(order)], axis=-1)


class Vocabulary:
    def __init__(self, keys, counts, metadata):
        self.keys = {n: np.asarray(keys[n], dtype=np.int64).reshape(-1, n) for n in (2, 3, 4)}
        self.counts = {n: np.asarray(counts[n], dtype=np.int64) for n in (2, 3, 4)}
        self.metadata = metadata
        self.indices = {}
        for n in (2, 3, 4):
            tuples = [tuple(row) for row in self.keys[n].tolist()]
            if not tuples or tuples != sorted(set(tuples)) or len(tuples) != len(self.counts[n]):
                raise ValueError('nonempty, sorted unique training keys required')
            self.indices[n] = {key: i + 1 for i, key in enumerate(tuples)}

    @classmethod
    def build(cls, payload, source_mapping, prepared_sha):
        counts = {n: Counter() for n in (2, 3, 4)}
        for index in payload['train']:
            # The terminal EOS output has no next-token target. Earlier prompt
            # positions still affect supervised suffix outputs through attention.
            ids = payload['sequences'][index]['ids'][:-1]
            if not ids:
                raise ValueError('empty training prefix')
            x = source_mapping.compressed_tokenizer(np.asarray([ids], dtype=np.int64))
            for n in counts:
                counts[n].update(tuple(row) for row in windows(x, n, source_mapping.pad_id).reshape(-1, n).tolist())
        keys = {n: sorted(counts[n]) for n in counts}
        meta = {'schema_version': 1, 'prepared_sha256': prepared_sha,
                'training_sequences': len(payload['train']), 'pad_id': source_mapping.pad_id,
                'lookup_sha256': array_sha(source_mapping.compressed_tokenizer.lookup_table),
                'domain': 'Canonical compressed-token training keys only, orders 2/3/4; terminal EOS output excluded',
                'normalization': 'Same pinned Engram compressed tokenizer as hash controls; not raw-token identity',
                'oov_rule': 'All-zero per-head embedding for unseen canonical keys in the primary pair'}
        return cls(keys, {n: [counts[n][k] for k in keys[n]] for n in keys}, meta)

    def save(self, path):
        path = Path(path)
        with path.open('wb') as stream:
            np.savez_compressed(stream, metadata=np.array(json.dumps(self.metadata, sort_keys=True)),
                **{f'keys{n}': self.keys[n] for n in self.keys},
                **{f'counts{n}': self.counts[n] for n in self.keys})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as values:
            return cls({n: values[f'keys{n}'] for n in (2, 3, 4)},
                       {n: values[f'counts{n}'] for n in (2, 3, 4)}, json.loads(str(values['metadata'])))

    def known_ids(self, canonical):
        output = []
        for n in (2, 3, 4):
            shape = canonical.shape
            ids = np.array([self.indices[n].get(tuple(row), 0) for row in
                            windows(canonical, n, self.metadata['pad_id']).reshape(-1, n).tolist()], dtype=np.int64).reshape(shape)
            output.extend([ids, ids])
        return np.stack(output, axis=-1)


def reference_mapping(source, vocabulary):
    mapping = copy.copy(source)
    mapping.vocab_size_per_ngram = [len(vocabulary.keys[n]) for n in (2, 3, 4)]
    mapping.vocab_size_across_layers = mapping.calculate_vocab_size_across_layers()
    return mapping


class IdentityMapping:
    def __init__(self, source, vocabulary, exact, native_oov=False):
        if source.pad_id != vocabulary.metadata['pad_id'] or array_sha(source.compressed_tokenizer.lookup_table) != vocabulary.metadata['lookup_sha256']:
            raise ValueError('token normalization or padding changed')
        self.source = reference_mapping(source, vocabulary)
        self.vocabulary = vocabulary
        self.exact = exact
        self.native_oov = native_oov
        self.cache = OrderedDict()
        self.layer_ids = self.source.layer_ids
        self.vocab_size_across_layers = {
            # Pad exact tables with never-addressed rows to match both prime
            # hash heads' parameter and row-byte budgets exactly.
            layer: [[size + 1 for size in heads] for heads in self.source.vocab_size_across_layers[layer]]
            for layer in self.layer_ids}

    def hash(self, input_ids):
        canonical = self.source.compressed_tokenizer(input_ids)
        result = {layer: [] for layer in self.layer_ids}
        for row in canonical:
            key = (row.tobytes(), self.native_oov)
            cached = self.cache.get(key)
            if cached is None:
                known = self.vocabulary.known_ids(row[None, :])
                cached = {}
                for layer in self.layer_ids:
                    if self.exact:
                        ids = known
                    else:
                        ids = self.source._get_ngram_hashes(row[None, :], layer) + 1
                        if not self.native_oov:
                            ids = np.where(known != 0, ids, 0)
                    cached[layer] = ids[0]
                self.cache[key] = cached
                if len(self.cache) > 65536:
                    self.cache.popitem(last=False)
            else:
                self.cache.move_to_end(key)
            for layer in self.layer_ids:
                result[layer].append(cached[layer])
        return {layer: np.stack(rows) for layer, rows in result.items()}


def collision_audit(vocabulary, source):
    source = reference_mapping(source, vocabulary)
    layer = source.layer_ids[0]
    result = {}
    for n in (2, 3, 4):
        pairs = source._get_ngram_hashes(vocabulary.keys[n], layer)[:, -1, (n-2)*2:(n-1)*2]
        heads = []
        for head in range(2):
            _, reverse, occupancy = np.unique(pairs[:, head], return_inverse=True, return_counts=True)
            collided = occupancy[reverse] > 1
            heads.append({'buckets': source.vocab_size_across_layers[layer][n-2][head],
                          'occupied_buckets': len(occupancy), 'keys_in_shared_buckets': int(collided.sum()),
                          'training_occurrences_in_shared_buckets': int(vocabulary.counts[n][collided].sum()),
                          'max_bucket_keys': int(occupancy.max())})
        result[str(n)] = {'unique_training_keys': len(pairs), 'training_occurrences': int(vocabulary.counts[n].sum()),
                          'heads': heads, 'distinct_joint_head_addresses': len(np.unique(pairs, axis=0)),
                          'key_bytes_uncompressed': int(vocabulary.keys[n].nbytes)}
    return result


def configure_memory(replay, module, vocabulary, exact, seed):
    """Exact rows start as clones of the matched hash bucket rows, not new noise."""
    import torch
    branch = replay.memory
    mapping = IdentityMapping(branch.hash_mapping, vocabulary, exact)
    sizes = [x for heads in mapping.vocab_size_across_layers[branch.layer_id] for x in heads]
    hash_sizes = [x + 1 for heads in mapping.source.vocab_size_across_layers[branch.layer_id] for x in heads]
    # Constructors run on CPU; preserve the global RNG for subsequent operations.
    with torch.random.fork_rng(devices=[]):
        embedding = module.MultiHeadEmbedding(sizes, 32)
    offsets = embedding.offsets.tolist()
    reference_hashes = {}
    for n in (2, 3, 4):
        reference_hashes[n] = mapping.source._get_ngram_hashes(vocabulary.keys[n], branch.layer_id)[:, -1]
    with torch.no_grad():
        for head, (offset, size) in enumerate(zip(offsets, sizes)):
            n = 2 + head // 2
            generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'{seed}:identity:{n}:{head % 2}'.encode()).digest()[:8], 'little'))
            reference = torch.empty(hash_sizes[head], 32)
            torch.nn.init.normal_(reference, mean=0., std=replay.text.config.initializer_range, generator=generator)
            reference[0].zero_()
            if exact:
                embedding.embedding.weight[offset:offset+size].copy_(reference)
                embedding.embedding.weight[offset].zero_()
                indices = torch.from_numpy(reference_hashes[n][:, head] + 1)
                embedding.embedding.weight[offset+1:offset+1+len(indices)].copy_(reference[indices])
            else:
                embedding.embedding.weight[offset:offset+size].copy_(reference)
    zeros = torch.tensor(offsets, device=replay.device)
    embedding.to(replay.device)
    def zero_unknown_grad(grad):
        grad.index_fill_(0, zeros, 0.)
        return grad
    embedding.embedding.weight.register_hook(zero_unknown_grad)
    branch.hash_mapping = mapping
    branch.multi_head_embedding = embedding
    return {'addressing': 'exact-canonical-key' if exact else 'shared-hash-bucket',
            'head_sizes': sizes, 'physical_rows': sum(sizes), 'row_bytes': sum(sizes) * 32 * 4,
            'unique_keys_per_order': {str(n): len(vocabulary.keys[n]) for n in (2, 3, 4)},
            'logical_exact_rows_with_zero_sentinels': 2 * sum(len(vocabulary.keys[n]) + 1 for n in (2,3,4)),
            'initialization': 'Exact key vectors clone matched hash vectors per head; all nonrow parameters identical by seed',
            'oov': vocabulary.metadata['oov_rule'], 'collision_audit': collision_audit(vocabulary, mapping.source)}
