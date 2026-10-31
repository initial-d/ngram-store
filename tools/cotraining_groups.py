"""Optional model-side compaction of identical embedding-bag incidence columns.

This is NOT a byte-preserving conversion of arbitrary embedding rows. Averaging
a group preserves complete certified bag sums in real arithmetic, with measured
floating-point rounding. Individual rows and uncertified/OOV bags can change.
The existing frozen row-store contract is independent of this module.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


class BagPartition:
    """Exact equality of (bag ID, multiplicity) signatures; empty rows stay alone.

    Python tuple equality resolves hash collisions. Word, padding or otherwise
    independently accessed IDs should be protected. Merely sharing a source word
    is insufficient when multiplicities or other source words differ.
    """

    def __init__(self, row_to_group, domain_sha256, domain_bags):
        mapping = np.asarray(row_to_group)
        if mapping.ndim != 1 or not len(mapping) or mapping.dtype.kind not in "iu":
            raise ValueError("nonempty integer row-to-group map required")
        mapping = mapping.astype(np.int64, copy=True)
        if mapping.min() < 0 or mapping.max() >= len(mapping):
            raise ValueError("group ID out of range")
        self.sizes = np.bincount(mapping)
        if np.any(self.sizes == 0):
            raise ValueError("group IDs must be contiguous")
        self.mapping = mapping
        self.domain_sha256 = str(domain_sha256)
        self.domain_bags = int(domain_bags)
        self.mapping.flags.writeable = self.sizes.flags.writeable = False

    @classmethod
    def fit(cls, bags, nrows, *, protected=()):
        if type(nrows) is not int or nrows < 1:
            raise ValueError("positive row count required")
        protected = set(protected)
        if any(type(i) is not int or not 0 <= i < nrows for i in protected):
            raise ValueError("protected row ID out of range")
        signatures = [[] for _ in range(nrows)]
        digest = hashlib.sha256()
        domain_bags = 0
        for bid, bag in enumerate(bags):
            counts = Counter(bag)
            if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
                   or not 0 <= i < nrows for i in counts):
                raise ValueError("bag row ID out of range")
            pairs = sorted(counts.items())
            digest.update(np.asarray([len(pairs)], dtype="<u8").tobytes())
            digest.update(np.asarray(pairs, dtype="<u8").tobytes())
            for row, count in pairs:
                if row not in protected:
                    signatures[row].append((bid, count))
            domain_bags += 1
        if not domain_bags:
            raise ValueError("at least one registered bag required")
        mapping = np.empty(nrows, dtype=np.int64)
        groups = {}
        next_group = 0
        for row, signature in enumerate(signatures):
            # Unobserved rows are not evidence of equal future updates.
            key = tuple(signature) if signature and row not in protected else ("singleton", row)
            group = groups.get(key)
            if group is None:
                group = groups[key] = next_group
                next_group += 1
            mapping[row] = group
        return cls(mapping, digest.hexdigest(), domain_bags)

    def summary(self):
        return {"original_rows": len(self.mapping), "group_rows": len(self.sizes),
                "rows_in_nonsingleton_groups": int(self.sizes[self.sizes > 1].sum()),
                "nonsingleton_groups": int((self.sizes > 1).sum()),
                "domain_bags": self.domain_bags, "domain_sha256": self.domain_sha256,
                "mapping_sha256": hashlib.sha256(self.mapping.astype("<u8").tobytes()).hexdigest(),
                "contract": "certified aggregate equivalence in real arithmetic; not original-row byte identity"}

    def certifies(self, bag):
        """A bag is safe iff every used group has equal per-member multiplicity."""
        bag = np.asarray(bag)
        if bag.ndim != 1:
            raise ValueError("one-dimensional bag required")
        if not len(bag):
            return True
        if bag.dtype.kind not in "iu" or bag.min() < 0 or bag.max() >= len(self.mapping):
            raise ValueError("bag row ID out of range")
        rows, count = np.unique(bag, return_counts=True)
        group = self.mapping[rows]
        order = np.argsort(group, kind="stable")
        group, count = group[order], count[order]
        start = np.r_[0, np.flatnonzero(group[1:] != group[:-1]) + 1]
        observed = np.diff(np.r_[start, len(group)])
        return bool(np.all(observed == self.sizes[group[start]]) and
                    np.all(np.minimum.reduceat(count, start) == np.maximum.reduceat(count, start)))

    def means(self, rows, chunk_rows=4096):
        """Float64 accumulation; no assumption that FP32 bag sums are bit-exact."""
        rows = np.asarray(rows)
        if rows.ndim != 2 or len(rows) != len(self.mapping) or rows.dtype.kind != "f":
            raise ValueError("matching floating point rows required")
        if type(chunk_rows) is not int or chunk_rows < 1:
            raise ValueError("positive chunk size required")
        means = np.zeros((len(self.sizes), rows.shape[1]), dtype=np.float64)
        for start in range(0, len(rows), chunk_rows):
            part = rows[start:start + chunk_rows]
            if not np.isfinite(part).all():
                raise ValueError("finite rows required for aggregate equivalence")
            np.add.at(means, self.mapping[start:start + len(part)], part.astype(np.float64))
        means /= self.sizes[:, None]
        return means.astype(rows.dtype)

    def representatives(self):
        first = np.full(len(self.sizes), len(self.mapping), dtype=np.int64)
        np.minimum.at(first, self.mapping, np.arange(len(self.mapping)))
        return first[self.mapping]

    def save(self, path):
        np.savez(path, mapping=self.mapping.astype("<u8"), domain_sha256=self.domain_sha256,
                 domain_bags=self.domain_bags)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["mapping"], str(data["domain_sha256"]), int(data["domain_bags"]))


class CertifiedBagPlan:
    """Pre-certified ragged bag schedule; reduce each group once with multiplicity.

    Compilation is outside query timings and must be included in resource costs.
    The native weighted reducer accumulates in float64, then outputs float32.
    """
    def __init__(self, partition, bags, *, padding_idx=None):
        representatives = partition.representatives()
        ids, weights, offsets, denominators = [], [], [0], []
        original_occurrences = 0
        for bag in bags:
            bag = list(bag)
            if padding_idx is not None:
                bag = [i for i in bag if i != padding_idx]
            if not partition.certifies(bag):
                raise ValueError('uncertified bag: incomplete group or unequal multiplicity')
            denominators.append(max(1, len(bag)))
            original_occurrences += len(bag)
            counts = Counter(int(representatives[i]) for i in bag)
            for root, count in sorted(counts.items()):
                if int(np.float32(count)) != count:
                    raise ValueError('multiplicity cannot be represented exactly as float32')
                ids.append(root)
                weights.append(count)
            offsets.append(len(ids))
        self.ids = np.asarray(ids, dtype='<u8')
        self.weights = np.asarray(weights, dtype='<f4')
        self.offsets = np.asarray(offsets, dtype='<u8')
        self.denominators = np.asarray(denominators, dtype=np.int64)
        self.partition_sha256 = partition.summary()['mapping_sha256']
        self.original_occurrences = original_occurrences
        for values in (self.ids, self.weights, self.offsets, self.denominators):
            values.flags.writeable = False

    def summary(self):
        return {'bags': len(self.denominators), 'original_occurrences': self.original_occurrences,
                'group_occurrences': len(self.ids),
                'plan_bytes': sum(a.nbytes for a in (self.ids, self.weights, self.offsets, self.denominators)),
                'partition_sha256': self.partition_sha256}


class CertifiedBagRows:
    """Frozen native FP32 aggregates with mandatory per-query certification.

    Produced by benchmark_cotraining. This API intentionally exposes aggregates,
    rather than claiming arbitrary source rows were preserved. Native handles
    retain their usual single-worker ownership and trusted-artifact boundary.
    """
    def __init__(self, path, native_library=None):
        from .native_row_store import NativeRows
        path = Path(path)
        metadata = json.loads((path / 'manifest.json').read_text())
        self.partition = BagPartition.load(path / 'domain.npz')
        if metadata.get('partition') != self.partition.summary():
            raise ValueError('aggregate partition metadata mismatch')
        self.partition_sha256 = metadata['partition']['mapping_sha256']
        digest = hashlib.sha256()
        with (path / 'table.men').open('rb') as stream:
            for part in iter(lambda: stream.read(8 << 20), b''):
                digest.update(part)
        if digest.hexdigest() != metadata['artifact_sha256']:
            raise ValueError('aggregate table hash mismatch')
        self.table = NativeRows(path / 'table.men', native_library)
        if self.table.header['n_rows'] != len(self.partition.mapping) or self.table.header['row_bytes'] % 4:
            self.table.close()
            raise ValueError('aggregate table shape mismatch')
        self.dim = self.table.header['row_bytes'] // 4

    def compile(self, bags, *, padding_idx=None):
        return CertifiedBagPlan(self.partition, bags, padding_idx=padding_idx)

    def evaluate(self, plan, *, mean=False, chunk_rows=64):
        if not isinstance(plan, CertifiedBagPlan) or plan.partition_sha256 != self.partition_sha256:
            raise ValueError('compiled bag plan belongs to a different partition')
        output = np.empty((len(plan.denominators), self.dim), dtype=np.float32)
        if len(output):
            self.table.embedding_bag_into(plan.ids.ctypes.data, len(plan.ids), plan.offsets.ctypes.data,
                len(output), 1, False, (1 << 64) - 1, chunk_rows, output.ctypes.data, output.size,
                weights_address=plan.weights.ctypes.data, weights_bytes=plan.weights.nbytes)
            if mean:
                output /= plan.denominators[:, None]
        return output

    def aggregate(self, bags, *, mean=False, padding_idx=None):
        return self.evaluate(self.compile(bags, padding_idx=padding_idx), mean=mean)

    def close(self):
        self.table.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def export_certified_bags(rows, partition, output, *, native_library=None, block_rows=16, compress=True):
    """Export a separate aggregate-certified artifact; original row packers unchanged.

    FP32 finite source rows only. Produces/verifies a native table, exact incidence
    map and an explicit aggregate contract. Every decoded canonical row is byte
    checked; source row bytes are allowed to change. Caller owns key registries
    and must include them in complete application costs.
    """
    from .native_row_store import NativeRows, pack_native
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] < 1 or rows.dtype != np.dtype('<f4'):
        raise ValueError('2D little-endian float32 rows required')
    means = partition.means(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        source = output / 'staging.f32'
        digest = hashlib.sha256()
        with source.open('wb') as stream:
            for begin in range(0, len(rows), 4096):
                raw = means[partition.mapping[begin:begin + 4096]].tobytes()
                stream.write(raw)
                digest.update(raw)
        storage = pack_native(source, output / 'table.men', rows.shape[1] * 4, block_rows,
                              representatives=partition.representatives().tolist(), predictor='xor', compress=compress)
        decoded = hashlib.sha256()
        with NativeRows(output / 'table.men', native_library) as native:
            for begin in range(0, len(rows), 4096):
                end = min(len(rows), begin + 4096)
                actual = native.rows(range(begin, end))
                if actual != means[partition.mapping[begin:end]].tobytes():
                    raise ValueError('native canonical-row verification failed')
                decoded.update(actual)
        if decoded.hexdigest() != digest.hexdigest():
            raise ValueError('canonical table hash mismatch')
        source.unlink()
        partition.save(output / 'domain.npz')
        artifact_digest = hashlib.sha256()
        with (output / 'table.men').open('rb') as stream:
            for part in iter(lambda: stream.read(8 << 20), b''):
                artifact_digest.update(part)
        metadata = {'name': 'certified-groups', 'dtype': 'little-endian float32', 'dim': rows.shape[1],
                    'source_row_byte_preserving': False, 'native_decodes_declared_rows_byte_exactly': True,
                    'all_original_ids_checked': len(rows), 'decoded_sha256': digest.hexdigest(),
                    'artifact_sha256': artifact_digest.hexdigest(), 'storage': storage,
                    'partition': partition.summary(),
                    'contract': 'certified bag aggregates in real arithmetic; FP rounding measured; reject uncertified bags'}
        (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return {**metadata, 'artifact_and_certificate_bytes': sum(f.stat().st_size for f in output.iterdir()),
                'omitted_application_costs': 'key registry, model word/output rows and compiled schedules'}
    except BaseException:
        shutil.rmtree(output)
        raise


def grouped_embedding_bag(dense_rows, partition, *, padding_idx=None, scale_gradient=True):
    """Experimental sparse SGD parameterization for a fixed certified bag domain.

    A k-member group's shared gradient is divided by k, reproducing the
    per-original-row update rather than silently changing the learning rate.
    Supports sums; callers may apply the same forward averaging convention as
    their baseline. Rejects incomplete/unequal group queries by default.
    Row regularizers, weight decay and arbitrary dynamic lookups are outside
    the demonstrated equivalence contract.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    if dense_rows.ndim != 2 or dense_rows.shape[0] != len(partition.mapping):
        raise ValueError("dense rows do not match partition")
    if padding_idx is not None and (not 0 <= padding_idx < len(partition.mapping)
                                   or partition.sizes[partition.mapping[padding_idx]] != 1):
        raise ValueError("padding ID must be a protected singleton")

    class GroupedBag(nn.Module):
        def __init__(self, initial_weight):
            super().__init__()
            self.partition = partition
            self.weight = nn.Parameter(initial_weight)
            self.register_buffer("row_to_group", torch.tensor(partition.mapping.copy(), device=initial_weight.device))
            self.register_buffer("group_sizes", torch.tensor(partition.sizes.copy(), device=initial_weight.device))
            if scale_gradient:
                self.weight.register_hook(self._per_row_gradient)

        def _per_row_gradient(self, gradient):
            gradient = gradient.coalesce()
            index = gradient.indices()
            values = gradient.values() / self.group_sizes[index[0], None]
            return torch.sparse_coo_tensor(index, values, gradient.shape,
                                           dtype=values.dtype, device=values.device,
                                           is_coalesced=True, check_invariants=False)

        def forward(self, bags, *, check=True):
            if bags.ndim != 2 or bags.dtype != torch.long:
                raise ValueError("2D int64 original-ID bags required")
            if check and any(not partition.certifies(bag) for bag in bags.detach().cpu().numpy()):
                raise ValueError("uncertified bag: incomplete group or unequal multiplicity")
            padding = None if padding_idx is None else int(partition.mapping[padding_idx])
            return F.embedding_bag(self.row_to_group[bags], self.weight,
                                   mode="sum", sparse=True, padding_idx=padding)

        def expanded_rows(self):
            return self.weight[self.row_to_group]

    # Pass constructor data as an argument: capturing dense_rows in the local
    # class would keep the entire original table alive after conversion.
    initial_weight = torch.from_numpy(partition.means(dense_rows.detach().cpu().numpy())).to(dense_rows.device)
    return GroupedBag(initial_weight)
