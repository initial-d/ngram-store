"""Opt-in, source-specific L2 bounds for partial incidence-group queries.

The strict certificate remains weight-independent. This separate path uses
actual source residuals and includes native reduction/cast rounding, comparing
returned vectors to real-arithmetic sums/means of the declared FP32 source.
"""
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time

import numpy as np

from .cotraining_groups import CertifiedBagRows, _count_bags
from .external_cotraining import file_hash


def _up(value):
    if np.isscalar(value):
        return math.nextafter(float(value), math.inf) if value != 0 else 0.
    value = np.asarray(value, dtype=np.float64)
    return np.where(value == 0, 0., np.nextafter(value, np.inf))


def _down(value):
    if np.isscalar(value):
        return max(0., math.nextafter(float(value), -math.inf))
    value = np.asarray(value, dtype=np.float64)
    return np.maximum(0., np.nextafter(value, -np.inf))


def _gamma(operations):
    # IEEE round-to-nearest unit roundoff, with outward rounding of the factor.
    x = _up(np.asarray(operations, dtype=np.float64) * 2.**-53)
    if np.any(x >= .5):
        raise ValueError('too many operations for the floating-point bound')
    return _up(x / _down(1. - x))


def _norm_upper(values):
    values = np.asarray(values, dtype=np.float64)
    total = np.zeros(values.shape[:-1], dtype=np.float64)
    for j in range(values.shape[-1]):
        total = _up(total + _up(values[..., j] * values[..., j]))
    return _up(np.sqrt(total))


def _norm_lower(values):
    values = np.asarray(values, dtype=np.float64)
    total = np.zeros(values.shape[:-1], dtype=np.float64)
    for j in range(values.shape[-1]):
        total = _down(total + _down(values[..., j] * values[..., j]))
    return _down(np.sqrt(total))


def _sum_upper(values):
    result = 0.
    for value in values:
        result = float(_up(result + value))
    return result


def build_error_bounds(source, store, output, *, native_library=None, chunk_rows=1024):
    """Build an immutable sidecar without retaining source vectors at query time.

    Scratch uses two disk-backed G-by-d FP64 arrays. This optional build has its
    own RSS/disk/time costs and does not inherit compact_native's memory budget.
    Native store and source are never modified. Existing outputs are refused.
    """
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError('positive chunk size required')
    source, store, output = Path(source), Path(store), Path(output).resolve()
    source_hash = file_hash(source)
    metadata = json.loads((store / 'manifest.json').read_text())
    if metadata.get('source_sha256', source_hash) != source_hash:
        raise ValueError('source hash differs from the native build source')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    maps = []
    try:
        with CertifiedBagRows(store, native_library) as native:
            partition, dim = native.partition, native.dim
            nrows, ngroups = len(partition.mapping), len(partition.sizes)
            if source.stat().st_size != nrows * dim * 4:
                raise ValueError('source shape differs from certified table')
            rows = np.memmap(source, mode='r', dtype='<f4', shape=(nrows, dim))
            row_norms = np.memmap(output / 'row-l2.f64', mode='w+', dtype='<f8', shape=(nrows,))
            totals = np.memmap(output / 'group-total-l2.f64', mode='w+', dtype='<f8', shape=(ngroups,))
            defects = np.memmap(output / 'group-defect-l2.f64', mode='w+', dtype='<f8', shape=(ngroups,))
            maps.extend((rows, row_norms, totals, defects))
            totals[:] = 0
            with tempfile.TemporaryDirectory(prefix='.residual-', dir=output) as tmp:
                signed = np.memmap(Path(tmp) / 'signed.f64', mode='w+', dtype='<f8', shape=(ngroups, dim))
                absolute = np.memmap(Path(tmp) / 'absolute.f64', mode='w+', dtype='<f8', shape=(ngroups, dim))
                maps.extend((signed, absolute))
                signed[:] = absolute[:] = 0
                for begin in range(0, nrows, chunk_rows):
                    end = min(nrows, begin + chunk_rows)
                    original = rows[begin:end].astype(np.float64)
                    projected = np.frombuffer(native.table.rows(range(begin, end)), dtype='<f4').reshape(-1, dim).astype(np.float64)
                    if not np.isfinite(original).all() or not np.isfinite(projected).all():
                        raise ValueError('finite FP32 source and projected rows required')
                    residual = original - projected
                    # An outward-rounded subtraction magnitude encloses the
                    # exact difference of the two exactly promoted FP32 values.
                    magnitude = _up(np.abs(residual))
                    magnitude[original == projected] = 0.
                    norms = _norm_upper(magnitude)
                    row_norms[begin:end] = norms
                    groups = partition.mapping[begin:end]
                    np.add.at(totals, groups, norms)
                    np.add.at(signed, groups, residual)
                    np.add.at(absolute, groups, magnitude)
                for begin in range(0, ngroups, chunk_rows):
                    end = min(ngroups, begin + chunk_rows)
                    counts = partition.sizes[begin:end]
                    gamma = _gamma(counts + 1)
                    denominator = _down(1. - gamma)
                    totals[begin:end] = _up(totals[begin:end] / denominator)
                    abs_upper = _up(absolute[begin:end] / denominator[:, None])
                    # Includes residual subtraction and sequential accumulation
                    # error. In particular, rounded means need not have a
                    # mathematically zero residual sum.
                    component = _up(np.abs(signed[begin:end]) + _up(gamma[:, None] * abs_upper))
                    defects[begin:end] = _norm_upper(component)
                for array in (signed, absolute):
                    array.flush()
                    array._mmap.close()
                maps = maps[:-2]
            for array in (row_norms, totals, defects):
                array.flush()
            report = {'schema_version': 1, 'format': 'source-residual-l2-v1',
                'shape': [nrows, dim], 'groups': ngroups,
                'source_sha256': source_hash, 'artifact_sha256': metadata['artifact_sha256'],
                'partition_sha256': partition.mapping_sha256,
                'files_sha256': {name: file_hash(output / name) for name in
                    ('row-l2.f64', 'group-total-l2.f64', 'group-defect-l2.f64')},
                'construction': {'seconds': time.perf_counter() - started,
                    'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                        (1 if sys.platform == 'darwin' else 1024),
                    'scratch_logical_bytes': 2 * ngroups * dim * 8,
                    'chunk_rows': chunk_rows},
                'contract': 'Source-specific L2 bounds; returned native vectors versus real-arithmetic FP32 source aggregates.',
                'limitations': ['Optional approximation, not original-row byte preservation.',
                    'Assumes IEEE round-to-nearest arithmetic; no FP32 source-reducer byte identity.',
                    'Query-relative L2 is not semantic quality, cosine similarity or downstream loss.',
                    'Scratch/RSS/runtime maps have separate costs; no hard memory cap.',
                    'Hash-bound trusted immutable source/store/sidecar; no quantized table substitution.']}
            if file_hash(source) != source_hash or file_hash(store / 'table.men') != metadata['artifact_sha256']:
                raise ValueError('source or native table changed during bounds construction')
            (output / 'bounds.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            return {**report, 'sidecar_bytes': sum(p.stat().st_size for p in output.iterdir())}
    except BaseException:
        for array in maps:
            array._mmap.close()
        maps.clear()
        shutil.rmtree(output)
        raise
    finally:
        for array in maps:
            array._mmap.close()


@dataclass(frozen=True)
class BagErrorAssessment:
    """Candidate outputs and conservative bounds, before a user's budget gate."""
    vectors: np.ndarray
    projection_l2_bounds: np.ndarray
    rounding_l2_bounds: np.ndarray
    absolute_l2_bounds: np.ndarray
    relative_l2_bounds: np.ndarray
    exact_domain: np.ndarray
    group_occurrences: int

    def accepts(self, *, max_relative_l2=None, max_absolute_l2=None):
        if max_relative_l2 is None and max_absolute_l2 is None:
            raise ValueError('an explicit L2 error budget is required')
        accepted = np.ones(len(self.vectors), dtype=bool)
        for budget, bounds in ((max_relative_l2, self.relative_l2_bounds),
                               (max_absolute_l2, self.absolute_l2_bounds)):
            if budget is not None:
                if isinstance(budget, (bool, np.bool_)) or not np.isscalar(budget) or not math.isfinite(budget) or budget < 0:
                    raise ValueError('finite nonnegative L2 budget required')
                accepted &= bounds <= budget
        return accepted

    def summary(self):
        return {'exact_domain': self.exact_domain.tolist(),
            'projection_l2_bounds': self.projection_l2_bounds.tolist(),
            'rounding_l2_bounds': self.rounding_l2_bounds.tolist(),
            'absolute_l2_bounds': self.absolute_l2_bounds.tolist(),
            'relative_l2_bounds': [float(v) if math.isfinite(v) else None for v in self.relative_l2_bounds],
            'unbounded_relative_error': (~np.isfinite(self.relative_l2_bounds)).tolist(),
            'group_occurrences': self.group_occurrences,
            'contract': 'Bounds versus real-arithmetic sums/means of the declared FP32 source; approximation is opt-in.'}


class BudgetedBagRows:
    """Frozen aggregate queries with actual-residual bounds and explicit budgets.

    assess() exposes candidates for diagnostics. aggregate() refuses the entire
    batch if any bound exceeds a budget. No original source matrix is loaded.
    """
    def __init__(self, store, bounds, native_library=None):
        self.native = CertifiedBagRows(store, native_library)
        self.arrays = []
        try:
            bounds = Path(bounds)
            metadata = json.loads((bounds / 'bounds.json').read_text())
            store_metadata = json.loads((Path(store) / 'manifest.json').read_text())
            partition = self.native.partition
            if metadata.get('format') != 'source-residual-l2-v1' or metadata.get('schema_version') != 1:
                raise ValueError('unsupported error-bound sidecar')
            if (metadata.get('artifact_sha256') != store_metadata['artifact_sha256']
                    or metadata.get('partition_sha256') != partition.mapping_sha256
                    or metadata.get('shape') != [len(partition.mapping), self.native.dim]
                    or metadata.get('groups') != len(partition.sizes)
                    or store_metadata.get('source_sha256', metadata.get('source_sha256')) != metadata.get('source_sha256')):
                raise ValueError('error bounds belong to a different source/table/partition')
            for name, count in (('row-l2.f64', len(partition.mapping)),
                                ('group-total-l2.f64', len(partition.sizes)),
                                ('group-defect-l2.f64', len(partition.sizes))):
                path = bounds / name
                if path.stat().st_size != count * 8 or file_hash(path) != metadata['files_sha256'][name]:
                    raise ValueError('error-bound array size/hash mismatch')
                array = np.memmap(path, mode='r', dtype='<f8', shape=(count,))
                self.arrays.append(array)
                if not np.isfinite(array).all() or np.any(array < 0):
                    raise ValueError('finite nonnegative residual bounds required')
            self.row_norms, self.group_totals, self.group_defects = self.arrays
            self.source_sha256 = metadata['source_sha256']
        except BaseException:
            self.close()
            raise

    def _projection_bound(self, rows, counts):
        groups = defaultdict(list)
        for row, count in zip(rows, counts):
            groups[int(self.native.partition.mapping[row])].append((int(row), int(count)))
        total = 0.
        for group, members in groups.items():
            if self.group_totals[group] == 0:
                continue
            # t=0 is the ordinary triangle bound. Centering counts by t exploits
            # cancellation even when one/few members are absent or counts differ:
            # sum c_i r_i = sum (c_i-t)r_i + t sum r_i.
            # The convex piecewise-linear bound has a minimum at 0 or one of
            # the observed positive counts. Missing members have count zero.
            present_lower = 0.
            for row, _ in members:
                present_lower = float(_down(present_lower + self.row_norms[row]))
            missing = float(_up(max(0., float(self.group_totals[group]) - present_lower)))
            best = math.inf
            for center in {0, *(count for _, count in members)}:
                cost = float(_up(center * self.group_defects[group]))
                if center:
                    cost = float(_up(cost + _up(center * missing)))
                for row, count in members:
                    cost = float(_up(cost + _up(abs(count - center) * self.row_norms[row])))
                best = min(best, cost)
            total = float(_up(total + best))
        return total

    def assess(self, bags, *, mean=False, padding_idx=None, chunk_rows=64):
        bags = list(bags)
        counted = _count_bags(bags, nrows=len(self.native.partition.mapping), padding_idx=padding_idx)
        lengths = counted['lengths']
        if np.any(lengths > 2**53):
            raise ValueError('bag length cannot be represented exactly as float64')
        projection = np.zeros(len(lengths))
        ids, weights, offsets = [], [], [0]
        boundaries = np.searchsorted(counted['bags'], np.arange(len(lengths) + 1))
        for bid in range(len(lengths)):
            selected = slice(boundaries[bid], boundaries[bid + 1])
            rows, counts = counted['rows'][selected], counted['counts'][selected]
            projection[bid] = self._projection_bound(rows, counts)
            grouped = {}
            for row, count in zip(rows, counts):
                group = int(self.native.partition.mapping[row])
                if group in grouped:
                    grouped[group][1] += int(count)
                else:
                    grouped[group] = [int(row), int(count)]
            representatives = sorted(grouped.values())
            if any(weight > 2**24 for _, weight in representatives):
                raise ValueError('group multiplicity exceeds the exact float32 integer range')
            ids.extend(row for row, _ in representatives)
            weights.extend(weight for _, weight in representatives)
            offsets.append(len(ids))
        ids = np.asarray(ids, dtype='<u8')
        weights = np.asarray(weights, dtype='<f4')
        offsets = np.asarray(offsets, dtype='<u8')
        # One gather and one column-wise interval norm pass across the complete
        # schedule avoid B separate d-column Python loops. Arithmetic order for
        # each row and bag remains identical to the original bounded path.
        canonical = np.frombuffer(self.native.table.rows(ids.tolist()), dtype='<f4').reshape(-1, self.native.dim)
        weighted_norms = _up(weights.astype(np.float64) * _norm_upper(canonical))
        magnitudes = [_sum_upper(weighted_norms[offsets[bid]:offsets[bid+1]])
                      for bid in range(len(lengths))]
        vectors = np.zeros((len(lengths), self.native.dim), dtype='<f4')
        if len(lengths):
            self.native.table.embedding_bag_into(ids.ctypes.data, len(ids), offsets.ctypes.data,
                len(lengths), 1, False, (1 << 64) - 1, chunk_rows, vectors.ctypes.data, vectors.size,
                weights_address=weights.ctypes.data, weights_bytes=weights.nbytes)
        if not np.isfinite(vectors).all():
            raise ValueError('nonfinite native output cannot satisfy an error budget')
        # Products of binary32 rows and <=24-bit integer weights are exact in
        # binary64. Bound addition error, then binary32 output rounding.
        operations = np.diff(offsets).astype(np.int64)
        tiny = float(_up(math.sqrt(self.native.dim) * 2.**-150))
        cast_factor = float(_up(2.**-24 / _down(1. - 2.**-24)))
        rounding = _up(_up(_gamma(operations) * magnitudes) + _up(cast_factor * _norm_upper(vectors)))
        rounding = _up(rounding + np.where(operations > 0, tiny, 0.))
        if mean:
            denominator = np.maximum(lengths, 1)
            vectors /= denominator[:, None]
            if not np.isfinite(vectors).all():
                raise ValueError('nonfinite native mean cannot satisfy an error budget')
            projection = _up(projection / denominator)
            rounding = _up(rounding / denominator)
            # Includes division rounding as well as the second binary32 cast.
            division = _up(float(_gamma(2)) * _norm_upper(vectors))
            rounding = _up(rounding + _up(cast_factor * _norm_upper(vectors)) + division
                           + np.where(lengths > 0, tiny, 0.))
        absolute = _up(projection + rounding)
        lower = _down(_norm_lower(vectors) - absolute)
        relative = np.full(len(lengths), np.inf)
        np.divide(absolute, lower, out=relative, where=lower > 0)
        relative = _up(relative)
        relative[(absolute == 0) & (lower == 0)] = 0.
        # Strict membership diagnostics are independent of residual magnitudes.
        exact_domain = self.native.partition.certifies_many(bags, padding_idx=padding_idx)
        for array in (vectors, projection, rounding, absolute, relative, exact_domain):
            array.flags.writeable = False
        return BagErrorAssessment(vectors, projection, rounding, absolute, relative, exact_domain, len(ids))

    def aggregate(self, bags, *, max_relative_l2=None, max_absolute_l2=None, mean=False, padding_idx=None):
        result = self.assess(bags, mean=mean, padding_idx=padding_idx)
        accepted = result.accepts(max_relative_l2=max_relative_l2, max_absolute_l2=max_absolute_l2)
        if not accepted.all():
            bid = int(np.flatnonzero(~accepted)[0])
            raise ValueError(f'bag {bid} exceeds L2 error budget: absolute bound {result.absolute_l2_bounds[bid]:.9g}, '
                             f'relative bound {result.relative_l2_bounds[bid]:.9g}')
        return result

    def close(self):
        for array in self.arrays:
            array._mmap.close()
        self.arrays.clear()
        self.native.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
