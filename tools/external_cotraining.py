"""Disk-backed exact incidence construction and ordered FP32 group projection.

No resident array scales with total rows, occurrences or groups. The caller's
input iterator and the Python/NumPy runtime are outside the managed buffer
budget. Filesystem page cache is reclaimable and is not bounded by this API.
Runtime materialization and native packing retain their existing RAM contracts.
"""
from collections import OrderedDict
from contextlib import ExitStack
import hashlib
import heapq
import itertools
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import tempfile
import zipfile

import numpy as np

U64 = struct.Struct('<Q')
PAIR = struct.Struct('<QQ')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def _budget(value):
    if type(value) is not int or value < 8 << 20:
        raise ValueError('managed memory budget must be at least 8 MiB')
    return value


def _records(stream, fields):
    record = struct.Struct('<' + 'Q' * fields)
    for block in iter(lambda: stream.read(record.size * 256), b''):
        if len(block) % record.size:
            raise ValueError('truncated integer record file')
        yield from record.iter_unpack(block)


def _merge(paths, output, fields):
    """At most 16 readers, one fixed block and one heap record per reader."""
    record = struct.Struct('<' + 'Q' * fields)
    with ExitStack() as stack:
        readers = [iter(_records(stack.enter_context(p.open('rb')), fields)) for p in paths]
        heap = []
        for index, reader in enumerate(readers):
            first = next(reader, None)
            if first is not None:
                heap.append((first, index))
        heapq.heapify(heap)
        with output.open('wb') as stream:
            buffer = bytearray()
            while heap:
                item, index = heapq.heappop(heap)
                buffer.extend(record.pack(*item))
                following = next(readers[index], None)
                if following is not None:
                    heapq.heappush(heap, (following, index))
                if len(buffer) >= 65536:
                    stream.write(buffer)
                    buffer.clear()
            stream.write(buffer)


class _Sorter:
    """Fixed-size in-place sorts and bounded-fan-in passes; no all-run list."""
    def __init__(self, path, fields, budget, progress=None):
        self.path = Path(path)
        self.path.mkdir()
        self.fields = fields
        self.record = struct.Struct('<' + 'Q' * fields)
        self.limit = max(self.record.size, budget // 8)
        self.buffer = bytearray()
        self.runs = 0
        self.progress = progress

    def add(self, *values):
        self.buffer.extend(self.record.pack(*values))
        if len(self.buffer) >= self.limit:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        dtype = np.dtype([(str(i), '<u8') for i in range(self.fields)])
        array = np.frombuffer(self.buffer, dtype=dtype)
        array.sort(order=dtype.names, kind='quicksort')
        array.tofile(self.path / f'0-{self.runs}')
        del array
        self.buffer = bytearray()
        self.runs += 1
        if self.progress:
            self.progress('sorted-run')

    def finish(self):
        self.flush()
        if not self.runs:
            (self.path / '0-0').touch()
            self.runs = 1
        phase, count = 0, self.runs
        while count > 1:
            next_count = 0
            for begin in range(0, count, 16):
                paths = [self.path / f'{phase}-{i}' for i in range(begin, min(begin + 16, count))]
                _merge(paths, self.path / f'{phase + 1}-{next_count}', self.fields)
                if self.progress:
                    self.progress('merged-run')
                for path in paths:
                    path.unlink()
                next_count += 1
            phase, count = phase + 1, next_count
        return self.path / f'{phase}-0'


def _signature_hash():
    return hashlib.sha256()


class _BagPayload:
    """Bounded small-bag byte buffer; large bags spill to a visible owned file."""
    def __init__(self, path, limit):
        self.path, self.limit = path, limit
        self.buffer, self.stream = bytearray(), None

    def write(self, pair):
        if self.stream is None and len(self.buffer) + len(pair) > self.limit:
            self.stream = self.path.open('w+b')
            self.stream.write(self.buffer)
            self.buffer = bytearray()
        if self.stream is None:
            self.buffer.extend(pair)
        else:
            self.stream.write(pair)

    def update_digest(self, digest):
        if self.stream is None:
            digest.update(self.buffer)
        else:
            self.stream.seek(0)
            for block in iter(lambda: self.stream.read(65536), b''):
                digest.update(block)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.close()
            self.path.unlink()


def _same_signature(stream, left, right, size):
    stream.flush()
    for begin in range(0, size, 65536):
        length = min(65536, size - begin)
        a = os.pread(stream.fileno(), length, left + begin)
        b = os.pread(stream.fileno(), length, right + begin)
        if len(a) != length or len(b) != length:
            raise ValueError('truncated signature spool')
        if a != b:
            return False
    return True


def fit_external(bags, nrows, output, *, memory_bytes=64 << 20, protected=(), progress=None):
    """Match BagPartition.fit exactly, including first-row group numbering.

    Bags and protected IDs may be single-pass iterators. Empty/unobserved and
    protected rows remain singletons. Complete signatures resolve digest
    collisions; digests only index candidates. The output contains two raw
    integer arrays and a hash-linked manifest, not an open database.
    """
    memory_bytes = _budget(memory_bytes)
    if type(nrows) is not int or not 0 < nrows <= np.iinfo(np.int64).max:
        raise ValueError('positive int64 row count required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def notify(stage):
        if progress:
            progress(stage, output)

    try:
        with tempfile.TemporaryDirectory(prefix='.work-', dir=output) as temporary:
            work = Path(temporary)
            raw = _Sorter(work / 'raw', 2, memory_bytes, notify)
            domain_bags = 0
            for bid, bag in enumerate(bags):
                if bid > np.iinfo(np.int64).max:
                    raise ValueError('too many registered bags')
                for row in bag:
                    if (not isinstance(row, (int, np.integer)) or isinstance(row, (bool, np.bool_))
                            or not 0 <= row < nrows):
                        raise ValueError('bag row ID out of range')
                    raw.add(bid, int(row))
                domain_bags += 1
            if not domain_bags:
                raise ValueError('at least one registered bag required')
            raw_path = raw.finish()
            notify('raw-bag-sort')
            protect = _Sorter(work / 'protected', 1, memory_bytes, notify)
            for row in protected:
                if type(row) is not int or not 0 <= row < nrows:
                    raise ValueError('protected row ID out of range')
                protect.add(row)
            protected_path = protect.finish()
            events = _Sorter(work / 'events', 3, memory_bytes, notify)
            digest = hashlib.sha256()
            with raw_path.open('rb') as source:
                expected_bid = 0
                for bid, group in itertools.groupby(_records(source, 2), key=lambda r: r[0]):
                    while expected_bid < bid:
                        digest.update(U64.pack(0))
                        expected_bid += 1
                    with _BagPayload(work / 'bag.bin', memory_bytes // 64) as payload:
                        pairs = 0
                        for row, occurrences in itertools.groupby(group, key=lambda r: r[1]):
                            count = sum(1 for _ in occurrences)
                            payload.write(PAIR.pack(row, count))
                            events.add(row, bid, count)
                            pairs += 1
                        digest.update(U64.pack(pairs))
                        payload.update_digest(digest)
                        if payload.stream is not None:
                            notify('bag-spill')
                    expected_bid = bid + 1
                while expected_bid < domain_bags:
                    digest.update(U64.pack(0))
                    expected_bid += 1
            raw_path.unlink()
            event_path = events.finish()
            notify('row-incidence-sort')
            connection = sqlite3.connect(work / 'candidates.sqlite')
            try:
                connection.execute('PRAGMA journal_mode=OFF')
                connection.execute('PRAGMA synchronous=OFF')
                connection.execute('PRAGMA temp_store=FILE')
                connection.execute('PRAGMA mmap_size=0')
                connection.execute(f'PRAGMA cache_size=-{memory_bytes // 8 // 1024}')
                connection.execute('CREATE TABLE groups (gid INTEGER PRIMARY KEY, digest BLOB, '
                                   'length INTEGER, offset INTEGER, first_bag INTEGER, '
                                   'first_count INTEGER, size INTEGER)')
                connection.execute('CREATE INDEX candidate ON groups(digest,length)')
                next_group = 0
                with (output / 'mapping.u64').open('wb') as mapping, event_path.open('rb') as stream, \
                        protected_path.open('rb') as protected_stream, (work / 'signatures.bin').open('w+b') as signatures:
                    protected_reader = iter(_records(protected_stream, 1))
                    current_protected = next(protected_reader, None)

                    def assign(row, offset=0, length=0, signature=None, first=(0, 0)):
                        nonlocal next_group, current_protected
                        while current_protected is not None and current_protected[0] < row:
                            current_protected = next(protected_reader, None)
                        is_protected = current_protected is not None and current_protected[0] == row
                        gid = None
                        if length and not is_protected:
                            for candidate in connection.execute(
                                    'SELECT gid,offset,first_bag,first_count FROM groups WHERE digest=? AND length=?',
                                    (signature, length)):
                                # A length-one signature can be compared exactly from these two integers.
                                equal = ((candidate[2], candidate[3]) == first if length == 1 else
                                         _same_signature(signatures, offset, candidate[1], length * PAIR.size))
                                if equal:
                                    gid = candidate[0]
                                    break
                        if gid is None:
                            gid = next_group
                            next_group += 1
                            connection.execute('INSERT INTO groups VALUES (?,?,?,?,?,?,1)',
                                               (gid, signature if length and not is_protected else None,
                                                length if not is_protected else 0, offset, *first))
                        else:
                            connection.execute('UPDATE groups SET size=size+1 WHERE gid=?', (gid,))
                        mapping.write(U64.pack(gid))
                        if row % 8192 == 8191:
                            connection.commit()

                    expected_row = 0
                    for row, group in itertools.groupby(_records(stream, 3), key=lambda r: r[0]):
                        while expected_row < row:
                            assign(expected_row)
                            expected_row += 1
                        offset, length, first, hashed = signatures.tell(), 0, None, _signature_hash()
                        for _, bag, count in group:
                            pair = PAIR.pack(bag, count)
                            signatures.write(pair)
                            hashed.update(pair)
                            first = first if first is not None else (bag, count)
                            length += 1
                        assign(row, offset, length, hashed.digest(), first)
                        expected_row = row + 1
                    while expected_row < nrows:
                        assign(expected_row)
                        expected_row += 1
                connection.commit()
                notify('exact-signature-grouping')
                with (output / 'sizes.u64').open('wb') as sizes:
                    for size, in connection.execute('SELECT size FROM groups ORDER BY gid'):
                        sizes.write(U64.pack(size))
                multi_rows, multi_groups = connection.execute(
                    'SELECT coalesce(sum(size),0),count(*) FROM groups WHERE size>1').fetchone()
            finally:
                connection.close()
        summary = {'original_rows': nrows, 'group_rows': next_group,
                   'rows_in_nonsingleton_groups': multi_rows, 'nonsingleton_groups': multi_groups,
                   'domain_bags': domain_bags, 'domain_sha256': digest.hexdigest(),
                   'mapping_sha256': file_hash(output / 'mapping.u64'),
                   'contract': 'certified aggregate equivalence in real arithmetic; not original-row byte identity'}
        metadata = {'format': 'external-incidence-v1', 'partition': summary,
                    'managed_memory_bytes': memory_bytes,
                    'files_sha256': {name: file_hash(output / name) for name in ('mapping.u64', 'sizes.u64')}}
        (output / 'external.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return ExternalBagPartition(output)
    except BaseException:
        shutil.rmtree(output)
        raise


class ExternalBagPartition:
    """Validated disk artifact; materialize only when runtime RAM is available."""
    def __init__(self, path):
        self.path = Path(path)
        self.metadata = json.loads((self.path / 'external.json').read_text())
        if self.metadata.get('format') != 'external-incidence-v1':
            raise ValueError('unsupported external incidence format')
        summary = self.metadata['partition']
        for name, count in (('mapping.u64', summary['original_rows']), ('sizes.u64', summary['group_rows'])):
            if (self.path / name).stat().st_size != count * 8 or \
                    file_hash(self.path / name) != self.metadata['files_sha256'][name]:
                raise ValueError('external array size/hash mismatch')

    def summary(self):
        return dict(self.metadata['partition'])

    def materialize(self):
        """Explicit O(rows + groups) runtime loading; not part of bounded build."""
        from .cotraining_groups import BagPartition
        s = self.summary()
        partition = BagPartition(np.fromfile(self.path / 'mapping.u64', dtype='<u8'),
                                 s['domain_sha256'], s['domain_bags'])
        if partition.summary() != s:
            raise ValueError('external partition semantics mismatch')
        return partition

    def save_npz(self, output):
        """Write the existing BagPartition reader format without loading its map."""
        output = Path(output)
        with output.open('xb') as target:
            try:
                with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED) as archive:
                    with archive.open('mapping.npy', 'w', force_zip64=True) as member:
                        np.lib.format.write_array_header_1_0(member, {
                            'descr': '<u8', 'fortran_order': False, 'shape': (self.summary()['original_rows'],)})
                        with (self.path / 'mapping.u64').open('rb') as source:
                            shutil.copyfileobj(source, member, 65536)
                    for name, value in (('domain_sha256', self.summary()['domain_sha256']),
                                        ('domain_bags', self.summary()['domain_bags'])):
                        with archive.open(name + '.npy', 'w') as member:
                            np.save(member, np.asarray(value), allow_pickle=False)
            except BaseException:
                output.unlink()
                raise

    def project_file(self, source, dim, output, *, memory_bytes=64 << 20, progress=None):
        """Ordered float64 accumulation, matching BagPartition.means FP32 bytes."""
        memory_bytes = _budget(memory_bytes)
        if type(dim) is not int or not 0 < dim <= memory_bytes // 64 // 8:
            raise ValueError('dimension must fit the managed row buffer')
        source, output = Path(source), Path(output)
        s = self.summary()
        if source.stat().st_size != s['original_rows'] * dim * 4:
            raise ValueError('source must contain matching little-endian FP32 rows')
        if output.exists():
            raise FileExistsError(output)
        batch_rows = max(1, memory_bytes // 64 // (dim * 4))
        capacity = max(1, memory_bytes // 4 // (dim * 8 + 512))
        cache, evictions, reads = OrderedDict(), 0, 0
        digest = hashlib.sha256()
        with tempfile.TemporaryDirectory(prefix='.means-', dir=output.parent) as temporary:
            work = Path(temporary)
            with (work / 'sums.f64').open('w+b', buffering=0) as sums, source.open('rb') as vectors, \
                    (self.path / 'mapping.u64').open('rb') as mapping:
                sums.truncate(s['group_rows'] * dim * 8)

                def write_sum(gid, value):
                    payload = value.astype('<f8', copy=False).tobytes()
                    if os.pwrite(sums.fileno(), payload, gid * dim * 8) != len(payload):
                        raise OSError('short accumulator write')

                for block in iter(lambda: vectors.read(batch_rows * dim * 4), b''):
                    digest.update(block)
                    rows = np.frombuffer(block, dtype='<f4').reshape(-1, dim)
                    ids = mapping.read(len(rows) * 8)
                    if len(ids) != len(rows) * 8 or not np.isfinite(rows).all():
                        raise ValueError('invalid source rows or mapping')
                    for gid, row in zip(np.frombuffer(ids, dtype='<u8'), rows):
                        gid = int(gid)
                        if not 0 <= gid < s['group_rows']:
                            raise ValueError('group ID out of range')
                        value = cache.pop(gid, None)
                        if value is None:
                            if len(cache) >= capacity:
                                old_gid, old_value = cache.popitem(last=False)
                                write_sum(old_gid, old_value)
                                evictions += 1
                            payload = os.pread(sums.fileno(), dim * 8, gid * dim * 8)
                            if len(payload) != dim * 8:
                                raise ValueError('truncated accumulator')
                            value = np.frombuffer(payload, dtype='<f8').copy()
                            reads += 1
                        value += row
                        cache[gid] = value
                if mapping.read(1):
                    raise ValueError('mapping length mismatch')
                for gid, value in cache.items():
                    write_sum(gid, value)
                cache.clear()
                means_path = work / 'means.f32'
                with means_path.open('wb') as means, (self.path / 'sizes.u64').open('rb') as sizes:
                    for gid, (size,) in enumerate(_records(sizes, 1)):
                        if not size:
                            raise ValueError('empty group')
                        value = np.frombuffer(os.pread(sums.fileno(), dim * 8, gid * dim * 8), dtype='<f8')
                        means.write((value / size).astype('<f4').tobytes())
                if progress:
                    progress('means-written', output.parent)
            if means_path.stat().st_size != s['group_rows'] * dim * 4:
                raise ValueError('projected row count mismatch')
            # Publish without overwriting a concurrently-created output.
            os.link(means_path, output)
        return {'source_sha256': digest.hexdigest(), 'means_sha256': file_hash(output),
                'group_rows': s['group_rows'], 'dim': dim, 'managed_memory_bytes': memory_bytes,
                'cache_capacity_rows': capacity, 'accumulator_reads': reads, 'evictions': evictions,
                'temporary_accumulator_bytes': s['group_rows'] * dim * 8,
                'contract': 'ordered float64 group means rounded to FP32; source rows change'}


def iter_bag_files(ids_path, offsets_path):
    """Single-pass little-endian uint64 CSR bags, using fixed read buffers."""
    ids_path, offsets_path = Path(ids_path), Path(offsets_path)
    if ids_path.stat().st_size % 8 or offsets_path.stat().st_size % 8:
        raise ValueError('complete uint64 CSR records required')
    with ids_path.open('rb') as ids, offsets_path.open('rb') as offsets:
        reader = iter(_records(offsets, 1))
        start = next(reader, None)
        if start != (0,):
            raise ValueError('CSR offsets must start at zero')
        start = 0
        for end, in reader:
            if end < start or end > ids_path.stat().st_size // 8:
                raise ValueError('CSR offsets out of range')

            consumed = [False]

            def bag(stop=end):
                while ids.tell() < stop * 8:
                    block = ids.read(min(256, stop - ids.tell() // 8) * 8)
                    if not block or len(block) % 8:
                        raise ValueError('truncated CSR bag')
                    yield from (value for value, in U64.iter_unpack(block))
                consumed[0] = True

            yield bag()
            if not consumed[0] or ids.tell() != end * 8:
                raise ValueError('consume each CSR bag before advancing')
            start = end
        if start * 8 != ids_path.stat().st_size:
            raise ValueError('CSR offsets must cover all IDs')


def iter_protected_ids(path):
    """Stream independently addressed IDs from a raw little-endian uint64 file."""
    with Path(path).open('rb') as stream:
        yield from (row for row, in _records(stream, 1))
