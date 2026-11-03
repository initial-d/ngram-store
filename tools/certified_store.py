"""From FP32/CSR inputs to a ready-to-query certified native bundle."""
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

from .external_cotraining import fit_external, iter_bag_files, iter_protected_ids


def compact_native(source, dim, bag_ids, bag_offsets, output, *, memory_bytes=64 << 20,
                   protected_ids=None, block_rows=8):
    """Build with disk-backed buffers; the resulting serving bundle has 3 files.

    Each query still goes through CertifiedBagRows, whose resident runtime map
    is not bounded by memory_bytes. No native C state is opened during building.
    """
    source, output = Path(source), Path(output)
    if type(dim) is not int or dim <= 0:
        raise ValueError('positive dimension required')
    nrows, tail = divmod(source.stat().st_size, dim * 4)
    if not nrows or tail:
        raise ValueError('complete nonempty FP32 rows required')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    peak_disk = 0

    def disk(path):
        size = 0
        with os.scandir(path) as entries:
            for entry in entries:
                size += disk(entry.path) if entry.is_dir(follow_symlinks=False) else entry.stat().st_size
        return size

    def progress(stage, path):
        nonlocal peak_disk
        # Sample stage boundaries. Do not build a resident list of temporary runs.
        if stage not in ('sorted-run', 'merged-run', 'bag-spill'):
            peak_disk = max(peak_disk, disk(output))

    try:
        partition = fit_external(iter_bag_files(bag_ids, bag_offsets), nrows, output / 'partition',
            memory_bytes=memory_bytes,
            protected=iter_protected_ids(protected_ids) if protected_ids else (), progress=progress)
        partition_seconds = time.perf_counter() - started
        before = time.perf_counter()
        projection = partition.project_file(source, dim, output / 'means.f32',
                                            memory_bytes=memory_bytes, progress=progress)
        projection_seconds = time.perf_counter() - before
        before = time.perf_counter()
        storage = partition.export_native(output / 'means.f32', dim, output / 'table.men',
            memory_bytes=memory_bytes, block_rows=block_rows,
            expected_means_sha256=projection['means_sha256'])
        partition.save_npz(output / 'domain.npz')
        peak_disk = max(peak_disk, disk(output))
        native_seconds = time.perf_counter() - before
        manifest = {'schema_version': 1, 'name': 'certified-groups', 'engine': 'NGramStore',
            'shape': [nrows, dim], 'dim': dim, 'dtype': 'little-endian float32',
            'source_row_byte_preserving': False, 'original_bytes_preserved': False,
            'native_decodes_declared_rows_byte_exactly': True,
            'partition': partition.summary(), 'projection': projection, 'storage': storage,
            'artifact_sha256': storage['artifact_sha256'],
            'source_sha256': projection['source_sha256'],
            'construction': {'partition_seconds': partition_seconds,
                'projection_seconds': projection_seconds, 'native_export_seconds': native_seconds,
                'seconds_before_cleanup': time.perf_counter() - started,
                'sampled_workspace_logical_bytes': peak_disk,
                'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                    (1 if sys.platform == 'darwin' else 1024)},
            'contract': 'Certified complete bag sums/means in real arithmetic; individual rows change.',
            'limitations': ['Managed buffers do not cap process RSS or OS page cache.',
                'Runtime maps and query compilation have separate resident memory costs.',
                'Native raw lookup does not enforce the certificate; use query-bags/CertifiedBagRows.',
                'No original-row byte identity or universal model-quality claim.']}
        shutil.rmtree(output / 'partition')
        (output / 'means.f32').unlink()
        manifest['serving_bytes_without_manifest'] = sum((output / n).stat().st_size
                                                       for n in ('table.men', 'domain.npz'))
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return {**manifest, 'complete_serving_bytes': disk(output)}
    except BaseException:
        shutil.rmtree(output)
        raise
