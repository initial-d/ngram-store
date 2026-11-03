"""Fresh CPU construction workers; native/runtime checks are separate."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time

import numpy as np

from .certified_store import compact_native
from .cotraining_groups import BagPartition, CertifiedBagRows
from .external_cotraining import file_hash, iter_bag_files
from .native_row_store import pack_native


def environment():
    return {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'command': shlex.join([sys.executable, '-m', 'tools.benchmark_certified_serving', *sys.argv[1:]]),
        'python': sys.version, 'numpy': np.__version__, 'os': platform.platform(),
        'source_files_sha256': {n: file_hash(Path(__file__).with_name(n)) for n in
            ('benchmark_certified_serving.py', 'certified_store.py', 'external_cotraining.py',
             'cotraining_groups.py', 'native_row_store.py', 'certified_torch.py')},
        'device': 'cpu'}


def worker(job_path):
    job = json.loads(Path(job_path).read_text())
    inputs = json.loads(Path(job['input']).read_text())
    for key in ('source', 'ids', 'offsets'):
        if file_hash(inputs[key]) != inputs[key + '_sha256']:
            raise ValueError('input changed: ' + key)
    out = Path(job['output'])
    started = time.perf_counter()
    if job['method'] == 'external':
        result = compact_native(inputs['source'], inputs['dim'], inputs['ids'], inputs['offsets'],
                                out, memory_bytes=64 << 20)
    else:
        out.mkdir(parents=True, exist_ok=False)
        partition = BagPartition.fit(iter_bag_files(inputs['ids'], inputs['offsets']), inputs['rows'])
        source = np.memmap(inputs['source'], mode='r', dtype='<f4', shape=(inputs['rows'], inputs['dim']))
        means = partition.means(source)
        stage = out / 'staging.f32'
        with stage.open('wb') as stream:
            for begin in range(0, len(source), 4096):
                stream.write(means[partition.mapping[begin:begin + 4096]].tobytes())
        storage = pack_native(stage, out / 'table.men', inputs['dim'] * 4, block_rows=8,
            representatives=partition.representatives().tolist(), deduplicate=False, predictor='xor', compress=False)
        staging_bytes = stage.stat().st_size
        stage.unlink()
        partition.save(out / 'domain.npz')
        result = {'schema_version': 1, 'name': 'certified-groups', 'shape': [inputs['rows'], inputs['dim']],
            'source_row_byte_preserving': False, 'partition': partition.summary(), 'storage': storage,
            'artifact_sha256': file_hash(out / 'table.men'), 'source_sha256': inputs['source_sha256'],
            'projection': {'means_sha256': hashlib.sha256(means.tobytes()).hexdigest()},
            'legacy_expanded_staging_bytes': staging_bytes}
        (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
        result['complete_serving_bytes'] = sum(p.stat().st_size for p in out.iterdir())
    report = {**environment(), 'input': inputs, 'job': job, 'result': result, 'status': 'completed',
        'total_seconds': time.perf_counter() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'native_table_sha256': file_hash(out / 'table.men'), 'domain_file_sha256': file_hash(out / 'domain.npz')}
    if result['source_sha256'] != inputs['source_sha256'] or file_hash(inputs['source']) != inputs['source_sha256']:
        raise ValueError('source changed during construction')
    Path(job['report']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'method': job['method'], 'seconds': report['total_seconds'],
                      'peak_rss_bytes': report['peak_rss_bytes']}), flush=True)


def verify(cfg):
    from .certified_torch import CertifiedEmbeddingBag
    import torch
    torch.set_num_threads(1)
    inputs = json.loads(cfg.input.read_text())
    bags = json.loads(cfg.bags.read_text())
    offsets = np.asarray([0, *np.cumsum([len(b) for b in bags])], dtype='<u8')
    flat = np.asarray([r for b in bags for r in b], dtype='<u8')
    trace = hashlib.sha256(flat.tobytes() + offsets.tobytes()).hexdigest()
    if trace != 'cedaa2f0ac19ceedfea4eda07139204d8c179d84e3f33e213811168cd93f83fe':
        raise ValueError('native trace changed')
    started = time.perf_counter()
    outputs = {}
    if file_hash(cfg.store / 'table.men') != file_hash(cfg.previous / 'table.men'):
        raise ValueError('complete native table differs from the earlier raw artifact')
    with CertifiedBagRows(cfg.store, cfg.native_library) as new, \
            CertifiedBagRows(cfg.previous, cfg.native_library) as old:
        if new.partition.summary() != old.partition.summary():
            raise ValueError('previous native partition differs')
        a, b = new.compile(bags), old.compile(bags)
        for name in ('ids', 'weights', 'offsets', 'denominators'):
            if getattr(a, name).tobytes() != getattr(b, name).tobytes():
                raise ValueError('native plan arrays differ')
        for mean in (False, True):
            x, y = new.evaluate(a, mean=mean), old.evaluate(b, mean=mean)
            if x.tobytes() != y.tobytes():
                raise ValueError('native output differs')
            outputs['mean' if mean else 'sum'] = hashlib.sha256(x.tobytes()).hexdigest()
        row = next(i for i, g in enumerate(new.partition.mapping) if new.partition.sizes[g] > 1)
        try:
            new.aggregate([[row]])
        except ValueError:
            pass
        else:
            raise ValueError('incomplete group accepted')
        native = new.evaluate(a, mean=True)
    with CertifiedEmbeddingBag(cfg.store, mode='mean', include_last_offset=True,
                               native_library=cfg.native_library) as adapter:
        actual = adapter(torch.from_numpy(flat.astype(np.int64)), torch.from_numpy(offsets.astype(np.int64)))
        if actual.numpy().tobytes() != native.tobytes():
            raise ValueError('PyTorch adapter differs from native output')
    source = np.memmap(inputs['source'], mode='r', dtype='<f4', shape=(inputs['rows'], inputs['dim']))
    reference = torch.nn.functional.embedding_bag(torch.from_numpy(flat.astype(np.int64)),
        torch.from_numpy(source.copy()), offsets=torch.from_numpy(offsets.astype(np.int64)),
        mode='mean', include_last_offset=True).numpy()
    error = native.astype(np.float64) - reference
    report = {**environment(), 'status': 'completed', 'torch': torch.__version__, 'trace_sha256': trace,
        'native_library_sha256': file_hash(cfg.native_library), 'native_outputs_sha256': outputs,
        'plan': a.summary(), 'all_plan_arrays_identical': True, 'incomplete_group_rejected': row,
        'pytorch_adapter_identical_to_native_bytes': True,
        'source_pytorch_fp32_comparison': {'maximum_absolute_error': float(np.abs(error).max()),
            'global_relative_l2': float(np.linalg.norm(error) / max(np.linalg.norm(reference.astype(np.float64)), 1e-30)),
            'byte_identical': native.tobytes() == reference.tobytes()},
        'seconds': time.perf_counter() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'limitations': ['Separate resident/native/PyTorch verifier, excluded from build RSS.',
            'PyTorch source-copy/reference reduction is measured for integration, not serving latency.',
            'No training, downstream or LLM-quality evidence; projection/native rounding can change FP32 outputs.']}
    cfg.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'native_and_pytorch_verified': True, 'peak_rss_bytes': report['peak_rss_bytes']}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--input', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--store', type=Path)
    p.add_argument('--previous', type=Path)
    p.add_argument('--bags', type=Path)
    p.add_argument('--native-library', type=Path)
    p.add_argument('--output', type=Path)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU-only: empty CUDA_VISIBLE_DEVICES required')
    if cfg.worker:
        worker(cfg.worker)
    elif cfg.verify:
        verify(cfg)
    else:
        cfg.output_dir.mkdir(parents=True, exist_ok=False)
        report = {**environment(), 'cpu': subprocess.check_output(['lscpu', '-J'], text=True),
            'runs': [], 'failures': [], 'limitations': ['Shared warm CPU/filesystem, single-threaded fresh build workers.',
                'No C/runtime verification in either build worker; measured separately.',
                'Managed buffers do not bound RSS or OS page cache; full application keys/output weights excluded.']}
        for method in ('legacy', 'external'):
            job = {'method': method, 'input': str(cfg.input), 'output': str(cfg.output_dir / method),
                   'report': str(cfg.output_dir / (method + '.json'))}
            job_path = cfg.output_dir / (method + '-job.json')
            job_path.write_text(json.dumps(job, indent=2) + '\n')
            with (cfg.output_dir / (method + '.log')).open('w') as log:
                r = subprocess.run([sys.executable, '-m', 'tools.benchmark_certified_serving',
                    '--worker', str(job_path)], stdout=log, stderr=subprocess.STDOUT)
            if r.returncode:
                report['failures'].append({'method': method, 'exit_code': r.returncode})
            else:
                report['runs'].append(json.loads(Path(job['report']).read_text()))
            (cfg.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if report['failures']:
            raise SystemExit(1)
        a, b = report['runs']
        if a['result']['partition'] != b['result']['partition'] or \
                a['native_table_sha256'] != b['native_table_sha256']:
            raise ValueError('complete native tables or partitions differ')
        compared = 0
        with (cfg.output_dir / 'legacy/table.men').open('rb') as x, (cfg.output_dir / 'external/table.men').open('rb') as y:
            while True:
                left, right = x.read(65536), y.read(65536)
                if left != right:
                    raise ValueError('native table bytes differ')
                if not left:
                    break
                compared += len(left)
        report.update(status='completed', all_native_table_bytes_identical=compared,
            rss_reduction_factor=a['peak_rss_bytes'] / b['peak_rss_bytes'],
            build_time_factor=b['total_seconds'] / a['total_seconds'])
        (cfg.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
