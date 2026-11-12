"""Read-only validation of completed jobs before a finite queue resumes."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def completed_task(root, task, protocol_sha):
    root = Path(root)
    model, edits = task['model'], task['edits']
    prepared = root / f'prepared-{model}-{edits}' / 'prepare.json'
    meta = json.loads(prepared.read_text())
    if task['action'] == 'cache':
        stem = root / f'cache-{model}-{edits}' / f"shard-{task['shard']}"
        report_path, raw = stem.with_suffix('.json'), stem.with_suffix('.f32')
        if not report_path.exists():
            if raw.exists():
                raise ValueError('unverified partial cache must be preserved before retry: ' + str(raw))
            return None
        report = json.loads(report_path.read_text())
        if report['status'] != 'complete' or report['shard'] != task['shard'] or report['shards'] != 6:
            raise ValueError('cache completion manifest does not match task')
        if report['prepared_sha256'] != sha(prepared) or report['batch_size'] != meta['batch_size']:
            raise ValueError('cache prepared data changed')
        if raw.stat().st_size != report['bytes'] or sha(raw) != report['sha256']:
            raise ValueError('cache bytes changed')
        if report['tokens'] * report['hidden_size'] * 4 != report['bytes']:
            raise ValueError('cache tensor dimensions changed')
        entries = sorted(report['sequences'].values())
        offset = 0
        for start, length in entries:
            if start != offset or length <= 0:
                raise ValueError('cache sequence ranges are not contiguous')
            offset += length
        if offset != report['tokens']:
            raise ValueError('cache token coverage changed')
        batches = json.loads((prepared.parent / 'batches.json').read_text())
        if sha(prepared.parent / 'batches.json') != meta['batches_sha256']:
            raise ValueError('cache batch definition changed')
        expected = {str(i) for batch in batches[task['shard']::6] for i in batch}
        if set(report['sequences']) != expected:
            raise ValueError('cache sequence coverage changed')
        reference = json.loads(next((root / f'cache-{model}-1024').glob('shard-*.json')).read_text())
        if report['frozen_weight_sha256'] != reference['frozen_weight_sha256']:
            raise ValueError('cache immutable prefix differs from validated small cache')
    else:
        folder = root / 'runs' / task['name']
        report_path = folder / 'report.json'
        if not folder.exists():
            return None
        if not report_path.exists():
            raise ValueError('partial run must be preserved before retry: ' + str(folder))
        report = json.loads(report_path.read_text())
        if report['status'] != 'complete' or report['protocol_sha256'] != protocol_sha:
            raise ValueError('run incomplete or protocol changed: ' + str(folder))
        if task['action'] == 'retrieval':
            for k, candidate in report['candidates'].items():
                if sha(folder / f'case-scores-k{k}.json') != candidate['case_scores_sha256']:
                    raise ValueError('retrieval scores changed')
        else:
            if report['prepared_sha256'] != sha(prepared):
                raise ValueError('run prepared data changed')
            artifacts = report.get('score_artifacts', {'case-scores.json': {'sha256': report.get('case_scores_sha256')}})
            for name, info in artifacts.items():
                if sha(folder / name) != info['sha256']:
                    raise ValueError('run scores changed')
        if task['action'] == 'train':
            if report['seed'] != task['seed'] or report['variant'] != task['variant']:
                raise ValueError('run variant/seed mismatch')
            delta = folder / 'delta.safetensors'
            if delta.stat().st_size != report['delta']['bytes'] or sha(delta) != report['delta']['sha256']:
                raise ValueError('training delta changed')
    return {**task, 'reused': True, 'completion_report': str(report_path),
            'completion_report_sha256': sha(report_path), 'exit_code': 0,
            'recovery_note': 'Validated completed artifact; original process outcome remains in previous queue history'}
