"""Package audited deployment reports and numeric predictions outside Git.

Original bytes and commands are retained so audit hashes remain verifiable.
The allowlist excludes trained tables, keys, corpora and model checkpoints.
The standard-library verifier checks every capsule, member and audit link;
prediction arithmetic can additionally be checked with the NumPy audit tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


HASH_FIELDS = ('report_sha256', 'gpu_report_sha256', 'row_packet_report_sha256',
               'canonical_cache_report_sha256', 'frozen_selection_sha256')
AUDITS = ('application-results.json', 'application-summary.json',
          'application-audit-exit.json', 'qwen-results.json', 'qwen-summary.json',
          'results.json', 'summary.json')


def digest_file(path):
    with Path(path).open('rb') as stream:
        return digest_stream(stream)


def digest_stream(stream):
    result = hashlib.sha256()
    while True:
        block = stream.read(1 << 20)
        if not block:
            return result.hexdigest()
        result.update(block)


def load(path):
    return json.loads(Path(path).read_text())


def package(root, output, limit=400 << 20):
    root = root.resolve()
    if limit <= 0:
        raise ValueError('positive capsule size required')
    paths = {}
    links = {}

    def take(path, expected=None, predictions=False):
        path = Path(path)
        # Lexical and checkpoint files never enter through a broad directory copy.
        name = str(path.relative_to(root))
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise ValueError('regular task-local evidence required: ' + name)
        if path.suffix not in ('.json', '.txt', '.py', '.c', '.cc', '.cpp', '.h', '.hpp', '.npz'):
            raise ValueError('not allowlisted: ' + name)
        if path.suffix == '.npz':
            if not predictions:
                raise ValueError('only declared prediction payloads allowed')
            with zipfile.ZipFile(path) as archive:
                if set(archive.namelist()) != {'logits.npy', 'labels.npy', 'nll.npy'}:
                    raise ValueError('unexpected prediction arrays: ' + name)
        sha = digest_file(path)
        if expected is not None and sha != expected:
            raise ValueError('frozen evidence changed: ' + name)
        paths[name] = {'sha256': sha, 'bytes': path.stat().st_size}
        if expected is not None:
            links[str(path)] = {'member': name, 'sha256': expected}
        return name

    for name in AUDITS:
        take(root / 'cpu-audit' / name)
    if load(root / 'cpu-audit/application-audit-exit.json')['exit_code'] != 0:
        raise ValueError('successful terminal application audit required')
    completed = root / 'application-frontier/shared-recovery-takeover/all-results-complete.json'
    if load(completed)['status'] != 'complete':
        raise ValueError('terminal completion required')
    take(completed)
    for name in ('application-results.json', 'qwen-results.json', 'results.json'):
        value = load(root / 'cpu-audit' / name)
        if not value['status'].startswith('audited-complete-'):
            raise ValueError('complete audit required')
        for field in HASH_FIELDS:
            for path, expected in value.get(field, {}).items():
                take(path, expected)
                report = load(path)
                if 'payload_sha256' in report:
                    take(Path(path).with_suffix('.npz'), report['payload_sha256'], True)
    app = root / 'application-frontier'
    take(app / 'source-selection.json', load(root / 'cpu-audit/application-results.json')['source_selection_sha256'])
    patterns = (
        '*.json', 'source-jobs/*.json', 'source-quality/*.json',
        'models/*/built.json', 'models/*/selection.json', 'models/*/jobs/*.json',
        'models/*/quality/*-calibration.json', 'models/*/parallel-structure/*/complete.json',
        'native-cache/*/complete.json',
    )
    for pattern in patterns:
        for path in sorted(app.glob(pattern)):
            take(path)
            value = load(path)
            if 'payload_sha256' in value:
                take(path.with_suffix('.npz'), value['payload_sha256'], True)
    for directory, patterns in (
        ('native-serving', ('*/*.json',)),
        ('native-layouts', ('*/*.json', '*/jobs/*.json')),
        ('row-packets/prepared', ('*/*.json', '*/jobs/*.json')),
        ('row-packets/results', ('*/*.json',)),
    ):
        for pattern in patterns:
            for path in sorted((root / directory).glob(pattern)):
                take(path)
    for path in sorted(root.glob('*.json')):
        take(path)
    for name in ('cpuinfo.txt', 'meminfo.txt', 'nvidia-smi.txt', 'physical-core-isolation.json'):
        take(root / 'inventory' / name)
    for runtime in sorted(root.glob('runtime*')):
        for folder in ('tools', 'native', 'src', 'meft_memory', 'ngram_store'):
            for path in sorted((runtime / folder).rglob('*')):
                if path.is_file() and not path.is_symlink() and path.suffix in ('.py', '.c', '.cc', '.cpp', '.h', '.hpp'):
                    take(path)
    for path in sorted((root / 'cpu-audit/tools').glob('*.py')):
        take(path)

    output.mkdir(parents=True, exist_ok=False)
    capsules = []; group = []; size = 0

    def emit(names):
        filename = 'controlled-deployment-raw-%03d.tar.gz' % (len(capsules) + 1)
        destination = output / filename
        with tarfile.open(destination, 'w:gz', compresslevel=1) as archive:
            for name in names:
                archive.add(root / name, arcname=name, recursive=False)
        capsules.append({'name': filename, 'bytes': destination.stat().st_size,
                         'sha256': digest_file(destination), 'members': names})

    for name, record in sorted(paths.items()):
        if group and size + record['bytes'] > limit:
            emit(group); group = []; size = 0
        group.append(name); size += record['bytes']
    if group:
        emit(group)
    manifest = {'version': 1, 'original_root': str(root), 'members': paths,
                'audit_links': links, 'capsules': capsules,
                'scope': 'Numeric public-classifier predictions, raw query/decode samples, frozen jobs/selections, failures and runtime source. No trained rows, lexical keys, corpora or checkpoints. Hash verification does not rerun model training or prove source-row identities without the excluded trained fixtures.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def verify(directory):
    manifest = load(directory / 'manifest.json'); seen = set()
    for capsule in manifest['capsules']:
        path = directory / capsule['name']
        if path.stat().st_size != capsule['bytes'] or digest_file(path) != capsule['sha256']:
            raise ValueError('capsule changed: ' + path.name)
        names = []
        with tarfile.open(path, 'r|gz') as archive:
            for member in archive:
                name = member.name
                if not member.isfile() or name in seen or name not in manifest['members']:
                    raise ValueError('unexpected or duplicate archive member: ' + name)
                record = manifest['members'][name]
                if member.size != record['bytes'] or digest_stream(archive.extractfile(member)) != record['sha256']:
                    raise ValueError('member changed: ' + name)
                seen.add(name); names.append(name)
        if names != capsule['members']:
            raise ValueError('capsule member schedule changed')
    if seen != set(manifest['members']):
        raise ValueError('missing evidence members')
    for record in manifest['audit_links'].values():
        if manifest['members'][record['member']]['sha256'] != record['sha256']:
            raise ValueError('audit hash link changed')
    return {'status': 'verified', 'capsules': len(manifest['capsules']),
            'members': len(seen), 'audit_links': len(manifest['audit_links'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify)))
    elif args.root and args.output:
        manifest = package(args.root, args.output)
        print(json.dumps({'status': 'packaged', 'members': len(manifest['members']),
                          'capsules': len(manifest['capsules'])}))
    else:
        parser.error('--root and --output, or --verify, required')


if __name__ == '__main__':
    main()
