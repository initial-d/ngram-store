"""Deterministic bounded release capsules for the complete numeric evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def capsule(number, names, files, evidence, output, release_url):
    name = f'memory-application-raw-{number:03d}.tar.gz'; path = output / name
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed, tarfile.open(fileobj=compressed, mode='w') as archive:
        for member in names:
            info = tarfile.TarInfo(member); info.size = files[member]['bytes']; info.mode = 0o644; info.mtime = 0
            with (evidence / member).open('rb') as stream:
                archive.addfile(info, stream)
    return {'name': name, 'url': release_url.rstrip('/') + '/' + name,
            'bytes': path.stat().st_size, 'sha256': digest(path), 'members': names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release-url', required=True)
    parser.add_argument('--maximum-member-bytes', type=int, default=400 << 20)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 32:
        parser.error('Capsule workers must be within 1–32')
    args.output.mkdir(parents=True, exist_ok=False)
    files = {str(p.relative_to(args.evidence)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in sorted(args.evidence.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    (args.evidence / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'status': 'complete', 'files': files}, indent=2, sort_keys=True) + '\n')
    files['manifest.json'] = {'bytes': (args.evidence / 'manifest.json').stat().st_size,
                              'sha256': digest(args.evidence / 'manifest.json')}
    groups = []; current = []; size = 0
    for name, entry in files.items():
        if current and size + entry['bytes'] > args.maximum_member_bytes:
            groups.append(current); current = []; size = 0
        if entry['bytes'] > args.maximum_member_bytes:
            raise ValueError('Single member exceeds release capsule budget: ' + name)
        current.append(name); size += entry['bytes']
    if current:
        groups.append(current)
    assets = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for asset in pool.map(lambda item: capsule(item[0], item[1], files, args.evidence, args.output, args.release_url), enumerate(groups, 1)):
            assets.append(asset)
            print(json.dumps({'capsule': asset['name'], 'members': len(asset['members']), 'bytes': asset['bytes']}), flush=True)
    (args.output / 'raw-manifest.json').write_text(json.dumps({'schema_version': 1, 'files': files, 'assets': assets}, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
