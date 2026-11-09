"""Deterministic bounded release capsules for the complete numeric evidence."""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release-url', required=True)
    parser.add_argument('--maximum-member-bytes', type=int, default=400 << 20)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
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
    for number, names in enumerate(groups, 1):
        name = f'memory-application-raw-{number:03d}.tar.gz'; path = args.output / name
        with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed, tarfile.open(fileobj=compressed, mode='w') as archive:
            for member in names:
                info = tarfile.TarInfo(member); info.size = files[member]['bytes']; info.mode = 0o644; info.mtime = 0
                with (args.evidence / member).open('rb') as stream:
                    archive.addfile(info, stream)
        assets.append({'name': name, 'url': args.release_url.rstrip('/') + '/' + name,
                       'bytes': path.stat().st_size, 'sha256': digest(path), 'members': names})
        print(json.dumps({'capsule': name, 'members': len(names), 'bytes': path.stat().st_size}), flush=True)
    (args.output / 'raw-manifest.json').write_text(json.dumps({'schema_version': 1, 'files': files, 'assets': assets}, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
