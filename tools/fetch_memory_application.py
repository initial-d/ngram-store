"""Fetch every frozen numeric capsule and verify its complete member inventory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def fetch(asset, files, output):
    names = asset['members']
    if all((output / name).is_file() and (output / name).stat().st_size == files[name]['bytes']
           and digest(output / name) == files[name]['sha256'] for name in names):
        return asset['name'] + ': all existing members verified'
    if Path(asset['name']).name != asset['name']:
        raise ValueError('Capsule name must be a basename')
    temporary = output / (asset['name'] + '.download')
    for attempt in range(3):
        try:
            total = 0; checksum = hashlib.sha256()
            with urllib.request.urlopen(asset['url'], timeout=60) as response, temporary.open('wb') as target:
                for block in iter(lambda: response.read(8 << 20), b''):
                    total += len(block)
                    if total > asset['bytes']:
                        raise ValueError('Downloaded archive exceeds the frozen byte count')
                    checksum.update(block); target.write(block)
            if total != asset['bytes'] or checksum.hexdigest() != asset['sha256']:
                raise ValueError('Downloaded capsule checksum mismatch')
            break
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
    with tarfile.open(temporary, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(names) or {m.name for m in members} != set(names):
            raise ValueError('Capsule inventory differs from the complete frozen manifest')
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or member.size != files[member.name]['bytes']:
                raise ValueError('Invalid capsule member')
        for member in members:
            target = output / member.name; target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_suffix(target.suffix + '.pending'); checksum = hashlib.sha256()
            with archive.extractfile(member) as source, pending.open('wb') as stream:
                for block in iter(lambda: source.read(8 << 20), b''):
                    checksum.update(block); stream.write(block)
            if checksum.hexdigest() != files[member.name]['sha256']:
                pending.unlink()
                raise ValueError('Capsule member checksum mismatch')
            pending.replace(target)
    temporary.unlink()
    return asset['name'] + ': all members downloaded and verified'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); manifest = json.loads(args.manifest.read_text())
    names = [name for asset in manifest['assets'] for name in asset['members']]
    if len(names) != len(set(names)) or set(names) != set(manifest['files']):
        raise ValueError('Missing or duplicate frozen capsule coverage')
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(lambda a: fetch(a, manifest['files'], args.output), manifest['assets']):
            print(result, flush=True)


if __name__ == '__main__':
    main()
