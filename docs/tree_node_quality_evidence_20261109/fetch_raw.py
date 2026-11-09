"""Fetch complete public numeric evidence; verify each archive and every member."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
TARGET = ROOT.parents[1] / 'artifacts/tree-node-quality-20261109/raw'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fetch(asset, files):
    names = asset['members']
    if all((TARGET / n).is_file() and (TARGET / n).stat().st_size == files[n]['bytes']
           and digest(TARGET / n) == files[n]['sha256'] for n in names):
        return asset['name'] + ': verified existing complete members'
    TARGET.mkdir(parents=True, exist_ok=True)
    temporary = TARGET / (asset['name'] + '.download')
    for attempt in range(3):
        try:
            h = hashlib.sha256(); size = 0
            with urllib.request.urlopen(asset['url'], timeout=60) as response, temporary.open('wb') as out:
                for block in iter(lambda: response.read(8 << 20), b''):
                    out.write(block); h.update(block); size += len(block)
                    assert size <= asset['bytes'], 'asset exceeds declared size'
            assert size == asset['bytes'] and h.hexdigest() == asset['sha256'], 'archive checksum mismatch'
            break
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
    with tarfile.open(temporary, 'r:gz') as archive:
        members = archive.getmembers()
        assert {m.name for m in members} == set(names) and len(members) == len(names)
        for member in members:
            assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
            assert member.size == files[member.name]['bytes']
            target = TARGET / member.name; target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_suffix(target.suffix + '.pending')
            h = hashlib.sha256()
            with archive.extractfile(member) as source, pending.open('wb') as out:
                for b in iter(lambda: source.read(8 << 20), b''):
                    h.update(b); out.write(b)
            assert h.hexdigest() == files[member.name]['sha256'], 'member checksum mismatch'
            pending.replace(target)
    temporary.unlink()
    return asset['name'] + ': all members fetched and verified'


def main():
    manifest = json.loads((ROOT / 'raw-manifest.json').read_text())
    names = [n for a in manifest['assets'] for n in a['members']]
    assert len(names) == len(set(names)) and set(names) == set(manifest['files'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda a: fetch(a, manifest['files']), manifest['assets']):
            print(result, flush=True)


if __name__ == '__main__':
    main()
