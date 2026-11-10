"""Concurrent compression and batched Git reads preserve raw capsule content."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.package_memory_application import capsule, digest


class CapsulePreparationTests(unittest.TestCase):
    def test_serial_and_parallel_capsules_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); evidence = root / 'data'; a = root / 'serial'; b = root / 'parallel'
            for path in (evidence, a, b): path.mkdir()
            files = {}
            for index in range(4):
                path = evidence / f'{index}.json'; path.write_bytes((b'{"numeric":123}\n' + bytes([index])) * 1000)
                files[path.name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
            groups = [[name] for name in files]
            serial = [capsule(i + 1, names, files, evidence, a, 'https://example.invalid/raw') for i, names in enumerate(groups)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                parallel = list(pool.map(lambda item: capsule(item[0] + 1, item[1], files, evidence, b, 'https://example.invalid/raw'), enumerate(groups)))
            self.assertEqual(serial, parallel)
            for item in serial:
                self.assertEqual((a / item['name']).read_bytes(), (b / item['name']).read_bytes())
                with tarfile.open(a / item['name']) as archive:
                    self.assertEqual(archive.getnames(), item['members'])

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'Optional NumPy required for publisher')
    def test_batch_git_reads_match_original_binary_text_and_symlink_blobs(self):
        from tools.publish_memory_application import Archive, runtimes
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); repo = root / 'repo'; repo.mkdir()
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            (repo / 'tools').mkdir(); (repo / 'native').mkdir()
            (repo / 'tools/a.py').write_text("path='/private/run'\n")
            (repo / 'native/raw.bin').write_bytes(b'\x00\xff\nraw\n')
            (repo / 'tools/link.py').symlink_to('a.py')
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
            commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
            archive = Archive(root / 'archive', {'/private': '<ROOT>'})
            actual_run = subprocess.run
            def run(*args, **kwargs):
                kwargs.setdefault('cwd', repo); return actual_run(*args, **kwargs)
            with patch('tools.publish_memory_application.subprocess.run', side_effect=run):
                runtimes(SimpleNamespace(commits=[commit]), archive)
            inventory = json.loads((archive.output / 'runtime/inventory.json').read_text())[commit]['files']
            with tarfile.open(archive.output / 'runtime' / (commit + '.tar.gz')) as tar:
                for name, record in inventory.items():
                    raw = subprocess.check_output(['git', '-C', str(repo), 'show', commit + ':' + name])
                    self.assertEqual(record['original_sha256'], hashlib.sha256(raw).hexdigest())
                    try: expected = archive.public(raw.decode()).encode()
                    except UnicodeDecodeError: expected = raw
                    self.assertEqual(tar.extractfile(name).read(), expected)


if __name__ == '__main__':
    unittest.main()
