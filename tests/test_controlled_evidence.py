import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.package_controlled_evidence import digest_file, verify
from tools.summarize_controlled_application import summarize


class EvidenceIntegrityTest(unittest.TestCase):
    def test_capsule_verification_rejects_changed_audit_links_and_missing_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); payload = b'{"status":"complete"}\n'; name = 'cpu-audit/result.json'
            archive_path = root / 'raw.tar.gz'
            with tarfile.open(archive_path, 'w:gz') as archive:
                entry = tarfile.TarInfo(name); entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))
            sha = hashlib.sha256(payload).hexdigest()
            manifest = {'members': {name: {'sha256': sha, 'bytes': len(payload)}},
                        'audit_links': {'/original/result.json': {'member': name, 'sha256': sha}},
                        'capsules': [{'name': archive_path.name, 'bytes': archive_path.stat().st_size,
                                      'sha256': digest_file(archive_path), 'members': [name]}]}
            path = root / 'manifest.json'; path.write_text(json.dumps(manifest))
            self.assertEqual(verify(root)['members'], 1)
            manifest['audit_links']['/original/result.json']['sha256'] = '0' * 64
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'audit hash link'):
                verify(root)
            manifest['audit_links'] = {}; manifest['members']['missing.json'] = {'sha256': sha, 'bytes': 1}
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'missing evidence'):
                verify(root)
            manifest['members'].pop('missing.json'); path.write_text(json.dumps(manifest))
            archive_path.write_bytes(archive_path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'capsule changed'):
                verify(root)

    def test_query_summary_rejects_unpaired_devices(self):
        baseline = {'job': 'direct-fp32', 'mode': 'host', 'status': 'passed', 'gpu_uuid': 'gpu0',
                    'cpu_affinity': [0], 'resource_policy': 'shared', 'p50_ms': 2.,
                    'documents_per_second': 100., 'complete_model_bytes': 1000}
        candidate = {**baseline, 'job': 'compressed', 'p50_ms': 10.,
                     'documents_per_second': 20., 'complete_model_bytes': 500}
        value = {'models': [{'model': 'm', 'family': ['ag_news', 'zero', 'exact', 0], 'quality': [],
                             'choices': [{'heldout': {'name': 'compressed'}}]}],
                 'timings': [{'model': 'm', 'reports': [baseline, candidate]}]}
        result = summarize(value)['selected_host_query_costs'][0]
        self.assertAlmostEqual(result['p50_slowdown_selected_over_baseline']['geometric_mean'], 5.)
        self.assertEqual(result['complete_model_saving_fraction']['minimum'], .5)
        candidate['gpu_uuid'] = 'gpu1'
        with self.assertRaisesRegex(ValueError, 'paired device'):
            summarize(value)
