"""Fail closed on partial/mixed build grids before presenting measured gains."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy required')
class ExternalEvidenceTests(unittest.TestCase):
    def test_complete_grid_and_native_checks_reject_missing_or_mixed_evidence(self):
        from tools.finalize_external_incidence import validate_grid
        root = Path(__file__).resolve().parents[1] / 'docs/external_incidence_evidence_20261102'
        data = {k: json.loads((root / name).read_text()) for k, name in
            (('report', 'report.json'), ('preparation', 'preparation.json'), ('verifier', 'native-verifier.json'))}
        self.assertEqual(len(validate_grid(**data)), 8)
        mutations = [
            lambda d: d['report']['runs'].pop(),
            lambda d: d['report']['runs'].append(d['report']['runs'][0]),
            lambda d: d['report']['failures'].append({'name': 'failed'}),
            lambda d: d['report']['runs'][1]['all_output_bytes_identical'].update(means=0),
            lambda d: d['report']['runs'][1]['source_files_sha256'].update(external_cotraining='changed'),
            lambda d: d['report']['runs'][1]['input'].update(source_sha256='changed'),
            lambda d: d['verifier'].update(all_group_means_identical_to_previous_native_rows=1),
            lambda d: d['verifier'].update(means_sha256='changed'),
            lambda d: d['verifier']['identical_native_outputs'].pop('sum'),
            lambda d: d['preparation']['sets'].append(d['preparation']['sets'][0]),
        ]
        for mutation in mutations:
            altered = copy.deepcopy(data)
            mutation(altered)
            with self.assertRaises(ValueError):
                validate_grid(**altered)
