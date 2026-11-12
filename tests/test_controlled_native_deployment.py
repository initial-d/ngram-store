"""Guard calibration-only selection and complete-model capacity accounting."""
from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(find_spec('numpy') is not None, 'optional NumPy research dependency required')
class ControlledLayoutSelectionTests(unittest.TestCase):
    def test_test_latency_never_changes_calibration_choice_and_complete_bytes_gate(self):
        from tools.controlled_native_deployment import select_layouts
        from tools.tree_node_quality import write
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); entries = []
            for name, byte_count, calibration, test in (
                ('direct', 1000, 2., 1.), ('row-packets-fast-expanded', 1001, .1, .1),
                ('row-packets-valid', 800, 1., 3.), ('row-packets-test-fast', 700, 1.5, .1)):
                report = root / (name + '.json')
                write(report, {'complete_branch_bytes': byte_count,
                    'query': {'calibration': {'p50_ms': calibration}, 'test': {'p50_ms': test}}})
                entries.append({'codec': 'sq4', 'variant': 'direct' if name == 'direct' else 'row-packets',
                    'job': str(root / name), 'report': str(report)})
            selected = select_layouts(entries, 'row-packets', .1)
            self.assertEqual(Path(selected[0]['job']).name, 'row-packets-valid')
            # A stricter COMPLETE-model budget triggers transparent direct fallback.
            self.assertEqual(select_layouts(entries, 'row-packets', .5), [entries[0]])
            for saving in (-.1, 1.):
                with self.assertRaises(ValueError): select_layouts(entries, 'row-packets', saving)


if __name__ == '__main__':
    unittest.main()
