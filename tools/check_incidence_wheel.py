"""Check the installed NumPy-only incidence extra from outside the checkout."""
from importlib.metadata import distribution
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import meft_memory
import ngram_store


def main():
    assert 'torch' not in sys.modules
    assert ngram_store.ExternalBagPartition is meft_memory.ExternalBagPartition
    assert 'numpy; extra == "incidence"' in distribution('meft-memory').requires
    checkout = Path(__file__).resolve().parents[1]
    assert checkout not in Path(ngram_store.__file__).resolve().parents
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        np.asarray([[1., 2.], [3., 4.], [9., 10.]], dtype='<f4').tofile(root / 'rows')
        np.asarray([0, 1, 2], dtype='<u8').tofile(root / 'ids')
        np.asarray([0, 2, 3], dtype='<u8').tofile(root / 'offsets')
        entry = next(e for e in distribution('meft-memory').entry_points if e.name == 'ngram-store')
        sys.argv = ['ngram-store', 'compact-bags', '--native', '--input', str(root / 'rows'), '--dim', '2',
                    '--bag-ids', str(root / 'ids'), '--bag-offsets', str(root / 'offsets'),
                    '--output', str(root / 'out'), '--memory-mib', '8']
        entry.load()()
        partition = ngram_store.BagPartition.load(root / 'out/domain.npz')
        assert partition.mapping.tolist() == [0, 0, 1]
        with ngram_store.CertifiedBagRows(root / 'out') as store:
            assert store.aggregate([[0, 1], [2]]).tobytes() == np.asarray([[4., 6.], [9., 10.]], dtype='<f4').tobytes()
            try:
                store.aggregate([[0]])
            except ValueError:
                pass
            else:
                raise AssertionError('incomplete group must be rejected')
        # The opt-in path binds actual source residuals to the installed native
        # bundle, retains no source matrix at query time and reports its bounds.
        ngram_store.build_error_bounds(root / 'rows', root / 'out', root / 'bounds')
        with ngram_store.BudgetedBagRows(root / 'out', root / 'bounds') as store:
            result = store.aggregate([[0]], max_absolute_l2=1.415)
            assert result.vectors.tolist() == [[2., 3.]]
            assert not result.exact_domain[0]
            assert result.absolute_l2_bounds[0] < 1.415
            try:
                store.aggregate([[0]], max_absolute_l2=1.)
            except ValueError:
                pass
            else:
                raise AssertionError('above-budget query must be rejected')
        # The wheel must include its own small training corpus and offline demo.
        sys.argv = ['ngram-store', 'demo-bags', '--output', str(root / 'demo')]
        entry.load()()
        assert json.loads((root / 'demo/report.json').read_text())['status'] == 'passed'
        assert 'torch' not in sys.modules
        print(json.dumps({'status': 'passed', 'python': sys.version, 'numpy': np.__version__,
            'package': ngram_store.__file__, 'torch_loaded': False,
            'installed_cli_mapping_and_means_identical': True}, indent=2))


if __name__ == '__main__':
    main()
