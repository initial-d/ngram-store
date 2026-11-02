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
        sys.argv = ['ngram-store', 'compact-bags', '--input', str(root / 'rows'), '--dim', '2',
                    '--bag-ids', str(root / 'ids'), '--bag-offsets', str(root / 'offsets'),
                    '--output', str(root / 'out'), '--memory-mib', '8']
        entry.load()()
        partition = ngram_store.BagPartition.load(root / 'out/domain.npz')
        assert partition.mapping.tolist() == [0, 0, 1]
        assert (root / 'out/means.f32').read_bytes() == np.asarray([[2., 3.], [9., 10.]], dtype='<f4').tobytes()
        assert 'torch' not in sys.modules
        print(json.dumps({'status': 'passed', 'python': sys.version, 'numpy': np.__version__,
            'package': ngram_store.__file__, 'torch_loaded': False,
            'installed_cli_mapping_and_means_identical': True}, indent=2))


if __name__ == '__main__':
    main()
