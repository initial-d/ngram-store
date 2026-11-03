"""MIT: run with an installed NGramStore wheel, NumPy and optionally PyTorch.

No checkout, external dataset, downloaded model or GPU is required. The small
trained example demonstrates integration, not representative performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from ngram_store import BagPartition, CertifiedBagRows, NativeRows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--torch', action='store_true', help='Also check the frozen CPU PyTorch operator and trained SGNS head.')
    cfg = p.parse_args()
    root = cfg.output.resolve()
    command = [sys.executable, '-m', 'ngram_store.cli']
    subprocess.run(command + ['demo-bags', '--output', str(root)], check=True, capture_output=True)
    training = json.loads((root / 'report.json').read_text())
    assert training['status'] == 'passed' and training['changed_rows'] > 0
    rows = np.fromfile(root / 'vectors.f32', dtype='<f4').reshape(-1, training['dim'])
    ids = np.fromfile(root / 'ids.u64', dtype='<u8')
    offsets = np.fromfile(root / 'offsets.u64', dtype='<u8')
    bags = [ids[int(a):int(b)].tolist() for a, b in zip(offsets, offsets[1:])]
    protected = np.fromfile(root / 'protected.u64', dtype='<u8').tolist()
    partition = BagPartition.fit(bags, len(rows), protected=protected)
    runtime = BagPartition.load(root / 'memory/domain.npz')
    assert runtime.summary() == partition.summary()
    expected = partition.means(rows)[partition.mapping]
    with NativeRows(root / 'memory/table.men') as native:
        assert native.rows(list(range(len(rows)))) == expected.tobytes()
    returned = subprocess.run(command + ['query-bags', '--store', str(root / 'memory'),
        '--bags', str(root / 'queries.json'), '--mode', 'mean'], check=True, capture_output=True, text=True)
    values = np.asarray(json.loads(returned.stdout)['vectors'], dtype=np.float32)
    assert values.tobytes() == np.asarray(json.loads((root / 'outputs.json').read_text()), dtype=np.float32).tobytes()
    rejected = subprocess.run(command + ['query-bags', '--store', str(root / 'memory'),
        '--bags', str(root / 'rejected-query.json')], capture_output=True, text=True)
    assert rejected.returncode == 2 and 'uncertified' in rejected.stderr
    report = {'status': 'passed', 'python': sys.version, 'numpy': np.__version__, 'os': platform.platform(),
        'source_vectors_sha256': hashlib.sha256(rows.tobytes()).hexdigest(),
        'every_row_matches_legacy_projection': True, 'cli_outputs_match_python': True,
        'incomplete_group_refused': True, 'training': training,
        'limitation': 'Self-reproduction entry, not independent adoption or representative capacity/quality evidence.'}
    if cfg.torch:
        import torch
        from ngram_store import CertifiedEmbeddingBag
        queries = json.loads((root / 'queries.json').read_text())
        flat = torch.tensor([i for bag in queries for i in bag], dtype=torch.long)
        edges = torch.tensor([0, *np.cumsum([len(b) for b in queries])], dtype=torch.long)
        dense = torch.nn.EmbeddingBag.from_pretrained(torch.from_numpy(rows), freeze=True,
            mode='mean', include_last_offset=True)
        targets = torch.from_numpy(np.fromfile(root / 'output.f32', dtype='<f4').reshape(-1, training['dim']))
        with CertifiedEmbeddingBag(root / 'memory', mode='mean', include_last_offset=True) as memory:
            actual, original = memory(flat, edges), dense(flat, edges)
            assert actual.numpy().tobytes() == values.tobytes()
            assert torch.allclose(actual, original, rtol=1e-5, atol=1e-7)
            head = torch.nn.Linear(training['dim'], len(targets), bias=False)
            with torch.no_grad():
                head.weight.copy_(targets)
                native_logits, dense_logits = head(actual), head(original)
                assert torch.allclose(native_logits, dense_logits, rtol=1e-5, atol=1e-7)
            report['pytorch'] = {'version': torch.__version__, 'native_output_bytes_identical': True,
                'maximum_source_embedding_error': float((actual - original).abs().max()),
                'maximum_source_sgns_logit_error': float((native_logits - dense_logits).abs().max()),
                'scope': 'Frozen CPU EmbeddingBag and example-trained head; no downstream quality claim.'}
    (root / 'reproduction.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'rows': len(rows), 'groups': len(partition.sizes),
        'source_vectors_sha256': report['source_vectors_sha256'], 'pytorch_checked': cfg.torch,
        'report': str(root / 'reproduction.json')}, indent=2))


if __name__ == '__main__':
    main()
