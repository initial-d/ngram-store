"""Recompute accuracy and NLL from saved logits and check the frozen selection.

Usage: python evidence/training_participation_sensitivity_results_20261111/verify.py
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
reports = {p.stem: json.loads(p.read_text()) for p in (ROOT / 'quality').glob('*.json')}
for name, r in reports.items():
    z = np.load(ROOT / 'quality' / (name + '.npz'))
    logits, labels = z['logits'].astype(np.float64), z['labels']
    shifted = logits - logits.max(axis=1, keepdims=True)
    nll = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    assert len(labels) == r['documents'], name
    assert abs(float(np.mean(logits.argmax(1) == labels)) - r['accuracy']) < 1e-12, name
    assert abs(float(nll.mean()) - r['mean_nll']) < 1e-9, name
selection = json.loads((ROOT / 'selection.json').read_text())
means = {lr: np.mean([reports[f'dbpedia_14-exact-uniform-s{s}-lr{lr}-calibration']['mean_nll'] for s in (3, 7, 11)])
         for lr in ('1', '3', '10')}
best = min(means, key=means.get)
assert float(best) == float(selection['learning_rate']) and selection['test_used'] is False
tests = sorted(n for n in reports if n.endswith('-test'))
assert tests == [f'dbpedia_14-exact-uniform-s{s}-lr{best}-test' for s in (11, 3, 7)], tests
complete = json.loads((ROOT / 'complete.json').read_text())
acc = np.mean([reports[t]['accuracy'] for t in tests])
assert abs(round(acc, 4) - complete['test_accuracy_mean']) < 1e-9
print(f'verified {len(reports)} reports; selected LR {best} on calibration NLL; test accuracy {acc:.4f}')
