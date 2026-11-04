"""Verify frozen-report provenance and every published coverage/cost summary."""
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from tools.analyze_budgeted_bags import summarize


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('evidence hash mismatch: ' + name)
    report_bytes = (ROOT / 'report.json').read_bytes()
    report = json.loads(report_bytes)
    retiming = json.loads((ROOT / 'retiming.json').read_text())
    if manifest['source_commits'] != {'first': report['source_commit'], 'batched': retiming['source_commit']}:
        raise ValueError('source commit provenance mismatch')
    if retiming['frozen_report_sha256'] != hashlib.sha256(report_bytes).hexdigest():
        raise ValueError('retiming references a different original report')
    if retiming['reference_module_sha256'] != report['source_files_sha256']['budgeted_bags.py']:
        raise ValueError('reference code differs from the first measured implementation')
    training = json.loads((REPO / 'docs/cross_corpus_evidence_20261102/training-report.json').read_text())
    if report['corpus_sha256'] != training['tokenized_corpus_sha256']:
        raise ValueError('different cached corpus')
    expected = {(r['convention'], r['order'], r['seed']): r['checkpoint_sha256'] for r in training['runs']}
    if any(r['checkpoint_sha256'] != expected[(r['convention'], r['order'], r['seed'])] for r in report['runs']):
        raise ValueError('checkpoint source differs from the original trained evidence')
    actual = summarize(report, retiming)
    if actual != json.loads((ROOT / 'summary.json').read_text()):
        raise ValueError('summary differs from the verified raw observations')
    print(json.dumps({'status': 'passed', 'checkpoints': len(report['runs']),
        'coverage_capacity_and_bound_checks': True, 'retiming_contract_and_source_checks': True}))


if __name__ == '__main__':
    main()
