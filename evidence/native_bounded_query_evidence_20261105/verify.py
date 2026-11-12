"""Verify timing samples, frozen numerical hashes and unchanged capacity offline."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from tools.analyze_budgeted_bags import read_report_bytes
from tools.analyze_native_bounded_bags import summarize


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('evidence hash mismatch: ' + name)
    report = json.loads((ROOT / 'report.json').read_text())
    old_root = REPO / 'evidence/budgeted_bag_evidence_20261104'
    original_bytes = read_report_bytes(old_root / 'report.json.gz')
    prior_bytes = (old_root / 'retiming.json').read_bytes()
    if (report['source_commit'] != manifest['source_commit']
            or report['source_sha256'] != manifest['measured_source_sha256']
            or report['frozen_report_sha256'] != hashlib.sha256(original_bytes).hexdigest()
            or report['prior_retiming_sha256'] != hashlib.sha256(prior_bytes).hexdigest()):
        raise ValueError('measured source or frozen reference provenance changed')
    actual = summarize(report, json.loads(original_bytes), json.loads(prior_bytes))
    if actual != json.loads((ROOT / 'summary.json').read_text()):
        raise ValueError('summary differs from validated raw samples')
    print(json.dumps({'status': 'passed', 'checkpoints': len(report['runs']),
        'paired_samples_per_trace': report['repeat'],
        'output_bound_budget_capacity_and_frozen_source_checks': True}))


if __name__ == '__main__':
    main()
