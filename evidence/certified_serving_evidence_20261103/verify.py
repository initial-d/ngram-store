"""Validate complete serving evidence and regenerate its file/hash manifest."""
import hashlib
import json
from pathlib import Path

r = Path(__file__).resolve().parent
data = json.loads((r / 'report.json').read_text())
v = json.loads((r / 'verifier.json').read_text())
assert data['status'] == v['status'] == 'completed' and not data['failures']
assert len(data['runs']) == 2
old, new = data['runs']
assert [old['job']['method'], new['job']['method']] == ['legacy', 'external']
assert old['source_commit'] == new['source_commit'] == data['source_commit'] == v['source_commit']
assert old['source_files_sha256'] == new['source_files_sha256'] == v['source_files_sha256']
assert old['input'] == new['input'] and old['input']['rows'] == 1855907
assert old['result']['partition'] == new['result']['partition']
assert old['result']['projection']['means_sha256'] == new['result']['projection']['means_sha256']
assert old['native_table_sha256'] == new['native_table_sha256'] == '2e6131f707097ebcf9377338b31d0247c53f25974b754da168569821638e89e7'
assert data['all_native_table_bytes_identical'] == old['result']['storage']['artifact_bytes'] == new['result']['storage']['artifact_bytes'] == 385017776
assert v['all_plan_arrays_identical'] and v['pytorch_adapter_identical_to_native_bytes']
assert v['plan']['bags'] == 512 and v['plan']['partition_sha256'] == new['result']['partition']['mapping_sha256']
assert v['native_outputs_sha256'] == {
    'sum': 'c26cb78b896e5b1f6a042af89f0c4dbf2c89b072d58922688f6548bea95ad162',
    'mean': '939b42d08e95ba66d0202f8dd13ad5dc35e7a29d10806ce09a039d37b475bef7'}
assert data['rss_reduction_factor'] == old['peak_rss_bytes'] / new['peak_rss_bytes']
assert data['build_time_factor'] == new['total_seconds'] / old['total_seconds']
installed = json.loads((r / 'installed-linux-reproduction.json').read_text())
assert installed['status'] == 'passed' and installed['pytorch']['native_output_bytes_identical']
assert installed['every_row_matches_legacy_projection'] and installed['incomplete_group_refused']
assert installed['training']['changed_rows'] == installed['training']['rows'] == 919
manifest = {}
for p in sorted(r.rglob('*')):
    if p.is_file() and p.name != 'manifest.json':
        manifest[str(p.relative_to(r))] = {'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
(r / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Complete serving evidence verified:', len(manifest), 'files')
