"""Dependency-free checks for real-weight preflight; not training evidence."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
data_sha='977a6acce4705507b5fd6bfcea8f61cd78f9ed0f9cd9c9a6bcd6c8a3ed61c833'
for model,gradients in [('qwen06',97),('qwen35',115)]:
    r=json.loads((root/f'{model}-preflight.json').read_text())
    assert r['status']=='passed' and r['official_oracle_choices']==104
    assert r['masked_replay']=={'logits_byte_exact':True,'all_masked_gradients_exact':True,'trainable_gradients':gradients}
    assert r['same_logits_official_formula_max_delta']==0 and r['batch_rank_changes']==r['batch_target_accuracy_changes']==0
    assert not any(r['loading'].values()) and r['physical_rows']==3628
    assert r['environment']['transformers']=='5.3.0' and 'A100' in r['environment']['gpu']
    p=json.loads((root/f'{model}-full-prepare.json').read_text())
    assert p['status']=='prepared' and p['data_sha256']==data_sha and p['edits']==20877
    assert p['training_sequences']==20877 and p['token_boundary_mismatches']==0
    assert sum(p['split_counts'].values())==20877 and p['edited_neighborhood_prompts']==4189
    assert p['unique_sequences']>300000 and p['tokens']>4000000
audit=json.loads((root/'precision-audit.json').read_text())
assert audit['fla_solver_precision']=='tf32' and audit['torch_matmul_tf32'] is False
assert 'Ran 7 tests' in (root/'cuda-fixture.log').read_text() and '\nOK\n' in (root/'cuda-fixture.log').read_text()
assert 'FAILED (errors=1)' in (root/'cuda-fixture-failure-1.log').read_text()
manifest=json.loads((root/'model-manifest.json').read_text())
assert manifest['read_only'] is True
for m in manifest['models'].values():
    assert m['total_file_bytes']==sum(f['bytes'] for f in m['files'].values())
    assert any(n.endswith('.safetensors') for n in m['files'])
    assert all(len(f['sha256'])==64 for f in m['files'].values())
print('Verified 0.6B/4B real-weight interfaces, official score formulas, full dataset preparation and precision disclosure; training outcomes are separate.')
