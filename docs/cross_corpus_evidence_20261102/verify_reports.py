from pathlib import Path
import copy,gzip,hashlib,json
from tools.analyze_corpus_replication import summarize
root=Path('docs/cross_corpus_evidence_20261102')
reports={key:json.loads((root/name).read_text()) for key,name in {
 'training':'training-report.json','codecs':'codec-analysis.json',
 'query':'cross-query-report.json','native':'native-retiming.json'}.items()}
summary=summarize(**reports)
assert summary['scored_codec_variants']==581
merged_bytes=gzip.decompress((root/'merged-report.json.gz').read_bytes())
assert hashlib.sha256(merged_bytes).hexdigest()==reports['codecs']['report_sha256']==reports['query']['frozen_report_sha256']
base=json.loads(gzip.decompress((root/'grid-report.json.gz').read_bytes()))
canonical=json.loads((root/'grid-orchestration.json').read_text())
for run in base['runs']:
 phase=next(p for p in canonical['checkpoint_phases'] if p['name']==f"order-{run['order']}-seed-{run['seed']}")
 assert hashlib.sha256(json.dumps(run,sort_keys=True).encode()).hexdigest()==phase['run_sha256']
 assert len(run['variants'])==57 and not run['failures']
for name,digest in base['source_files'].items():
 assert hashlib.sha256((Path('tools')/name).read_bytes()).hexdigest()==digest
print('PASS: 18 paired trainings, 9 complete 57-candidate grids, 27 byte-identical query retimings')
print('PASS: frozen report hash chain, canonical phase hashes, current engine source hashes')

def reject(label,mutate):
 changed=copy.deepcopy(reports)
 mutate(changed)
 try: summarize(**changed)
 except ValueError as error: print('PASS: reject',label,'->',str(error))
 else: raise AssertionError('accepted invalid report: '+label)
reject('duplicate training pair',lambda d:d['training']['runs'].append(d['training']['runs'][0]))
reject('duplicate query family/seed',lambda d:d['query']['runs'].__setitem__(0,copy.deepcopy(d['query']['runs'][1])))
reject('mixed training checkpoint',lambda d:d['query']['runs'][0].__setitem__('checkpoint_sha256','0'*64))
reject('mismatched selected codec cost',lambda d:d['query']['runs'][0].__setitem__('model_bytes',1))
reject('changed output bytes',lambda d:d['query']['runs'][0]['query']['known'].__setitem__('output_bytes_identical',False))
reject('rejected registered work',lambda d:d['query']['runs'][0]['query']['known']['fidelity'].__setitem__('accepted',511))
reject('missing quality-budget selection',lambda d:d['codecs']['validation_selected'].pop())
reject('failed native equality',lambda d:d['native'].__setitem__('plan_arrays_byte_identical',False))
