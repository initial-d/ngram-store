import gzip,hashlib,json,subprocess
from pathlib import Path
p=Path('${RUN_ROOT}');out=p/'public-evidence';aux=p/'public-auxiliary'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
roots=[(str(p),'${RUN_ROOT}'),('${TRAINING_ROOT}','${TRAINING_ROOT}'),('${ENV_ROOT}','${ENV_ROOT}'),('${CORPUS_ZIP}','${CORPUS_ZIP}')]
def public(v):
 if isinstance(v,dict):return {k:public(x) for k,x in v.items()}
 if isinstance(v,list):return [public(x) for x in v]
 if isinstance(v,str):
  for a,b in roots:v=v.replace(a,b)
  for tail in v.split('/' + 'llm-align/')[1:]:
   assert not tail or tail[0] in "'\" \r\n", 'unredacted environment path'
 return v
def write(path,v):path.write_text(json.dumps(public(v),indent=2,sort_keys=True,allow_nan=False)+'\n')
def original(source,target):
 v=json.loads(source.read_text());v['private_report_sha256']=sha(source);write(out/target,v)
original(p/'row-audit/report.initial-failed.json','row-audit-initial-failure.json')
original(p/'retiming/report.initial-interrupted.json','retiming-initial-interrupted.json')
original(p/'retiming/report.before-resume.json','retiming-before-resume.json')
row=json.loads((p/'row-audit/report.initial-failed.json').read_text());logs=[];rounding=[]
for failure in row['failures']:
 job=Path(failure['job']);model=job.parent.parent.name;name=job.stem
 path=p/'row-audit'/model/(name+'.log');assert path.exists()
 logs.append({'model':model,'name':name,'private_log_sha256':sha(path),'text':public(path.read_text())})
 repaired=json.loads((p/'row-audit'/model/(name+'.json')).read_text());rounding.append({'model':model,'name':name,'thread_rounding':repaired['thread_rounding']})
(aux/'row-initial-failure-logs.json.gz').write_bytes(gzip.compress((json.dumps(logs,sort_keys=True)+'\n').encode(),mtime=0))
write(out/'opq-rounding-summary.json',{'repaired':len(rounding),'changed_candidates':sum(bool(x['thread_rounding']['validation_gate_changes']) for x in rounding),'candidates':rounding})
initial=json.loads((p/'retiming/report.initial-interrupted.json').read_text());logs=[]
for failure in initial['failures']:
 i=initial['requests'].index(failure['request']);path=p/'retiming'/f'{i:04d}.log';assert path.exists()
 logs.append({'request':failure['request'],'private_log_sha256':sha(path),'text':path.read_text()})
(aux/'retiming-initial-failure-logs.json.gz').write_bytes(gzip.compress((json.dumps(public(logs),sort_keys=True)+'\n').encode(),mtime=0))
final=json.loads((p/'retiming/report.json').read_text());assert final['status']=='complete' and not final['failures']
library=p/'query-repair-source/build/libmeft_structural.so'
write(out/'retiming-recovery.json',{'source_commit':'76901984cc55dfed36715228529fec89b1270179','native_source_sha256':sha(p/'query-repair-source/native/meft_structural.cc'),'compiled_library_sha256':sha(library),'initial_failures':len(initial['failures']),'retained_before_resume':len(json.loads((p/'retiming/report.before-resume.json').read_text())['complete']),'recovery_source_report_sha256':final['resume']['source_report_sha256'],'serialization':'MEFTKEY1 sorted UTF-8 key bytes, offsets and original IDs','query_implementation':'std::unordered_map, separate from sorted binary search','new_query_library_environment':'MEFT_QUERY_STRUCTURAL_LIBRARY','regression_tests':{'run':10,'failures':0,'errors':0,'skipped':0,'private_success_log_sha256':sha(p/'query-repair-tests-v3.log')}})
records=[]
for item in final['complete']:
 r=item['request'];job=json.loads(Path(r['job']).read_text());path=Path(job['report']).with_name(r['name']+f'-cache{r["cached"]}-{r["index"]}-rowcache{r["cache"]}-retiming.json');assert sha(path)==item['report_sha256']
 value=public(json.loads(path.read_text()));value={'name':r['name'],**value}
 records.append({'request':r,'private_report_sha256':item['report_sha256'],'public_record_sha256':hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()})
write(out/'retiming-record-integrity.json',records)
for name in ('query-repair-build.log','query-repair-tests.log','query-repair-tests-v2.log','query-repair-tests-v3.log','query-repair-preflight.log','finalize-v4.sh','finalize-v4.log','retiming-resume.log','input-audit-v2.sh','input-audit-v3.sh','input-audit-v4.sh','input-audit-v5.sh','input-precheck.sh','input-precheck-v2.sh','input-precheck.log','input-precheck-v2.log','input-audit.initial-failed.log','input-audit.log','input-audit-v3.log','input-audit-v5.log','repair-rows.sh','repair-rows.log','archive-rows.sh','export-final-aux.py','export-final-aux.sh'):
 path=p/name
 if path.exists():(aux/name).write_text(public(path.read_text()))
write(aux/'faiss-thread-contract.json',{'build_omp_threads':4,'query_omp_threads':1,'build_training_source_rows_only':True,'validation_build_rounding':4,'frozen_test_confirmation_rounding':1,'posthoc_one_thread_validation_gates_changed':0,'limits':'OPQ reverse-transform bytes differ at approximately 1e-7 relative Frobenius; serialized codec files and the four-thread decode are verified before auditing the one-thread serving values.'})
codecs=[]
for f in sorted((p/'frontier').glob('*/bundles/*/codec.json')):
 if not f.parent.name.startswith(('faiss-', 'dedup-')):continue
 v=json.loads(f.read_text());assert v['training_threads']==4
 codecs.append({'model':f.parent.parent.parent.name,'name':f.parent.name,'private_metadata_sha256':sha(f),'metadata':v})
assert len(codecs)==18*38
write(out/'faiss-codec-metadata.json',codecs)
(p/'final-auxiliary-complete').touch()
print('Complete initial-failure logs, numerical repairs, retiming recovery and source provenance exported',flush=True)
