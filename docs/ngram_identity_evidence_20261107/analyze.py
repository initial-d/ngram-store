"""Audit frozen 425c187 runs and compute preregistered paired uncertainty, CPU only."""
import hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path('/llm-align/duyimin/ngram-identity-vs-hash-20261107')
sys.path.insert(0,str(ROOT/'source'))
from tools.counterfact_data import summarize
from tools.counterfact_analyze import arrays,paired_interval,harmonic

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def save(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)

out=ROOT/'final-audit';out.mkdir(exist_ok=True)
manifest={};by={};items=[];packed={};reports=sorted((ROOT/'runs').glob('*/report.json'))
assert len(reports)==24,len(reports)
for f in reports:
 name=f.parent.name; model,n,variant,s=name.split('-');edits=int(n);seed=int(s[1:]);r=json.loads(f.read_text())
 assert r['status']=='complete',name
 assert r['environment']['code_sha']=='425c1876614b9533ecf3b5e092e1dbed0bfd7651',name
 assert r['protocol']['identity_control'] and not r['protocol']['train_decoder_suffix'],name
 assert len(r['epochs'])==20 and [x['epoch'] for x in r['epochs']]==list(range(20)),name
 assert r['seed']==seed and r['variant']==variant,name
 assert r['memory_parameters']==r['trainable_parameters'],name
 assert all('memory' in x for x in r['trained_parameter_names']),name
 case=f.parent/'case-scores.json'
 assert sha(case)==r['score_artifacts']['case-scores.json']['sha256'],name
 assert sha(f.parent/'delta.safetensors')==r['delta']['sha256'],name
 assert sha(f.parent/'vocabulary.npz')==r['identity']['vocabulary_sha256'],name
 cases=json.loads(case.read_text());assert len(cases)==edits,name
 assert len({x['case_id'] for x in cases})==edits,name
 assert sorted(x['case_index'] for x in cases)==list(range(edits)),name
 for c in cases:
  assert c['split']==('calibration' if c['case_index']%10==0 else 'test'),name
  for g,rows in c['groups'].items():
   for p in rows:
    a,b=p['target_new'],p['target_true'];assert np.isfinite(a) and np.isfinite(b),name
    assert p['success']==(b<a if g=='neighborhood' else a<b),name
 metrics={k:summarize([c for c in cases if k=='all' or c['split']==k]) for k in ['all','calibration','test']}
 assert metrics==r['metrics'],name
 ids,av=arrays(cases)
 packed[name+'_ids']=np.array(ids,dtype=np.int64);packed[name+'_success']=av
 by[model,edits,variant,seed]=(ids,av,r)
 manifest[str(case.relative_to(ROOT))]=sha(case);manifest[str(f.relative_to(ROOT))]=sha(f)
 item={'run':name,'report_sha256':sha(f),'metrics':metrics,'trainable_parameters':r['trainable_parameters'],'physical_rows':r['physical_rows'],'row_bytes':r['identity']['row_bytes'],'identity_bundle_bytes':r['identity_bundle_bytes'],'training_seconds':r['training_seconds'],'source_sha':r['environment']['code_sha']}
 if variant=='hash':
  native=f.parent/'case-scores-native-oov.json';assert sha(native)==r['native_oov_control']['case_scores_sha256'],name
  assert r['native_oov_control']['same_trained_delta'],name
  cc=json.loads(native.read_text());nm={k:summarize([c for c in cc if k=='all' or c['split']==k]) for k in ['all','calibration','test']}
  assert nm==r['native_oov_control']['metrics'],name
  ni,nv=arrays(cc);assert ni==ids,name
  by[model,edits,'hash-native-oov',seed]=(ni,nv,r)
  packed[name+'_native_success']=nv;manifest[str(native.relative_to(ROOT))]=sha(native);item['native_oov_metrics']=nm
 items.append(item)
 print('audited',name,flush=True)
 del cases
np.savez_compressed(out/'paired-test-arrays.npz',**packed)
comparisons=[]
for model in ['qwen06','qwen35']:
 for edits in [1024,20877]:
  for control in ['hash','hash-native-oov']:
   aa=[];bb=[];seed_stats=[]
   for seed in [3,7,11]:
    ai,av,ar=by[model,edits,'exact',seed];bi,bv,br=by[model,edits,control,seed]
    assert ai==bi,name
    for field in ['protocol_sha256','prepared_sha256','trainable_parameters','memory_parameters','physical_rows','optimizer_steps','frozen_weight_sha256','frozen_decoder_suffix_sha256']:
     assert ar[field]==br[field],(model,edits,seed,field)
    assert ar['identity']['row_bytes']==br['identity']['row_bytes']
    assert ar['identity']['vocabulary_sha256']==br['identity']['vocabulary_sha256']
    assert [x['actual_sequences'] for x in ar['epochs']]==[x['actual_sequences'] for x in br['epochs']]
    aa.append(av);bb.append(bv)
    seed_stats.append({'seed':seed,'harmonic_delta':float(harmonic(av.mean(0))-harmonic(bv.mean(0))),'group_deltas':(av.mean(0)-bv.mean(0)).tolist()})
   result=paired_interval(np.stack(aa),np.stack(bb),replicates=2000)
   ns=result['group_deltas']['neighborhood']
   point_gate=bool(result['harmonic_delta_interval95'][0]>0 and ns['delta']>=-0.02)
   conservative_gate=bool(result['harmonic_delta_interval95'][0]>0 and ns['interval95'][0]>=-0.02)
   result.update(model=model,edits=edits,control=control,seeds=[3,7,11],seed_results=seed_stats,gate_positive_H_with_NS_point_within_2pp=point_gate,gate_positive_H_with_NS_interval_within_2pp=conservative_gate,mean_per_seed_H_delta=float(np.mean([s['harmonic_delta'] for s in seed_stats])),exact_group_means=np.stack(aa).mean((0,1)).tolist(),control_group_means=np.stack(bb).mean((0,1)).tolist())
   comparisons.append(result)
   print('comparison',model,edits,control,'H_delta_pp',100*result['harmonic_delta'],'CI_pp',[100*x for x in result['harmonic_delta_interval95']],'NS_delta_pp',100*ns['delta'],'conservative_gate',conservative_gate,flush=True)
summary={'status':'complete-audited','completed_trained_runs':24,'source_sha':'425c1876614b9533ecf3b5e092e1dbed0bfd7651','analyzer_sha256':sha(pathlib.Path(__file__)),'analyzed_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'runs':items,'exact_vs_control':comparisons,'input_sha256':manifest,'paired_test_arrays_sha256':sha(out/'paired-test-arrays.npz'),'bootstrap_rng_seed':20261106,'bootstrap_replicates':2000,'limitations':['Frozen Qwen backbone counterfactual editing, not pretraining or verified factual knowledge','Three seeds and related subjects limit uncertainty; bootstrap intervals are pointwise, not corrected for eight comparisons','Exact per-key rows clone hash initial vectors; this does not establish naturally occurring 99 percent cosine','Keys are canonicalized by the existing compressed tokenizer','Equal physical row parameters, but full bundle includes vocabulary, delta and recipe costs; no native compression or serving superiority is tested']}
save(out/'final-analysis.json',summary)
print('saved',out/'final-analysis.json',flush=True)
