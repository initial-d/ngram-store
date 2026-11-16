import hashlib,json,math,os,statistics,sys,time
from pathlib import Path
import numpy as np
work=Path('/llm-align/duyimin/ngram-direct-bags-20261116');sys.path.insert(0,str(work/'vector-runtime'))
from tools.native_row_store import NativeRows
os.sched_setaffinity(0,{112})
reports=[]
for model in ('ag_news-exact-zero-s3-lr10','ag_news-exact-uniform-s3-lr10'):
 parent_path=work/'vector-results'/f'{model}-raw.json';parent=json.loads(parent_path.read_text())
 import shlex
 cmd=shlex.split(parent['command'])
 def arg(name):return Path(cmd[cmd.index('--'+name)+1])
 codes=np.load(arg('codes'),mmap_mode='r');path=arg('traces');meta=json.loads(path.with_suffix('.json').read_text());trace=np.load(path)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==parent['hashes']['traces']
 docs=[np.ascontiguousarray(trace['ids'+str(i)],dtype=np.uint64) for i in range(len(meta['requests']))]
 for i,ids in enumerate(docs):assert hashlib.sha256(ids.tobytes()).hexdigest()==meta['requests'][i]['ids_sha256']
 queries=[]
 for begin in range(0,len(docs),32):
  parts=docs[begin:begin+32];ids=np.concatenate(parts);off=np.r_[0,np.cumsum([len(x) for x in parts],dtype=np.uint64)].astype(np.uint64)
  expected=np.array([(codes[x].astype(np.float64).sum(0)/max(len(x),1)).astype(np.float32) for x in parts])
  queries.append((ids,off,expected))
 samples={n:[] for n in ('direct','staged_vector')};libraries={'direct':work/'lib/candidate-vector.so','staged_vector':work/'lib/staged-vector.so'}
 with NativeRows(arg('table'),libraries['direct']) as direct,NativeRows(arg('table'),libraries['staged_vector']) as staged:
  stores={'direct':direct,'staged_vector':staged};outputs={n:[np.empty_like(e) for _,_,e in queries] for n in stores}
  for rep in range(-2,30):
   for i,(ids,off,expected) in enumerate(queries):
    order=('direct','staged_vector') if (rep+i)%2 else ('staged_vector','direct')
    for n in order:
     started=time.perf_counter_ns();stores[n].embedding_bag_array(ids,off,mode='mean',out=outputs[n][i]);elapsed=(time.perf_counter_ns()-started)/1000
     assert outputs[n][i].tobytes()==expected.tobytes()
     if rep>=0:samples[n].append(elapsed)
  staging={n:s.last_bag_staging_bytes for n,s in stores.items()}
 ratio=math.exp(statistics.mean(math.log(a/b) for a,b in zip(samples['staged_vector'],samples['direct'])))
 record={'model':model,'direct_vs_staged_vector_speed_ratio':ratio,'p50_us':{n:statistics.median(v) for n,v in samples.items()},'samples_us':samples,'staging_bytes':staging,'source_features_byte_match':True,'parent_report_sha256':hashlib.sha256(parent_path.read_bytes()).hexdigest(),'parent_input_hashes':parent['hashes'],'provenance':'Full table/code hashes inherited from the just-completed frozen-input parent run, not recomputed. Original trace hashes checked again; every result compared with freshly read source FP64-to-FP32 features.','libraries':{n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in libraries.items()},'cpu_affinity':[112],'batch_size':32,'repeat':30,'warmup':2,'commit':'854fd04a83405957dcc8e6eed576d796f9263195'}
 reports.append(record);print(model,ratio,record['p50_us'],flush=True)
(work/'vector-results/fusion-ablation.json').write_text(json.dumps(reports,indent=2))
