import concurrent.futures, hashlib, json, os, subprocess, sys
from pathlib import Path
work=Path('/llm-align/duyimin/ngram-direct-bags-20261116')
root=Path('/llm-align/duyimin/ngram-controlled-deployment-20261111')
previous=Path('/llm-align/duyimin/ngram-sparse-xor-20261115')
bundles=Path('/llm-align/duyimin/ngram-host-profile-20261112/bundles')
env=dict(os.environ,PYTHONPATH=str(work/'vector-runtime'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
models=['ag_news-exact-zero-s3-lr10','ag_news-exact-uniform-s3-lr10','dbpedia_14-exact-zero-s3-lr10','dbpedia_14-exact-uniform-s3-lr3']
jobs=[]
for model in models:
 dataset=model.split('-')[0]
 cmd=[sys.executable,'-m','tools.benchmark_native_bags','--table',str(bundles/model/'dedup-raw/rows.meft'),'--codes',str(root/'training-registered'/model/'rows.npy'),'--traces',str(previous/'traces'/f'{dataset}.npz'),'--baseline-library',str(work/'lib/baseline-vector.so'),'--library',str(work/'lib/candidate-vector.so'),'--output',str(work/'vector-results'/f'{model}-raw.json'),'--commit','854fd04a83405957dcc8e6eed576d796f9263195','--core',str(88+len(jobs)*2)]
 jobs.append((model+'-raw',cmd))
model=models[0]
cmd=[sys.executable,'-m','tools.benchmark_native_bags','--table',str(root/'application-frontier/models'/model/'bundles/byte-dedup-zstd/rows.meft'),'--codes',str(root/'training-registered'/model/'rows.npy'),'--traces',str(previous/'traces/ag_news.npz'),'--baseline-library',str(work/'lib/baseline-vector.so'),'--library',str(work/'lib/candidate-vector.so'),'--output',str(work/'vector-results'/f'{model}-zstd.json'),'--commit','854fd04a83405957dcc8e6eed576d796f9263195','--core','100','--batch-sizes','32','--repeat','3','--rounds','2','--warmup','1']
jobs.append((model+'-zstd',cmd))
def run(job):
 label,cmd=job
 with (work/'vector-results'/f'{label}.log').open('w') as out:
  p=subprocess.run(cmd,cwd=work/'vector-runtime',env=env,stdout=out,stderr=subprocess.STDOUT)
 print(label,p.returncode,flush=True)
 return {'label':label,'returncode':p.returncode,'command':cmd}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(run,jobs))
(work/'vector-results/run-status.json').write_text(json.dumps(results,indent=2))
files=[work/'runtime/native/meft_rows.c',work/'runtime/native/meft_rows.h',work/'runtime/tools/native_row_store.py',work/'runtime/tools/benchmark_native_bags.py',*work.glob('lib/*.so')]
manifest={'commit':'854fd04a83405957dcc8e6eed576d796f9263195','compiler':subprocess.check_output(['cc','--version'],text=True),'compile':'cc -std=c11 -O3 -fPIC -shared native/meft_rows.c -lz FLAGS -o LIBRARY','flags':'-I'+str(previous/'include')+' -DMEFT_USE_LZ4 -DMEFT_USE_ZSTD -DMEFT_USE_LIBDEFLATE -l:liblz4.so.1 -l:libzstd.so.1 -l:libdeflate.so.0','baseline_extra':'-DMEFT_DISABLE_RAW_BAG -DMEFT_DISABLE_FP32_BAG_VECTOR','headers_manifest':str(previous/'results/build.json'),'files':{str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
(work/'vector-results/build.json').write_text(json.dumps(manifest,indent=2))
