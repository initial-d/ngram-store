import json,sys,subprocess
from pathlib import Path
from ngram_store import TextClassifier
root=Path('/tmp/ngram-news-shift-20261105')
texts=['The football team won a championship match.', 'Researchers unveiled a new computer chip.', 'Stocks rallied after quarterly earnings.', 'International leaders met to discuss a ceasefire.', 'Researchers unveiled a neuroprocessor for computing.', 'The football team uses ultralightweight equipment.']
results={}
for name in ('precomputed-hybrid-source','precomputed-hybrid-int8','precomputed-hybrid-faiss-sq4','precomputed-hybrid-ngram-int8','precomputed-hybrid-faiss-sq8','precomputed-hybrid-faiss-pq-8x8','precomputed-hybrid-faiss-cluster-8'):
    directory='audit-hybrids' if name in ('precomputed-hybrid-faiss-sq8','precomputed-hybrid-faiss-pq-8x8','precomputed-hybrid-faiss-cluster-8') else 'hybrids'
    with TextClassifier(root/directory/'seed-3'/name) as model:
        out=model.predict(texts)
        results[name]=out['predictions'].tolist()
        oov_counts=out['oov_counts'].tolist()
assert results['precomputed-hybrid-source']==results['precomputed-hybrid-int8']
assert any(oov_counts), 'actual installed fallback must be exercised'
assert 'torch' not in sys.modules
command=[sys.executable,'-m','ngram_store','classify','--store',str(root/'hybrids/seed-3/precomputed-hybrid-ngram-int8'),'--text',*texts]
result=subprocess.run(command,text=True,capture_output=True,check=True)
payload=json.loads(result.stdout)
assert len(payload['predictions'])==len(texts)
print(json.dumps({'status':'passed','installed_package':str(Path(__import__('ngram_store').__file__).resolve()),'api_predictions':results,'oov_counts':oov_counts,'cli_predictions':payload['predictions'],'torch_loaded':False,'input_kind':'six author-written texts, including actual unknown-word fallback; actual AG News-trained bundles'}))
