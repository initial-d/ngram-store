#!/bin/bash
set -euo pipefail
TASK_ROOT=/llm-align/duyimin/ngram-news-app-20261105
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$TASK_ROOT/final-source"
/llm-align/duyimin/miniconda3/envs/gemma4/bin/python - <<'PY'
from pathlib import Path
import hashlib, importlib.metadata, json, os, platform, re, subprocess, sys
import numpy as np
from tools.text_classifier import tokenize, ngrams
from tools.train_news_classifier import read_csv
root = Path('/llm-align/duyimin/ngram-news-app-20261105')
out = root/'public-metadata'
out.mkdir(exist_ok=True)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value): (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
lex = json.loads((root/'training/common/lexicon.json').read_text())
words, grams = set(lex['tokens']), set(lex['grams'])
for name in ('data.json','lexicon.json','word-ids.u64','word-offsets.u64'):
    assert digest(root/'training/common'/name)==digest(root/'training-relu/common'/name), name
train_texts, labels = read_csv(root/'data/train.csv')
test_texts, _ = read_csv(root/'data/test.csv')
split = np.load(root/'training/common/split.npz')
def audit(texts,keep_arrays):
    tokens, unknown, truncated, empty, missing, candidates = [],[],[],[],[],[]
    cache = {}
    total_original = 0
    for text in texts:
        parsed, trunc = tokenize(text,lex['max_tokens'])
        total_original += len(re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower()))
        oov = [w for w in parsed if w not in words]
        counts = []
        for w in oov:
            if w not in cache:
                gs = ngrams(w,lex['max_order'])
                absent = sum(g not in grams for g in gs)
                cache[w] = (len(gs),absent,int(absent==len(gs)))
            counts.append(cache[w])
        tokens.append(len(parsed)); unknown.append(len(oov)); truncated.append(trunc)
        candidates.append(sum(c[0] for c in counts)); missing.append(sum(c[1] for c in counts)); empty.append(sum(c[2] for c in counts))
    result = {'documents':len(texts),'original_tokens':total_original,'retained_tokens':sum(tokens),
        'unknown_token_occurrences':sum(unknown),'truncated_documents':sum(truncated),
        'oov_heavy_documents':sum(o/t>=.1 for o,t in zip(unknown,tokens)),
        'unknown_word_ngram_candidates':sum(candidates),'missing_unknown_ngram_occurrences':sum(missing),
        'unknown_tokens_with_empty_bags':sum(empty),'empty_documents':sum(t==0 for t in tokens)}
    if keep_arrays: result.update(token_counts=tokens,oov_counts=unknown,truncated=truncated,
        missing_ngram_counts=missing,unknown_empty_bag_counts=empty)
    return result
write('preprocessing-audit.json',{'scope':'Post-run input audit only; no model/selection/threshold changes.',
    'registry_sha256':digest(root/'training/common/lexicon.json'),
    'train':audit([train_texts[int(i)] for i in split['train']],False),
    'validation':audit([train_texts[int(i)] for i in split['validation']],True),'test':audit(test_texts,True)})
cpu = Path('/proc/cpuinfo').read_text()
write('environment.json',{'python':sys.version,'os':platform.platform(),'machine':platform.machine(),
    'cpu_model':next(line.split(':',1)[1].strip() for line in cpu.splitlines() if line.startswith('model name')),
    'logical_cpus':os.cpu_count(),'mem_total':next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')),
    'versions':{p:importlib.metadata.version(p) for p in ('numpy','torch','faiss-cpu','matplotlib')},
    'compiler':subprocess.check_output(['cc','--version'],text=True).splitlines()[0],
    'native_library_sha256':digest(root/'final-source/build/libmeft_rows.so'),
    'measured_source_sha256':{str(p.relative_to(root/'final-source')):digest(p) for p in [
        root/'final-source/tools'/name for name in ('train_news_classifier.py','text_classifier.py','export_news_classifier.py','benchmark_news_classifier.py')]+[root/'final-source/native/meft_rows.c']},
    'inference_threads':1,'training_threads':4,'training_interop_threads':1,
    'cuda_visible_devices':'','scope':'Shared CPU host, no GPU inspection/use, no process stops, no exclusive or cold-cache claims.',
    'cost_assumptions':'No GPU, energy or monetary estimate; artifact and observed RSS thresholds are not OS-enforced limits.'})
print('Public metadata and preprocessing audit written; no text or weights copied.')
PY
