"""Isolated real-task inference workers and validation-only budget selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

from .export_news_classifier import BACKENDS
from .external_cotraining import file_hash
from .text_classifier import TextClassifier
from .train_news_classifier import read_csv, write_json


def peak_rss_bytes():
    # Linux ru_maxrss can retain the parent's high-water mark across fork/exec.
    # VmHWM describes the current executable's mm instead; a large controller
    # must not make a small inference worker appear to need its parent's RAM.
    if sys.platform=='linux':
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmHWM:'):
                return int(line.split()[1])*1024
        raise RuntimeError('current-process VmHWM missing')
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)


def metrics(labels, predicted, losses, selected=None):
    selected = np.ones(len(labels),dtype=bool) if selected is None else np.asarray(selected,dtype=bool)
    labels, predicted, losses = labels[selected], predicted[selected], losses[selected]
    if not len(labels):
        return {'documents':0,'accuracy':None,'macro_f1':None,'cross_entropy':None,'confusion':[[0]*4 for _ in range(4)]}
    confusion = np.zeros((4,4),dtype=np.int64)
    np.add.at(confusion,(labels,predicted),1)
    tp = np.diag(confusion)
    denominator = confusion.sum(0)+confusion.sum(1)
    f1 = np.zeros(4)
    np.divide(2*tp,denominator,out=f1,where=denominator>0)
    return {'documents':len(labels),'accuracy':float((labels==predicted).mean()),
            'macro_f1':float(f1.mean()),'cross_entropy':float(losses.mean()),'confusion':confusion.tolist()}


def worker(cfg):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU worker requires empty CUDA_VISIBLE_DEVICES')
    if cfg.backend.startswith('faiss-') or cfg.backend.startswith('precomputed-hybrid-faiss-'):
        import faiss
        faiss.omp_set_num_threads(1)
    requests = [json.loads(line) for line in cfg.requests.read_text().splitlines()]
    texts = [r['text'] for r in requests]
    labels = np.asarray([r['label'] for r in requests],dtype=np.int64)
    rss = peak_rss_bytes
    before = time.perf_counter()
    model = TextClassifier(cfg.serving/cfg.backend,native_library=cfg.native_library)
    startup_seconds = time.perf_counter()-before
    startup_rss = rss()
    predictions, losses, flags, tokens, oov = [], [], [], [], []
    before = time.perf_counter()
    for start in range(0,len(texts),64):
        end = min(len(texts),start+64)
        result = model.predict(texts[start:end])
        logits = result['logits']
        normalized = logits-logits.max(1,keepdims=True)
        ce = np.log(np.exp(normalized).sum(1))-normalized[np.arange(end-start),labels[start:end]]
        predictions.extend(map(int,result['predictions']))
        losses.extend(map(float,ce))
        flags.extend(map(bool,result['source_prediction_preserved']))
        tokens.extend(map(int,result['token_counts']))
        oov.extend(map(int,result['oov_counts']))
    quality_seconds = time.perf_counter()-before
    predictions, losses = np.asarray(predictions), np.asarray(losses)
    tokens, oov = np.asarray(tokens), np.asarray(oov)
    high_oov = oov/tokens >= .1
    report = {'status':'passed','backend':cfg.backend,'phase':cfg.phase,'source_commit':cfg.source_commit,
        'python':sys.version,'numpy':np.__version__,'os':platform.platform(),
        'command_argv':sys.orig_argv,'requests_sha256':file_hash(cfg.requests),
        'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        'classifier_sha256':file_hash(cfg.serving/cfg.backend/'classifier.json'),
        'peak_rss_measurement':'current-exec VmHWM' if sys.platform=='linux' else 'getrusage ru_maxrss',
        'torch_loaded':'torch' in sys.modules,'startup_seconds':startup_seconds,'startup_peak_rss_bytes':startup_rss,
        'quality_seconds':quality_seconds,'full_quality_documents_per_second':len(texts)/quality_seconds,
        'metrics':metrics(labels,predictions,losses),'oov_heavy_metrics':metrics(labels,predictions,losses,high_oov),
        'source_prediction_preservation_fraction':float(np.mean(flags)),
        'labels':labels.tolist(),'predictions':predictions.tolist(),'losses':losses.tolist(),
        'source_prediction_preserved':flags,'token_counts':tokens.tolist(),'oov_counts':oov.tolist(),
        'timings':{},'complete_model_bytes':sum(p.stat().st_size for p in (cfg.serving/cfg.backend).rglob('*') if p.is_file()),
        'limitations':['Shared warm CPU, one query thread, no exclusive/cold-cache or enforced RSS cap.',
            'Peak process RSS includes NumPy, keys, maps, request text buffers and query scratch.',
            'Prediction preservation is agreement with the source-real model, not ground-truth accuracy.',
            'Artifact-budget eligibility is separate from measured RSS; no dynamic word-vector cache.',
            'Hybrid controls explicitly include precomputed training-word features and an unknown-word subword fallback.']}
    if report['torch_loaded']:
        raise AssertionError('inference worker imported Torch')
    if cfg.phase == 'validation':
        selected = np.random.default_rng(42).permutation(len(texts))[:512]
        for batch in (1,32):
            groups = [[texts[int(i)] for i in selected[start:start+batch]] for start in range(0,512,batch)]
            for requests_batch in groups:
                model.predict(requests_batch)
            samples = []
            for repeat in range(5):
                for requests_batch in groups:
                    before = time.perf_counter()
                    model.predict(requests_batch)
                    samples.append(time.perf_counter()-before)
            report['timings'][str(batch)] = {'batch_size':batch,'repeat':5,'warmups':1,'seconds':samples,
                'p50_seconds':float(np.median(samples)),'p95_seconds':float(np.quantile(samples,.95)),
                'documents_per_second':(512*5)/sum(samples)}
    report['peak_rss_bytes'] = rss()
    report['process_lifetime_ru_maxrss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)
    model.close()
    write_json(cfg.output,report)


def prepare_requests(data, training, output):
    output.mkdir(parents=True,exist_ok=False)
    texts, labels = read_csv(data/'train.csv')
    split = np.load(training/'common/split.npz',allow_pickle=False)
    validation = split['validation']
    split.close()
    for phase, docs, targets in [('validation',[texts[int(i)] for i in validation],labels[validation]),
                                 ('test',*read_csv(data/'test.csv'))]:
        if phase=='test' and len(docs)!=7600:
            raise ValueError('complete 7600-document official test required')
        with (output/(phase+'.jsonl')).open('w') as stream:
            for text,label in zip(docs,targets):
                stream.write(json.dumps({'text':text,'label':int(label)})+'\n')


def select(validation):
    reference = validation['source-mmap']['metrics']
    selections = []
    for kind, field, budgets in [('artifact','complete_model_bytes',(16,32,64,128,256)),
                                 ('observed_rss','peak_rss_bytes',(128,192,256,384,512))]:
        for mib in budgets:
            eligible = [name for name,r in validation.items() if r[field]<=mib*(1<<20)
                and r['metrics']['accuracy']>=reference['accuracy']-.005
                and r['metrics']['cross_entropy']<=reference['cross_entropy']*1.02]
            winner = min(eligible,key=lambda name:validation[name]['timings']['32']['documents_per_second']*-1) if eligible else None
            selections.append({'budget_kind':kind,'budget_mib':mib,'eligible':eligible,'selected':winner})
    return selections


def run(cfg):
    cfg.output.mkdir(parents=True,exist_ok=False)
    report = {'status':'running','seed':cfg.seed,'source_commit':cfg.source_commit,'source_files_sha256':{
        name:file_hash(Path(__file__).with_name(name)) for name in ('benchmark_news_classifier.py','text_classifier.py','export_news_classifier.py')},
        'native_library_sha256':file_hash(cfg.native_library),'phase_order':['validation-all','freeze-selection','test-all'],
        'validation':{},'test':{},'selections':None,'failures':[]}
    backends = [v['backend'] for v in json.loads((cfg.serving/'export.json').read_text())['variants']]
    for phase in ('validation','test'):
        if phase == 'test':
            report['selections'] = select(report['validation'])
            write_json(cfg.output/'selection-before-test.json',{'seed':cfg.seed,'selections':report['selections'],
                'validation_report_sha256':{name:file_hash(cfg.output/(name+'-validation.json')) for name in backends}})
        for backend in backends:
            out = cfg.output/(backend+'-'+phase+'.json')
            command = [sys.executable,'-m','tools.benchmark_news_classifier','worker','--serving',str(cfg.serving),
                '--backend',backend,'--requests',str(cfg.requests/(phase+'.jsonl')),'--phase',phase,
                '--output',str(out),'--source-commit',cfg.source_commit,'--native-library',str(cfg.native_library)]
            with (cfg.output/(backend+'-'+phase+'.log')).open('w') as log:
                completed = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                report['failures'].append({'backend':backend,'phase':phase,'returncode':completed.returncode})
                write_json(cfg.output/'report.json',report)
                raise ValueError('isolated task worker failed: '+backend)
            result = json.loads(out.read_text())
            report[phase][backend] = result
            write_json(cfg.output/'report.json',report)
            print(json.dumps({'phase':phase,'backend':backend,'seed':cfg.seed,'accuracy':result['metrics']['accuracy'],
                              'model_mib':result['complete_model_bytes']/(1<<20),'rss_mib':result['peak_rss_bytes']/(1<<20)}),flush=True)
        reference = report[phase]['source-mmap']
        for backend,result in report[phase].items():
            if result['labels'] != reference['labels'] or result['requests_sha256'] != reference['requests_sha256']:
                raise ValueError('different task requests/labels')
            agreement = np.asarray(result['predictions']) == reference['predictions']
            if backend == 'ngram-bounded' and np.any(np.asarray(result['source_prediction_preserved']) & ~agreement):
                raise AssertionError('a certified prediction disagrees with the evaluated source reference')
            result['source_agreement_fraction'] = float(agreement.mean())
    report['status'] = 'passed'
    write_json(cfg.output/'report.json',report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command',required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--data',type=Path,required=True);prepare.add_argument('--training',type=Path,required=True)
    prepare.add_argument('--output',type=Path,required=True)
    for name in ('worker','run'):
        sub = commands.add_parser(name)
        sub.add_argument('--serving',type=Path,required=True);sub.add_argument('--requests',type=Path,required=True)
        sub.add_argument('--output',type=Path,required=True);sub.add_argument('--source-commit',required=True)
        sub.add_argument('--native-library',type=Path,required=True)
        if name=='worker':
            from .export_news_hybrids import HYBRIDS, AUDIT_HYBRIDS
            sub.add_argument('--backend',choices=[*BACKENDS,*HYBRIDS,*AUDIT_HYBRIDS],required=True)
            sub.add_argument('--phase',choices=('validation','test'),required=True)
        else:
            sub.add_argument('--seed',type=int,required=True)
    cfg = p.parse_args()
    if cfg.command == 'prepare':prepare_requests(cfg.data,cfg.training,cfg.output)
    elif cfg.command == 'worker':worker(cfg)
    else:run(cfg)


if __name__ == '__main__':
    main()
