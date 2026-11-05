"""Validate public numeric news-task evidence without NumPy, Torch or datasets."""
import argparse
import gzip
import json
import math
from pathlib import Path
from statistics import median


BASE = {'source-resident','source-mmap','fp16','int8','faiss-sq4','faiss-sq8','faiss-pq-8x8',
        'faiss-cluster-8','ngram-bounded','ngram-int8','precomputed-words'}


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())


def near(a,b):
    return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14)


def validate_metrics(record, mask=None):
    labels, predictions, losses = record['labels'],record['predictions'],record['losses']
    n = len(labels)
    if any(len(record[k])!=n for k in ('predictions','losses','source_prediction_preserved','token_counts','oov_counts')):
        raise ValueError('one result per request required')
    if any(not isinstance(x,int) or not 0<=x<4 for a in (labels,predictions) for x in a):
        raise ValueError('four-class labels/predictions required')
    if any(not math.isfinite(v) or v<0 for v in losses):
        raise ValueError('finite nonnegative loss required')
    if any(type(v) is not bool for v in record['source_prediction_preserved']):
        raise ValueError('boolean preservation flags required')
    if any(type(v) is not int for a in (record['token_counts'],record['oov_counts']) for v in a):
        raise ValueError('integer token/OOV counts required')
    if any(t<1 or not 0<=o<=t for t,o in zip(record['token_counts'],record['oov_counts'])):
        raise ValueError('invalid OOV/token audit')
    selected = list(range(n)) if mask is None else [i for i,v in enumerate(mask) if v]
    confusion = [[0]*4 for _ in range(4)]
    for i in selected:
        confusion[labels[i]][predictions[i]] += 1
    metric = record['metrics'] if mask is None else record['oov_heavy_metrics']
    if metric['documents']!=len(selected) or metric['confusion']!=confusion:
        raise ValueError('confusion/count mismatch')
    if not selected:
        if any(metric[k] is not None for k in ('accuracy','macro_f1','cross_entropy')):
            raise ValueError('empty slice must stay empty')
        return
    accuracy = sum(labels[i]==predictions[i] for i in selected)/len(selected)
    f1 = []
    for k in range(4):
        denominator = sum(confusion[k])+sum(row[k] for row in confusion)
        f1.append(2*confusion[k][k]/denominator if denominator else 0.)
    if not (near(metric['accuracy'],accuracy) and near(metric['macro_f1'],sum(f1)/4)
            and near(metric['cross_entropy'],math.fsum(losses[i] for i in selected)/len(selected))):
        raise ValueError('task metric differs from numeric observations')


def choices(validation):
    reference = validation['source-mmap']['metrics']
    result = []
    for kind,field,budgets in [('artifact','complete_model_bytes',(16,32,64,128,256)),
                             ('observed_rss','peak_rss_bytes',(128,192,256,384,512))]:
        for mib in budgets:
            eligible = [name for name,r in validation.items() if r[field]<=mib*(1<<20)
                        and r['metrics']['accuracy']>=reference['accuracy']-.005
                        and r['metrics']['cross_entropy']<=reference['cross_entropy']*1.02]
            selected = max(eligible,key=lambda k:validation[k]['timings']['32']['documents_per_second']) if eligible else None
            result.append({'budget_kind':kind,'budget_mib':mib,'eligible':eligible,'selected':selected})
    return result


def validate_run(run, task):
    expected = BASE | ({'linear-fused'} if task=='linear' else set())
    if (run['status']!='passed' or run['failures'] or set(run['validation'])!=expected
            or set(run['test'])!=expected or run['phase_order']!=['validation-all','freeze-selection','test-all']):
        raise ValueError('complete predeclared validation/test controls required')
    for phase,n in (('validation',12000),('test',7600)):
        source = run[phase]['source-mmap']
        for backend,record in run[phase].items():
            if (record['status']!='passed' or record['torch_loaded'] or record['backend']!=backend
                    or len(record['labels'])!=n or record['phase']!=phase
                    or record['requests_sha256']!=source['requests_sha256'] or record['labels']!=source['labels']):
                raise ValueError('different requests or nonisolated/failed inference')
            if record['os'].startswith('Linux') and record.get('peak_rss_measurement')!='current-exec VmHWM':
                raise ValueError('Linux worker RSS must exclude inherited pre-exec high-water')
            if record['token_counts']!=source['token_counts'] or record['oov_counts']!=source['oov_counts']:
                raise ValueError('different tokenization or OOV definition')
            validate_metrics(record)
            validate_metrics(record,[o/t>=.1 for t,o in zip(record['token_counts'],record['oov_counts'])])
            agreement = [a==b for a,b in zip(record['predictions'],source['predictions'])]
            if not near(record['source_agreement_fraction'],sum(agreement)/n):
                raise ValueError('source-agreement mismatch')
            flags = record['source_prediction_preserved']
            if not near(record['source_prediction_preservation_fraction'],sum(flags)/n):
                raise ValueError('preservation count mismatch')
            if backend=='ngram-bounded' and any(f and not a for f,a in zip(flags,agreement)):
                raise ValueError('a certified prediction disagrees with source')
            if backend!='ngram-bounded' and any(flags):
                raise ValueError('uncertified codec has no residual certificate')
            if record['complete_model_bytes']<=0 or record['peak_rss_bytes']<record['startup_peak_rss_bytes']:
                raise ValueError('invalid complete storage/process RSS accounting')
            if not near(record['full_quality_documents_per_second'],n/record['quality_seconds']):
                raise ValueError('full-task throughput mismatch')
            if phase=='validation':
                for batch in (1,32):
                    timing = record['timings'][str(batch)]
                    samples = timing['seconds']
                    if timing['repeat']!=5 or timing['warmups']!=1 or len(samples)!=2560//batch:
                        raise ValueError('predeclared complete-request timings required')
                    if any(not math.isfinite(v) or v<=0 for v in samples):
                        raise ValueError('invalid timing sample')
                    ordered = sorted(samples)
                    q = (len(samples)-1)*.95
                    lo,hi = math.floor(q),math.ceil(q)
                    p95 = ordered[lo]+(ordered[hi]-ordered[lo])*(q-lo)
                    if not (near(timing['p50_seconds'],median(samples)) and near(timing['p95_seconds'],p95)
                            and near(timing['documents_per_second'],2560/sum(samples))):
                        raise ValueError('request timing summary mismatch')
        if phase=='test' and [source['labels'].count(k) for k in range(4)]!=[1900]*4:
            raise ValueError('complete balanced official test required')
        if run[phase]['source-resident']['predictions']!=source['predictions']:
            raise ValueError('source resident/mmap controls differ')
    if run['selections']!=choices(run['validation']):
        raise ValueError('budget selection differs from validation observations')
    return run


def summarize(tasks):
    summary = {'tasks':{},'all_source_preservation_checks_pass':True,
               'scope':'Author-run full AG News CPU classification; nonlinear primary, linear-folding and OOV-ablation limits retained.'}
    for task,runs in tasks.items():
        if len(runs)!=3 or {r['seed'] for r in runs}!={3,7,11}:
            raise ValueError('three predeclared seeds required')
        for run in runs:validate_run(run,task)
        rows = []
        for backend in runs[0]['test']:
            tested = [r['test'][backend] for r in runs]
            validated = [r['validation'][backend] for r in runs]
            rows.append({'backend':backend,
                'accuracy':median(r['metrics']['accuracy'] for r in tested),
                'macro_f1':median(r['metrics']['macro_f1'] for r in tested),
                'cross_entropy':median(r['metrics']['cross_entropy'] for r in tested),
                'accuracy_difference_pp_median':median((run['test'][backend]['metrics']['accuracy']-
                    run['test']['source-mmap']['metrics']['accuracy'])*100 for run in runs),
                'accuracy_by_seed':{str(run['seed']):run['test'][backend]['metrics']['accuracy'] for run in runs},
                'oov_heavy_documents':median(r['oov_heavy_metrics']['documents'] for r in tested),
                'oov_heavy_accuracy':median(r['oov_heavy_metrics']['accuracy'] for r in tested) if tested[0]['oov_heavy_metrics']['documents'] else None,
                'source_agreement':median(r['source_agreement_fraction'] for r in tested),
                'pooled_test_documents':sum(len(r['labels']) for r in tested),
                'pooled_source_disagreements':sum(sum(a!=b for a,b in zip(run['test'][backend]['predictions'],
                    run['test']['source-mmap']['predictions'])) for run in runs),
                'pooled_source_preserved':sum(sum(r['source_prediction_preserved']) for r in tested),
                'source_disagreements_by_seed':{str(run['seed']):sum(a!=b for a,b in zip(
                    run['test'][backend]['predictions'],run['test']['source-mmap']['predictions'])) for run in runs},
                'source_preserved_fraction':median(r['source_prediction_preservation_fraction'] for r in tested),
                'source_uncertified_by_seed':{str(run['seed']):sum(not f for f in run['test'][backend]['source_prediction_preserved'])
                    for run in runs} if backend=='ngram-bounded' else None,
                'model_mib':median(r['complete_model_bytes']/(1<<20) for r in tested),
                'peak_rss_mib':median(r['peak_rss_bytes']/(1<<20) for r in tested),
                'single_request_p50_ms':median(r['timings']['1']['p50_seconds']*1000 for r in validated),
                'batch32_docs_per_second':median(r['timings']['32']['documents_per_second'] for r in validated),
                'batch32_docs_per_second_range':[min(r['timings']['32']['documents_per_second'] for r in validated),
                                                max(r['timings']['32']['documents_per_second'] for r in validated)],
                'full_test_docs_per_second':median(r['full_quality_documents_per_second'] for r in tested)})
        budgets = []
        for index,choice in enumerate(runs[0]['selections']):
            entry = {'budget_kind':choice['budget_kind'],'budget_mib':choice['budget_mib'],'seeds':[]}
            for run in runs:
                selection = run['selections'][index]
                record = None if selection['selected'] is None else run['test'][selection['selected']]
                reference = run['test']['source-mmap']['metrics']
                field = 'complete_model_bytes' if choice['budget_kind']=='artifact' else 'peak_rss_bytes'
                entry['seeds'].append({'seed':run['seed'],'selected':selection['selected'],
                    'test_budget_pass':None if record is None else record[field]<=choice['budget_mib']*(1<<20),
                    'test_accuracy':None if record is None else record['metrics']['accuracy'],
                    'test_quality_gate_pass':None if record is None else
                       (record['metrics']['accuracy']>=reference['accuracy']-.005 and record['metrics']['cross_entropy']<=reference['cross_entropy']*1.02)})
            budgets.append(entry)
        summary['tasks'][task] = {'rows':rows,'budget_audits':budgets}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    cfg = p.parse_args()
    def report(task,seed):
        path = cfg.results/task/f'seed-{seed}/report.json'
        return read(path if path.exists() else path.with_suffix('.json.gz'))
    tasks = {task:[report(task,seed) for seed in (3,7,11)] for task in ('linear','relu')}
    result = summarize(tasks)
    cfg.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
