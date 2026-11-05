"""Recompute external-news metrics, bootstrap intervals and choices using stdlib."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median

from .analyze_news_classifier import near, read, validate_metrics

HYBRIDS={'precomputed-hybrid-source','precomputed-hybrid-int8','precomputed-hybrid-faiss-sq4','precomputed-hybrid-ngram-int8'}
BACKENDS={'source-resident','source-mmap','fp16','int8','faiss-sq4','faiss-sq8','faiss-pq-8x8',
          'faiss-cluster-8','ngram-bounded','ngram-int8','precomputed-words'}|HYBRIDS


def passes(record, reference):
    metric, source=record['metrics'],reference['metrics']
    return (metric['accuracy']>=source['accuracy']-.005 and metric['macro_f1']>=source['macro_f1']-.005
            and metric['cross_entropy']<=source['cross_entropy']*1.02)


def choices(validation):
    result=[]
    for kind,field,budgets in [('artifact','complete_model_bytes',(16,32,48,64,96,128,256)),
                             ('observed_rss','peak_rss_bytes',(64,128,192,256,384))]:
        for mib in budgets:
            eligible=[name for name,r in validation.items()
                      if r[field]<=mib*(1<<20) and passes(r,validation['source-mmap'])]
            selected=max(eligible,key=lambda k:validation[k]['timings']['32']['documents_per_second']) if eligible else None
            result.append({'budget_kind':kind,'budget_mib':mib,'eligible':eligible,'selected':selected})
    return result


def quantile(values, fraction):
    ordered=sorted(values);position=(len(ordered)-1)*fraction
    lo,hi=math.floor(position),math.ceil(position)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


def validate_run(run):
    if (run['status']!='passed' or run['failures'] or set(run['validation'])!=BACKENDS or set(run['test'])!=BACKENDS
            or run['phase_order']!=['validation-all','freeze-selection','test-all']):
        raise ValueError('all 15 controls and validation-before-test execution required')
    for phase in ('validation','test'):
        source=run[phase]['source-mmap'];n=run['data']['phase_stats'][phase]['documents']
        for backend,record in run[phase].items():
            if (record['status']!='passed' or record['torch_loaded'] or record['backend']!=backend
                    or record['phase']!=phase or len(record['labels'])!=n
                    or record['requests_sha256']!=run['data']['requests_sha256'][phase]
                    or record['labels']!=source['labels'] or record['token_counts']!=source['token_counts']
                    or record['oov_counts']!=source['oov_counts']):
                raise ValueError('different inputs/registry, failed worker or Torch inference')
            validate_metrics(record)
            validate_metrics(record,[o/t>=.1 for o,t in zip(record['oov_counts'],record['token_counts'])])
            flags=record['source_prediction_preserved']
            agreement=[a==b for a,b in zip(record['predictions'],source['predictions'])]
            if not near(record['source_agreement_fraction'],sum(agreement)/n):
                raise ValueError('source agreement differs from paired predictions')
            if not near(record['source_prediction_preservation_fraction'],sum(flags)/n):
                raise ValueError('preservation count differs from flags')
            if backend=='ngram-bounded' and any(f and not a for f,a in zip(flags,agreement)):
                raise ValueError('a certified prediction disagrees with evaluated source')
            if backend!='ngram-bounded' and any(flags):
                raise ValueError('uncertified backend emitted a certificate')
            if record['complete_model_bytes']<=0 or record['peak_rss_bytes']<record['startup_peak_rss_bytes']:
                raise ValueError('invalid model/RSS accounting')
            if record['os'].startswith('Linux') and record['peak_rss_measurement']!='current-exec VmHWM':
                raise ValueError('Linux RSS must exclude inherited parent peak')
            if not near(record['full_quality_documents_per_second'],n/record['quality_seconds']):
                raise ValueError('full traversal throughput mismatch')
            if phase=='validation':
                for batch in (1,32):
                    timing=record['timings'][str(batch)];samples=timing['seconds']
                    if timing['repeat']!=5 or timing['warmups']!=1 or len(samples)!=2560//batch:
                        raise ValueError('fixed full-request timing protocol required')
                    if any(not math.isfinite(v) or v<=0 for v in samples):
                        raise ValueError('nonpositive/nonfinite timing')
                    if not (near(timing['documents_per_second'],2560/math.fsum(samples))
                            and near(timing['p50_seconds'],median(samples))
                            and near(timing['p95_seconds'],quantile(samples,.95))):
                        raise ValueError('timing summary differs from observations')
        if run[phase]['source-resident']['predictions']!=source['predictions']:
            raise ValueError('source mmap/resident differ')
        if run[phase]['precomputed-hybrid-source']['predictions']!=source['predictions']:
            raise ValueError('FP32 hybrid changed complete source predictions')
        stats=run['data']['phase_stats'][phase]
        if (sum(source['token_counts'])!=stats['tokens'] or sum(source['oov_counts'])!=stats['oov_tokens']
                or source['oov_heavy_metrics']['documents']!=stats['oov_heavy_documents']):
            raise ValueError('request preparation and runtime token counts differ')
    if run['selections']!=choices(run['validation']):
        raise ValueError('choices were not derived only from calibration metrics')


def validate_bootstrap(runs, backend, reference, record, oov):
    source=runs[0]['test'][reference]
    ids=[i for i,(o,t) in enumerate(zip(source['oov_counts'],source['token_counts'])) if not oov or o/t>=.1]
    delta=[]
    for i in ids:
        delta.append(sum(int(run['test'][backend]['predictions'][i]==source['labels'][i])
                         -int(run['test'][reference]['predictions'][i]==source['labels'][i]) for run in runs))
    if record['documents']!=len(ids):
        raise ValueError('bootstrap resampled the wrong document slice')
    if not ids:
        if record['difference_pp'] is not None or record['ci95_pp'] is not None or record['draws']:
            raise ValueError('empty bootstrap slice must remain empty')
        return
    counts=Counter(delta);histogram=[counts[k] for k in range(-3,4)]
    if record['histogram']!=histogram or record['resamples']!=2000 or record['rng_seed']!=42:
        raise ValueError('bootstrap histogram or frozen resampling parameters changed')
    if not near(record['difference_pp'],sum(delta)/(3*len(ids))*100):
        raise ValueError('paired mean accuracy change differs from predictions')
    draws=record['draws'];values=[]
    if len(draws)!=2000:
        raise ValueError('complete 2000-bootstrap draws required')
    for draw in draws:
        if len(draw)!=7 or any(type(v) is not int or v<0 for v in draw) or sum(draw)!=len(ids):
            raise ValueError('invalid document-resampling counts')
        if any(v and not h for v,h in zip(draw,histogram)):
            raise ValueError('resampled an absent outcome')
        values.append(sum(k*v for k,v in zip(range(-3,4),draw))/(3*len(ids))*100)
    if not all(near(v,q) for v,q in zip(record['ci95_pp'],(quantile(values,.025),quantile(values,.975)))):
        raise ValueError('bootstrap confidence interval differs from saved draws')


def summarize(runs, bootstrap):
    if len(runs)!=3 or {r['seed'] for r in runs}!={3,7,11}:
        raise ValueError('three predeclared fitted seeds required')
    for run in runs:
        validate_run(run)
        if run['data']!=runs[0]['data']:
            raise ValueError('the same external input registry/split must be reused')
    expected_zero={'source-mmap'}|HYBRIDS
    if set(bootstrap['source_comparisons'])!=BACKENDS or set(bootstrap['zero_oov_comparisons'])!=expected_zero:
        raise ValueError('all paired comparisons must be retained')
    for group,reference in (('source_comparisons','source-mmap'),('zero_oov_comparisons','precomputed-words')):
        for backend,slices in bootstrap[group].items():
            if set(slices)!={'all','oov_heavy'}:
                raise ValueError('both fixed slices required')
            for name,record in slices.items():
                validate_bootstrap(runs,backend,reference,record,name=='oov_heavy')
    rows=[]
    for backend in runs[0]['test']:
        tested=[r['test'][backend] for r in runs];validated=[r['validation'][backend] for r in runs]
        rows.append({'backend':backend,'model_mib':median(r['complete_model_bytes']/(1<<20) for r in tested),
            'peak_rss_mib':median(r['peak_rss_bytes']/(1<<20) for r in tested),
            'accuracy':median(r['metrics']['accuracy'] for r in tested),
            'macro_f1':median(r['metrics']['macro_f1'] for r in tested),
            'cross_entropy':median(r['metrics']['cross_entropy'] for r in tested),
            'accuracy_by_seed':{str(run['seed']):run['test'][backend]['metrics']['accuracy'] for run in runs},
            'oov_heavy_documents':tested[0]['oov_heavy_metrics']['documents'],
            'oov_heavy_accuracy':median(r['oov_heavy_metrics']['accuracy'] for r in tested) if tested[0]['oov_heavy_metrics']['documents'] else None,
            'source_disagreements':sum(sum(a!=b for a,b in zip(run['test'][backend]['predictions'],run['test']['source-mmap']['predictions'])) for run in runs),
            'source_preserved':sum(sum(r['source_prediction_preserved']) for r in tested),
            'total_test_predictions':sum(len(r['labels']) for r in tested),
            'batch32_docs_per_second':median(r['timings']['32']['documents_per_second'] for r in validated),
            'batch1_p50_ms':median(r['timings']['1']['p50_seconds']*1000 for r in validated),
            'batch1_p95_ms':median(r['timings']['1']['p95_seconds']*1000 for r in validated),
            'full_test_docs_per_second':median(r['full_quality_documents_per_second'] for r in tested),
            'source_accuracy_difference_pp':bootstrap['source_comparisons'][backend]['all']['difference_pp'],
            'source_accuracy_difference_ci95_pp':bootstrap['source_comparisons'][backend]['all']['ci95_pp']})
    audits=[]
    for run in runs:
        source=run['test']['source-mmap']
        for choice in run['selections']:
            field='complete_model_bytes' if choice['budget_kind']=='artifact' else 'peak_rss_bytes'
            selected=choice['selected'];record=run['test'].get(selected)
            audits.append({**choice,'seed':run['seed'],
                'test_budget_pass':None if record is None else record[field]<=choice['budget_mib']*(1<<20),
                'test_quality_pass':None if record is None else passes(record,source)})
    necessity=[]
    for run in runs:
        test=run['test'];source=test['source-mmap'];name='precomputed-hybrid-ngram-int8'
        for choice in run['selections']:
            if choice['budget_kind']!='artifact':
                continue
            budget=choice['budget_mib']*(1<<20)
            def eligible(phase,backend):
                return phase[backend]['complete_model_bytes']<=budget and passes(phase[backend],phase['source-mmap'])
            # Every other subword-preserving backend is checked, not only the
            # four hybrids; direct FAISS may invalidate an exclusivity claim.
            competitors=BACKENDS-{'precomputed-words',name,'ngram-int8','ngram-bounded'}
            unique=(eligible(run['validation'],name) and eligible(test,name)
                and not passes(run['validation']['precomputed-words'],run['validation']['source-mmap'])
                and not passes(test['precomputed-words'],source)
                and not any(eligible(run['validation'],other) or eligible(test,other) for other in competitors))
            necessity.append({'seed':run['seed'],'budget_mib':choice['budget_mib'],'ngram_route_unique':unique,
                'other_eligible_test_backends':[other for other in sorted(competitors) if eligible(test,other)]})
    brief_bootstrap={group:{backend:{name:{k:v for k,v in record.items() if k!='draws'}
                    for name,record in slices.items()} for backend,slices in entries.items()}
                    for group,entries in bootstrap.items()}
    return {'schema_version':1,'data':runs[0]['data'],'rows':rows,'budget_audits':audits,
        'necessity_audits':necessity,'paired_accuracy':brief_bootstrap,
        'scope':'Author-run cross-source temporal CPU news routing with frozen AG News model settings; no universal or production claim.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--bootstrap',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);cfg=p.parse_args()
    def locate(seed):
        path=cfg.results/f'seed-{seed}/report.json'
        return read(path if path.exists() else path.with_suffix('.json.gz'))
    result=summarize([locate(seed) for seed in (3,7,11)],read(cfg.bootstrap))
    cfg.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'rows':result['rows'],'budget_audits':result['budget_audits']},indent=2))


if __name__=='__main__':
    main()
