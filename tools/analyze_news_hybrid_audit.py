"""Stdlib check of the post-primary FAISS-hybrid audit, without article text."""
from statistics import median

from .analyze_news_classifier import near, validate_metrics
from .analyze_news_shift import choices, passes

EXTRA={'precomputed-hybrid-faiss-sq8','precomputed-hybrid-faiss-pq-8x8','precomputed-hybrid-faiss-cluster-8'}


def summarize(primary, audits):
    if {r['seed'] for r in audits}!={3,7,11} or len(audits)!=3:
        raise ValueError('three supplemental fitted-seed reports required')
    sources={r['seed']:r for r in primary}
    for audit in audits:
        base=sources[audit['seed']]
        if (audit['status']!='passed' or audit['failures'] or audit['trained_source_sha256']!=base['trained_source_sha256']
                or audit['phase_order']!=['supplemental-validation','freeze-supplemental-selection','supplemental-test']):
            raise ValueError('same frozen models and pre-test supplemental choices required')
        for phase in ('validation','test'):
            if set(audit[phase])!=EXTRA:
                raise ValueError('all remaining FAISS codecs require the same fast known-word path')
            source=base[phase]['source-mmap'];n=len(source['labels'])
            for name,record in audit[phase].items():
                if (record['status']!='passed' or record['torch_loaded'] or record['backend']!=name
                        or record['labels']!=source['labels'] or record['requests_sha256']!=source['requests_sha256']
                        or record['token_counts']!=source['token_counts'] or record['oov_counts']!=source['oov_counts']):
                    raise ValueError('supplemental inputs, registry or isolation differ')
                validate_metrics(record)
                validate_metrics(record,[o/t>=.1 for o,t in zip(record['oov_counts'],record['token_counts'])])
                if any(record['source_prediction_preserved']):
                    raise ValueError('FAISS hybrids have no source-preservation certificate')
                agreement=sum(a==b for a,b in zip(record['predictions'],source['predictions']))/n
                if not near(record['source_agreement_fraction'],agreement):
                    raise ValueError('source agreement differs from actual predictions')
                if any(record['thread_environment'].get(k)!='1' for k in
                       ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')):
                    raise ValueError('same explicit CPU runtime thread limits required')
                if phase=='validation':
                    for batch in (1,32):
                        t=record['timings'][str(batch)]
                        if len(t['seconds'])!=2560//batch or t['repeat']!=5 or t['warmups']!=1:
                            raise ValueError('same full-request timing protocol required')
                        if not near(t['documents_per_second'],2560/sum(t['seconds'])):
                            raise ValueError('supplemental throughput mismatch')
        if audit['selections']!=choices({**base['validation'],**audit['validation']}):
            raise ValueError('supplemental choices must use only calibration metrics')
    rows=[]
    for name in sorted(EXTRA):
        tested=[a['test'][name] for a in audits];validated=[a['validation'][name] for a in audits]
        rows.append({'backend':name,'model_mib':median(r['complete_model_bytes']/(1<<20) for r in tested),
            'peak_rss_mib':median(r['peak_rss_bytes']/(1<<20) for r in tested),
            'accuracy':median(r['metrics']['accuracy'] for r in tested),
            'macro_f1':median(r['metrics']['macro_f1'] for r in tested),
            'cross_entropy':median(r['metrics']['cross_entropy'] for r in tested),
            'batch32_docs_per_second':median(r['timings']['32']['documents_per_second'] for r in validated),
            'source_disagreements':sum(sum(x!=y for x,y in zip(a['test'][name]['predictions'],sources[a['seed']]['test']['source-mmap']['predictions'])) for a in audits),
            'calibration_quality_by_seed':{str(a['seed']):passes(a['validation'][name],sources[a['seed']]['validation']['source-mmap']) for a in audits},
            'test_quality_by_seed':{str(a['seed']):passes(a['test'][name],sources[a['seed']]['test']['source-mmap']) for a in audits}})
    return {'study_role':'Explicit post-primary strong-control audit; no source/codec/input/gate changes.',
            'rows':rows,'selections_by_seed':{str(a['seed']):a['selections'] for a in audits}}
