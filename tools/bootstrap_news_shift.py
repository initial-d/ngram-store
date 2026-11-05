"""Paired document bootstrap, retaining all integer resampling counts for audit."""
import argparse
from pathlib import Path

import numpy as np

from .analyze_news_classifier import read
from .train_news_classifier import write_json


def bootstrap(runs, backend, reference='source-mmap', *, oov=False):
    records=[r['test'] for r in runs]
    labels=np.asarray(records[0][reference]['labels'])
    # Sum correct/incorrect differences across the same three fitted seeds per
    # document; one resampled unit is a document, never one seed prediction.
    delta=np.zeros(len(labels),dtype=np.int64)
    for record in records:
        if record[reference]['labels']!=labels.tolist():
            raise ValueError('paired document IDs/labels differ')
        delta+=(np.asarray(record[backend]['predictions'])==labels).astype(np.int64)
        delta-=(np.asarray(record[reference]['predictions'])==labels).astype(np.int64)
    if oov:
        src=records[0][reference]
        delta=delta[np.asarray(src['oov_counts'])/np.asarray(src['token_counts'])>=.1]
    if not len(delta):
        return {'documents':0,'difference_pp':None,'ci95_pp':None,'histogram':[],'draws':[]}
    bins=np.arange(-3,4)
    histogram=np.bincount(delta+3,minlength=7)
    draws=np.random.default_rng(42).multinomial(len(delta),histogram/len(delta),size=2000)
    values=draws@bins/(3*len(delta))*100
    return {'documents':len(delta),'difference_pp':float(delta.mean()/3*100),
        'ci95_pp':np.quantile(values,[.025,.975]).tolist(),'histogram':histogram.tolist(),
        'draws':draws.tolist(),'resamples':2000,'rng_seed':42,
        'interpretation':'95% paired document bootstrap conditional on the three fitted models; no seed-independence assumption.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    cfg=p.parse_args()
    runs=[read(cfg.results/f'seed-{seed}/report.json') for seed in (3,7,11)]
    result={'source_comparisons':{},'zero_oov_comparisons':{}}
    for backend in runs[0]['test']:
        result['source_comparisons'][backend]={slice_name:bootstrap(runs,backend,oov=oov)
            for slice_name,oov in (('all',False),('oov_heavy',True))}
    for backend in ('source-mmap','precomputed-hybrid-source','precomputed-hybrid-int8',
                    'precomputed-hybrid-faiss-sq4','precomputed-hybrid-ngram-int8'):
        result['zero_oov_comparisons'][backend]={slice_name:bootstrap(runs,backend,'precomputed-words',oov=oov)
            for slice_name,oov in (('all',False),('oov_heavy',True))}
    write_json(cfg.output,result)


if __name__=='__main__':
    main()
