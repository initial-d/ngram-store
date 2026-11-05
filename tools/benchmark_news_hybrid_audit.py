"""Post-primary strong-control audit: give every FAISS codec precomputed words.

Preserve the original 15-control study. Reuse its models, codec training and
inputs; this explicitly retrospective audit adds no outcome-dependent tuning.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .benchmark_news_shift import select
from .export_news_hybrids import AUDIT_HYBRIDS
from .external_cotraining import file_hash
from .train_news_classifier import write_json


def run(cfg):
    primary=json.loads(cfg.primary.read_text())
    exported=json.loads((cfg.hybrids/'export.json').read_text())
    if (primary['status']!='passed' or exported['status']!='passed'
            or primary['trained_source_sha256']!=exported['source_sha256']
            or {v['backend'] for v in exported['variants']}!=set(AUDIT_HYBRIDS)):
        raise ValueError('same passed source and all three additional FAISS hybrids required')
    cfg.output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','seed':primary['seed'],'source_commit':cfg.source_commit,
        'primary_report_sha256':file_hash(cfg.primary),'trained_source_sha256':primary['trained_source_sha256'],
        'phase_order':['supplemental-validation','freeze-supplemental-selection','supplemental-test'],
        'study_role':'Post-primary strong-baseline audit; original 15-control study retained unchanged.',
        'validation':{},'test':{},'selections':None,'failures':[]}
    for phase in ('validation','test'):
        if phase=='test':
            report['selections']=select({**primary['validation'],**report['validation']})
            write_json(cfg.output/'selection-before-test.json',{'seed':primary['seed'],
                'primary_report_sha256':file_hash(cfg.primary),'selections':report['selections'],
                'validation_report_sha256':{name:file_hash(cfg.output/(name+'-validation.json')) for name in AUDIT_HYBRIDS}})
        for backend in AUDIT_HYBRIDS:
            out=cfg.output/(backend+'-'+phase+'.json')
            command=[sys.executable,'-m','tools.benchmark_news_classifier','worker','--serving',str(cfg.hybrids),
                '--backend',backend,'--requests',str(cfg.requests/(phase+'.jsonl')),'--phase',phase,
                '--output',str(out),'--source-commit',cfg.source_commit,'--native-library',str(cfg.native_library)]
            with (cfg.output/(backend+'-'+phase+'.log')).open('w') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                report['failures'].append({'backend':backend,'phase':phase,'returncode':completed.returncode})
                write_json(cfg.output/'report.json',report)
                raise ValueError('additional hybrid worker failed')
            record=json.loads(out.read_text());source=primary[phase]['source-mmap']
            if (record['requests_sha256']!=source['requests_sha256'] or record['labels']!=source['labels']
                    or record['token_counts']!=source['token_counts'] or record['oov_counts']!=source['oov_counts']):
                raise ValueError('different supplemental input/registry')
            record['source_agreement_fraction']=sum(a==b for a,b in zip(record['predictions'],source['predictions']))/len(source['labels'])
            report[phase][backend]=record
            write_json(cfg.output/'report.json',report)
            print(json.dumps({'phase':phase,'backend':backend,'seed':primary['seed'],
                'accuracy':record['metrics']['accuracy'],'model_mib':record['complete_model_bytes']/(1<<20)}),flush=True)
    report['status']='passed'
    write_json(cfg.output/'report.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('primary','hybrids','requests','output','native-library'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    run(p.parse_args())


if __name__=='__main__':main()
