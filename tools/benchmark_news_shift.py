"""Temporal external-news validation with frozen AG News models and fair hybrids.

No retraining, test vocabulary construction, changed source registry or network
service. Inputs/weights remain local; numeric reports contain no article text.
"""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .export_news_hybrids import HYBRIDS
from .external_cotraining import file_hash
from .text_classifier import ngrams, tokenize
from .train_news_classifier import read_csv, write_json


HUFFPOST_URL = ('https://huggingface.co/datasets/heegyu/news-category-dataset/resolve/'
               '304a05a55bc6abc0446d8fae0d0771716b6a271a/data.json')
HUFFPOST_SHA256 = '2cfc51fa4a82a96ecbe251fa383737aa1dea2e0a517f61dece633fcc25a4a065'
AG_TRAIN_SHA256 = '76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e'
CATEGORIES = {'WORLD NEWS':0, 'SPORTS':1, 'BUSINESS':2, 'TECH':3}
CUTOFF = '2017-01-01'
BASE = {'source-resident','source-mmap','fp16','int8','faiss-sq4','faiss-sq8',
        'faiss-pq-8x8','faiss-cluster-8','ngram-bounded','ngram-int8','precomputed-words'}


def content_hash(words):
    return hashlib.sha256(' '.join(words).encode('utf-8')).hexdigest()


def prepare_requests(huffpost, ag_train, common, output, *, verify_hashes=True):
    huffpost, ag_train, common, output = map(Path,(huffpost,ag_train,common,output))
    if verify_hashes and (file_hash(huffpost)!=HUFFPOST_SHA256 or file_hash(ag_train)!=AG_TRAIN_SHA256):
        raise ValueError('pinned public dataset hash mismatch')
    lexicon = json.loads((common/'lexicon.json').read_text())
    words, grams = set(lexicon['tokens']), set(lexicon['grams'])
    ag_texts, _ = read_csv(ag_train)
    ag_hashes = {content_hash(tokenize(text,lexicon['max_tokens'])[0]) for text in ag_texts}
    candidates, counts = [], Counter()
    with huffpost.open(encoding='utf-8') as stream:
        for row_id,line in enumerate(stream):
            counts['original_rows'] += 1
            row = json.loads(line)
            if row['category'] not in CATEGORIES:
                continue
            counts['selected_category_rows'] += 1
            when = date.fromisoformat(row['date']).isoformat()
            text = row['headline']+' '+row['short_description']
            parsed,truncated = tokenize(text,lexicon['max_tokens'])
            if not parsed:
                counts['empty_rows'] += 1
                continue
            candidates.append((when,row_id,CATEGORIES[row['category']],text,parsed,truncated,content_hash(parsed)))
    output.mkdir(parents=True,exist_ok=False)
    seen, audit = {}, []
    streams = {phase:(output/(phase+'.jsonl')).open('w',encoding='utf-8') for phase in ('validation','test')}
    phase_stats = {phase:Counter() for phase in streams}
    new_types = {phase:set() for phase in streams}
    try:
        for when,row_id,label,text,parsed,truncated,digest in sorted(candidates,key=lambda v:v[:2]):
            if digest in seen:
                counts['duplicate_rows'] += 1
                counts['conflicting_category_duplicate_rows'] += int(seen[digest]!=label)
                continue
            seen[digest] = label
            if digest in ag_hashes:
                counts['ag_input_overlap_rows'] += 1
                continue
            phase = 'validation' if when<CUTOFF else 'test'
            unknown = [word for word in parsed if word not in words]
            missing = attempted = 0
            for word in unknown:
                parts = ngrams(word,lexicon['max_order'])
                attempted += len(parts)
                missing += sum(part not in grams for part in parts)
            stats = phase_stats[phase]
            stats['documents'] += 1
            stats['tokens'] += len(parsed)
            stats['oov_tokens'] += len(unknown)
            stats['oov_heavy_documents'] += int(len(unknown)/len(parsed)>=.1)
            stats['attempted_oov_ngrams'] += attempted
            stats['missing_oov_ngrams'] += missing
            stats['truncated_documents'] += int(truncated)
            stats['class_'+str(label)] += 1
            new_types[phase].update(unknown)
            streams[phase].write(json.dumps({'text':text,'label':label})+'\n')
            audit.append({'row_id':row_id,'phase':phase,'date':when,'label':label,'content_sha256':digest,
                'tokens':len(parsed),'oov':len(unknown),'attempted_oov_ngrams':attempted,
                'missing_oov_ngrams':missing,'truncated':truncated})
    finally:
        for stream in streams.values():
            stream.close()
    if any(phase_stats[p]['documents']<512 for p in streams):
        raise ValueError('both complete temporal phases need at least 512 supported documents')
    write_json(output/'preprocessing-audit.json',audit)
    manifest = {'schema_version':1,'category':'Independent public temporal news-routing evaluation',
        'source_url':HUFFPOST_URL,'source_sha256':file_hash(huffpost),'ag_train_sha256':file_hash(ag_train),
        'attribution':'Rishabh Misra, News Category Dataset; original HuffPost articles; mirror heegyu/news-category-dataset',
        'original_dataset_url':'https://www.kaggle.com/datasets/rmisra/news-category-dataset',
        'mirror_declared_license':'CC BY 4.0','taxonomy_mapping':CATEGORIES,'cutoff':CUTOFF,
        'registry_sha256':file_hash(common/'lexicon.json'),'counts':dict(counts),
        'phase_stats':{phase:{**dict(phase_stats[phase]),'unique_oov_types':len(new_types[phase]),
            'date_min':min(a['date'] for a in audit if a['phase']==phase),
            'date_max':max(a['date'] for a in audit if a['phase']==phase)} for phase in streams},
        'requests_sha256':{phase:file_hash(output/(phase+'.jsonl')) for phase in streams},
        'audit_sha256':file_hash(output/'preprocessing-audit.json'),
        'limitations':['Cross-source category mapping is related, not identical to AG News.',
            'No retraining or target-specific label calibration; this is an external model-transfer test.',
            'Non-ASCII tokenization and 128-token truncation preserve the original source-model boundary.']}
    write_json(output/'data.json',manifest)
    print(json.dumps({'counts':manifest['counts'],'phase_stats':manifest['phase_stats']}),flush=True)
    return manifest


def quality_pass(record, reference):
    metric, source = record['metrics'],reference['metrics']
    return (metric['accuracy']>=source['accuracy']-.005
            and metric['macro_f1']>=source['macro_f1']-.005
            and metric['cross_entropy']<=source['cross_entropy']*1.02)


def select(validation):
    reference = validation['source-mmap']
    result = []
    for kind,field,budgets in [('artifact','complete_model_bytes',(16,32,48,64,96,128,256)),
                             ('observed_rss','peak_rss_bytes',(64,128,192,256,384))]:
        for mib in budgets:
            eligible = [name for name,r in validation.items()
                        if r[field]<=mib*(1<<20) and quality_pass(r,reference)]
            selected = max(eligible,key=lambda k:validation[k]['timings']['32']['documents_per_second']) if eligible else None
            result.append({'budget_kind':kind,'budget_mib':mib,'eligible':eligible,'selected':selected})
    return result


def run(cfg):
    cfg.output.mkdir(parents=True,exist_ok=False)
    original = json.loads((cfg.serving/'export.json').read_text())
    hybrids = json.loads((cfg.hybrids/'export.json').read_text())
    if original['status']!='passed' or hybrids['status']!='passed' or original['source_sha256']!=hybrids['source_sha256']:
        raise ValueError('same passed source exports required')
    locations = {v['backend']:cfg.serving for v in original['variants']}
    locations.update({v['backend']:cfg.hybrids for v in hybrids['variants']})
    if set(locations)!=BASE|set(HYBRIDS):
        raise ValueError('all 15 frozen source/codec/precomputed/hybrid controls required')
    backends = list(locations)
    order = np.random.default_rng(42).permutation(len(backends))
    backends = [backends[int(i)] for i in order]
    data = json.loads((cfg.requests/'data.json').read_text())
    report = {'status':'running','seed':cfg.seed,'source_commit':cfg.source_commit,
        'trained_source_sha256':original['source_sha256'],'native_library_sha256':file_hash(cfg.native_library),
        'source_files_sha256':{name:file_hash(Path(__file__).with_name(name)) for name in
            ('benchmark_news_shift.py','benchmark_news_classifier.py','text_classifier.py','export_news_hybrids.py')},
        'input_manifest_sha256':file_hash(cfg.requests/'data.json'),'data':data,
        'phase_order':['validation-all','freeze-selection','test-all'],'backend_order':backends,
        'validation':{},'test':{},'selections':None,'failures':[]}
    for phase in ('validation','test'):
        if phase=='test':
            report['selections'] = select(report['validation'])
            write_json(cfg.output/'selection-before-test.json',{'seed':cfg.seed,'selections':report['selections'],
                'validation_report_sha256':{name:file_hash(cfg.output/(name+'-validation.json')) for name in backends}})
        for backend in backends:
            out = cfg.output/(backend+'-'+phase+'.json')
            command = [sys.executable,'-m','tools.benchmark_news_classifier','worker',
                '--serving',str(locations[backend]),'--backend',backend,'--requests',str(cfg.requests/(phase+'.jsonl')),
                '--phase',phase,'--output',str(out),'--source-commit',cfg.source_commit,
                '--native-library',str(cfg.native_library)]
            with (cfg.output/(backend+'-'+phase+'.log')).open('w') as log:
                completed = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                report['failures'].append({'backend':backend,'phase':phase,'returncode':completed.returncode})
                write_json(cfg.output/'report.json',report)
                raise ValueError('isolated external-news worker failed: '+backend)
            result = json.loads(out.read_text())
            if result['requests_sha256'] != data['requests_sha256'][phase]:
                raise ValueError('prepared temporal input changed')
            report[phase][backend] = result
            write_json(cfg.output/'report.json',report)
            print(json.dumps({'seed':cfg.seed,'phase':phase,'backend':backend,'accuracy':result['metrics']['accuracy'],
                'model_mib':result['complete_model_bytes']/(1<<20),'rss_mib':result['peak_rss_bytes']/(1<<20)}),flush=True)
        reference = report[phase]['source-mmap']
        for backend,result in report[phase].items():
            if result['labels']!=reference['labels'] or result['token_counts']!=reference['token_counts'] or result['oov_counts']!=reference['oov_counts']:
                raise ValueError('different labels/tokenization/registry between controls')
            agreement = np.asarray(result['predictions'])==reference['predictions']
            if backend=='ngram-bounded' and np.any(np.asarray(result['source_prediction_preserved']) & ~agreement):
                raise AssertionError('certified external prediction disagrees with evaluated source')
            result['source_agreement_fraction'] = float(agreement.mean())
    report['status'] = 'passed'
    write_json(cfg.output/'report.json',report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command',required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--huffpost',type=Path,required=True)
    prepare.add_argument('--ag-train',type=Path,required=True)
    prepare.add_argument('--training',type=Path,required=True)
    prepare.add_argument('--output',type=Path,required=True)
    run_parser = commands.add_parser('run')
    for name in ('serving','hybrids','requests','output','native-library'):
        run_parser.add_argument('--'+name,type=Path,required=True)
    run_parser.add_argument('--source-commit',required=True)
    run_parser.add_argument('--seed',type=int,required=True)
    cfg = p.parse_args()
    if cfg.command=='prepare':
        prepare_requests(cfg.huffpost,cfg.ag_train,cfg.training/'common',cfg.output)
    else:
        run(cfg)


if __name__ == '__main__':
    main()
