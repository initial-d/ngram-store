"""Verify full numeric task evidence, source/bundle hashes and frozen selections."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0,str(REPO))
from tools.analyze_news_classifier import read, summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for path in ROOT.rglob('*'):
        if path.is_file() and (path.name=='lexicon.json' or path.suffix in ('.npy','.npz','.f32','.f64','.men','.faiss','.rows')):
            raise ValueError('data/weights must remain outside public numeric evidence: '+str(path))
    manifest = read(ROOT/'manifest.json')
    for name, expected in manifest['files_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('evidence hash mismatch: '+name)
    environment = read(ROOT/'environment.json')
    # The measured commit remains the reference when later application code
    # changes. Verification binds stored evidence, rather than relabeling old
    # measurements as timings of the current checkout.
    audit = read(ROOT/'preprocessing-audit.json')
    tasks = {}
    for task in ('linear','relu'):
        tasks[task] = []
        data = read(ROOT/f'training/{task}/data.json')
        if (data['train_documents'],data['validation_documents'],data['official_train_documents'],data['rows'],data['dim'])!=(108000,12000,120000,612759,64):
            raise ValueError('complete frozen dataset and trained table required')
        if data['train_csv_sha256']!='76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e' or data['test_csv_sha256']!='521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699':
            raise ValueError('pinned data hashes differ')
        if digest(ROOT/'AG-NEWS-NOTICE.txt')!=data['research_use_notice_sha256']:
            raise ValueError('research-use notice changed')
        if audit['registry_sha256']!=data['registry_sha256']['lexicon.json']:
            raise ValueError('preprocessing audit uses a different registry')
        for seed in (3,7,11):
            run = read(ROOT/f'results/{task}/seed-{seed}/report.json.gz')
            original_report = (json.dumps(run,indent=2,allow_nan=False)+'\n').encode()
            if hashlib.sha256(original_report).hexdigest()!=manifest['original_report_sha256'][f'{task}/seed-{seed}']:
                raise ValueError('compact evidence differs from original complete report')
            tasks[task].append(run)
            training = read(ROOT/f'training/{task}/seed-{seed}.json')
            export = read(ROOT/f'exports/{task}/seed-{seed}/export.json')
            if (training['status']!='passed' or training['data']!=data or training['seed']!=seed
                    or training['settings'].get('activation','identity')!=('identity' if task=='linear' else 'relu')
                    or len(training['epochs'])!=8 or training['device']!='cpu'
                    or training['source_sha256']==training['initial_rows_sha256']):
                raise ValueError('complete real CPU training provenance required')
            if any(not math.isfinite(e[k]) for e in training['epochs'] for k in ('train_loss','validation_ce','validation_accuracy','seconds')):
                raise ValueError('nonfinite training evidence')
            if training['selected_epoch']!=min(training['epochs'],key=lambda e:e['validation_ce'])['epoch']:
                raise ValueError('epoch selection differs from validation loss')
            if (run['source_commit']!=manifest['measured_source_commit']
                    or run['native_library_sha256']!=environment['native_library_sha256']
                    or export['status']!='passed' or export['source_sha256']!=training['source_sha256']):
                raise ValueError('frozen evaluation/export source differs')
            for name, expected in run['source_files_sha256'].items():
                if expected!=environment['measured_source_sha256']['tools/'+name]:
                    raise ValueError('worker source differs from measured source')
            frozen = read(ROOT/f'results/{task}/seed-{seed}/selection-before-test.json')
            if frozen['seed']!=seed or frozen['selections']!=run['selections']:
                raise ValueError('test changed frozen validation choices')
            for backend,record in run['validation'].items():
                original = {k:v for k,v in record.items() if k!='source_agreement_fraction'}
                raw = (json.dumps(original,indent=2,allow_nan=False)+'\n').encode()
                if hashlib.sha256(raw).hexdigest()!=frozen['validation_report_sha256'][backend]:
                    raise ValueError('validation observations changed after selection')
                metadata = read(ROOT/f'exports/{task}/seed-{seed}/{backend}.json')
                inventory = read(ROOT/f'exports/{task}/seed-{seed}/{backend}-inventory.json')
                variant = next(v for v in export['variants'] if v['backend']==backend)
                model_bytes = sum(v['bytes'] for v in inventory.values())
                if (digest(ROOT/f'exports/{task}/seed-{seed}/{backend}.json')!=record['classifier_sha256']
                        or record['classifier_sha256']!=variant['classifier_sha256']
                        or inventory['classifier.json']['sha256']!=record['classifier_sha256']
                        or metadata['files_sha256']!=variant['files_sha256']
                        or set(inventory)!=set(metadata['files_sha256'])|{'classifier.json'}
                        or any(inventory[k]['sha256']!=v for k,v in metadata['files_sha256'].items())
                        or any(model_bytes!=v for v in (record['complete_model_bytes'],run['test'][backend]['complete_model_bytes'],variant['complete_model_bytes']))):
                    raise ValueError('complete bundle accounting/hash binding differs')
                if (metadata['trained_source_sha256']!=training['source_sha256']
                        or metadata['head_sha256']!=training['head_sha256']
                        or metadata['source_commit']!=training['source_commit']
                        or metadata['seed']!=seed or metadata['selected_epoch']!=training['selected_epoch']):
                    raise ValueError('backend uses a different trained model')
                if backend!='linear-fused' and metadata['files_sha256']['head.npz']!=training['head_sha256']:
                    raise ValueError('source head changed')
                if backend.startswith('source-') and metadata['files_sha256']['source.f32']!=training['source_sha256']:
                    raise ValueError('source table control changed')
                if backend!='precomputed-words' and any(metadata['files_sha256'][k]!=v for k,v in data['registry_sha256'].items()):
                    raise ValueError('registered keys or word bags changed')
                for path in (ROOT/f'exports/{task}/seed-{seed}/{backend}-metadata').rglob('*.json'):
                    name = str(path.relative_to(ROOT/f'exports/{task}/seed-{seed}/{backend}-metadata'))
                    if digest(path)!=inventory[name]['sha256'] or path.stat().st_size!=inventory[name]['bytes']:
                        raise ValueError('codec/partition/bounds metadata binding differs')
            for phase in ('validation','test'):
                source = run[phase]['source-mmap']
                if (source['token_counts']!=audit[phase]['token_counts'] or source['oov_counts']!=audit[phase]['oov_counts']
                        or audit[phase]['empty_documents'] or sum(audit[phase]['truncated'])!=audit[phase]['truncated_documents']):
                    raise ValueError('input audit differs from measured requests')
    summary = summarize(tasks)
    if any(run['test']['int8']['predictions']!=run['test']['ngram-int8']['predictions'] for run in tasks['relu']):
        raise ValueError('reported nonlinear INT8 prediction identity differs')
    installed = read(ROOT/'actual-installed-inference.json')
    if installed['status']!='passed' or installed['torch_loaded'] or installed['source_commit']!=manifest['measured_source_commit']:
        raise ValueError('actual installed-bundle functional check failed')
    for backend, result in installed['results'].items():
        if result['classifier_sha256']!=tasks['relu'][0]['test'][backend]['classifier_sha256']:
            raise ValueError('installed fixture uses a different actual news bundle')
    if summary!=read(ROOT/'summary.json'):
        raise ValueError('summary differs from complete numeric predictions/timings')
    print(json.dumps({'status':'passed','complete_runs':6,'official_test_documents_per_run':7600,
        'validation_only_selection':True,'bundle_source_and_input_audits':True,
        'prediction_preservation_checks':True,'torch_free_cpu_inference':True}))


if __name__=='__main__': main()
