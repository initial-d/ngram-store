"""Verify public cross-source news evidence without NumPy, Torch or article text."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_news_shift import read, summarize
from tools.analyze_news_hybrid_audit import summarize as summarize_audit


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    files={str(p.relative_to(HERE)):p for p in HERE.rglob('*') if p.is_file() and p!=HERE/'manifest.json'}
    if set(files)!=set(manifest['files_sha256']):
        raise ValueError('evidence file set changed')
    for name,path in files.items():
        if path.name=='lexicon.json' or path.suffix in ('.npy','.npz','.f32','.f64','.men','.rows','.faiss','.jsonl'):
            raise ValueError('article input, lexicon or weights must not be redistributed')
        if digest(path.read_bytes())!=manifest['files_sha256'][name]:
            raise ValueError('changed evidence file: '+name)
    data=read(HERE/'requests/data.json')
    if data['source_sha256']!='2cfc51fa4a82a96ecbe251fa383737aa1dea2e0a517f61dece633fcc25a4a065':
        raise ValueError('independent pinned dataset changed')
    if data['ag_train_sha256']!='76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e':
        raise ValueError('AG News source inputs changed')
    if data['taxonomy_mapping']!={'WORLD NEWS':0,'SPORTS':1,'BUSINESS':2,'TECH':3}:
        raise ValueError('literal source category mapping changed')
    audit_path=HERE/'requests/preprocessing-audit.json.gz'
    audit_bytes=gzip.decompress(audit_path.read_bytes())
    if digest(audit_bytes)!=data['audit_sha256']:
        raise ValueError('input-only audit changed')
    audit=json.loads(audit_bytes)
    if len({r['row_id'] for r in audit})!=len(audit) or len({r['content_sha256'] for r in audit})!=len(audit):
        raise ValueError('deduplicated article IDs/hashes required')
    if any(set(r)!={'row_id','phase','date','label','content_sha256','tokens','oov','attempted_oov_ngrams',
                    'missing_oov_ngrams','truncated'} for r in audit):
        raise ValueError('only numeric/date/hash input audit fields allowed')
    for label in range(4):
        selected=sorted((r for r in audit if r['label']==label),key=lambda r:(r['content_sha256'],r['row_id']))
        cut=len(selected)//5
        if any(r['phase']!=('validation' if i<cut else 'test') for i,r in enumerate(selected)):
            raise ValueError('stratified content-hash calibration split changed')
    for phase in ('validation','test'):
        selected=[r for r in audit if r['phase']==phase];stats=data['phase_stats'][phase]
        if [(r['date'],r['row_id']) for r in selected]!=sorted((r['date'],r['row_id']) for r in selected):
            raise ValueError('input order changed')
        counts={'documents':len(selected),'tokens':sum(r['tokens'] for r in selected),
            'oov_tokens':sum(r['oov'] for r in selected),
            'oov_heavy_documents':sum(r['oov']/r['tokens']>=.1 for r in selected),
            'attempted_oov_ngrams':sum(r['attempted_oov_ngrams'] for r in selected),
            'missing_oov_ngrams':sum(r['missing_oov_ngrams'] for r in selected),
            'truncated_documents':sum(r['truncated'] for r in selected)}
        counts.update({'class_'+str(label):sum(r['label']==label for r in selected) for label in range(4)})
        if any(stats.get(k,0)!=v for k,v in counts.items()):
            raise ValueError('published input statistics differ from article audit')
    runs=[]
    for seed in (3,7,11):
        directory=HERE/'results'/f'seed-{seed}'
        run=read(directory/'report.json.gz');runs.append(run)
        if run['data']!=data or run['input_manifest_sha256']!=digest((HERE/'requests/data.json').read_bytes()):
            raise ValueError('input manifest differs across seeds')
        freeze=read(directory/'selection-before-test.json')
        if freeze['seed']!=seed or freeze['selections']!=run['selections']:
            raise ValueError('pre-evaluation selection file differs from final choices')
        for backend,expected in freeze['validation_report_sha256'].items():
            raw=gzip.decompress((directory/'calibration'/f'{backend}.json.gz').read_bytes())
            if digest(raw)!=expected:
                raise ValueError('calibration record changed after freeze')
            record=json.loads(raw);combined=dict(run['validation'][backend]);combined.pop('source_agreement_fraction')
            if record!=combined:
                raise ValueError('calibration source of frozen choices differs from complete report')
        training=read(HERE/'training'/f'seed-{seed}.json')
        if (training['status']!='passed' or training['seed']!=seed or training['device']!='cpu'
                or training['source_sha256']!=run['trained_source_sha256']
                or training['source_sha256']==training['initial_rows_sha256']
                or training['selected_epoch']!=min(training['epochs'],key=lambda e:e['validation_ce'])['epoch']):
            raise ValueError('unchanged validation-selected CPU source training required')
        if len(training['epochs'])!=8 or training['data']['train_documents']!=108000 or training['data']['validation_documents']!=12000:
            raise ValueError('full frozen AG News source-training protocol required')
        for phase in ('validation','test'):
            selected=[r for r in audit if r['phase']==phase]
            source=run[phase]['source-mmap']
            if (source['labels']!=[r['label'] for r in selected] or source['token_counts']!=[r['tokens'] for r in selected]
                    or source['oov_counts']!=[r['oov'] for r in selected]):
                raise ValueError('runtime request order/counts differ from input-only audit')
        for backend in run['test']:
            bundle=read(HERE/'bundles'/f'seed-{seed}'/backend/'classifier.json')
            sizes=read(HERE/'bundles'/f'seed-{seed}'/backend/'file-sizes.json')
            if (set(sizes)!=set(bundle['files_sha256'])|{'classifier.json'}
                    or any(type(v) is not int or v<=0 for v in sizes.values())
                    or sum(sizes.values())!=run['test'][backend]['complete_model_bytes']):
                raise ValueError('complete deployment bytes differ from per-file size audit')
            if digest((HERE/'bundles'/f'seed-{seed}'/backend/'classifier.json').read_bytes())!=run['test'][backend]['classifier_sha256']:
                raise ValueError('classifier metadata differs from isolated inference')
            if bundle['trained_source_sha256']!=training['source_sha256'] or bundle['head_sha256']!=training['head_sha256']:
                raise ValueError('a backend uses another trained source/head')
            if bundle['activation']!='relu' or bundle['dim']!=64:
                raise ValueError('source graph/dimension changed')
            hybrid=backend.startswith('precomputed-hybrid-')
            if hybrid:
                files=bundle['files_sha256']
                if 'word-ids.u64' in files or 'word-offsets.u64' in files or bundle['rows']!=570867:
                    raise ValueError('hybrid must omit whole-word rows and unused bag arrays')
                known=read(HERE/'bundles'/f'seed-{seed}'/'precomputed-words/classifier.json')
                if files['word-vectors.f32']!=known['files_sha256']['word-vectors.f32']:
                    raise ValueError('hybrids must share identical known-word FP32 features')
                if sizes['word-vectors.f32']!=62035*64*4:
                    raise ValueError('complete precomputed training-word rows required')
    summary=summarize(runs,read(HERE/'bootstrap.json.gz'))
    if summary!=read(HERE/'summary.json'):
        raise ValueError('published summary differs from complete numeric evidence')
    extra_root=HERE/'supplemental-faiss-hybrids';audits=[]
    for seed in (3,7,11):
        directory=extra_root/'results'/f'seed-{seed}'
        audit=read(directory/'report.json.gz');audits.append(audit)
        primary_bytes=gzip.decompress((HERE/'results'/f'seed-{seed}/report.json.gz').read_bytes())
        if audit['primary_report_sha256']!=digest(primary_bytes):
            raise ValueError('supplemental audit must retain the completed primary report')
        freeze=read(directory/'selection-before-test.json')
        if freeze['selections']!=audit['selections'] or freeze['primary_report_sha256']!=audit['primary_report_sha256']:
            raise ValueError('supplemental calibration freeze changed')
        for name,expected in freeze['validation_report_sha256'].items():
            raw=gzip.decompress((directory/'calibration'/f'{name}.json.gz').read_bytes())
            record=dict(audit['validation'][name]);record.pop('source_agreement_fraction')
            if digest(raw)!=expected or json.loads(raw)!=record:
                raise ValueError('supplemental calibration changed after freeze')
        for name in audit['test']:
            folder=extra_root/'bundles'/f'seed-{seed}'/name
            bundle=read(folder/'classifier.json');sizes=read(folder/'file-sizes.json')
            original=read(HERE/'bundles'/f'seed-{seed}'/name[len('precomputed-hybrid-'):]/'classifier.json')
            known=read(HERE/'bundles'/f'seed-{seed}'/'precomputed-words/classifier.json')
            if (digest((folder/'classifier.json').read_bytes())!=audit['test'][name]['classifier_sha256']
                    or bundle['trained_source_sha256']!=audit['trained_source_sha256']
                    or bundle['head_sha256']!=known['head_sha256'] or bundle['rows']!=570867
                    or bundle['files_sha256']['word-vectors.f32']!=known['files_sha256']['word-vectors.f32']
                    or bundle['files_sha256']['memory/codec.faiss']!=original['files_sha256']['memory/codec.faiss']):
                raise ValueError('FAISS hybrids must reuse source words/head/codec and rebased grams')
            if (set(sizes)!=set(bundle['files_sha256'])|{'classifier.json'}
                    or sum(sizes.values())!=audit['test'][name]['complete_model_bytes']
                    or 'word-ids.u64' in sizes or 'word-offsets.u64' in sizes):
                raise ValueError('complete supplemental deployment size audit changed')
    if summarize_audit(runs,audits)!=read(extra_root/'summary.json'):
        raise ValueError('supplemental summary differs from numeric evidence')
    print('Verified three source models, 15 primary + 3 supplemental controls, frozen choices, task metrics, and paired document-bootstrap draws.')


if __name__=='__main__':main()
