"""Summarize complete selected public-classifier quality/capacity/query pairs.

Only numeric logits/labels and reports are exported. Trained rows, registries,
keys and corpora stay outside the capsule. Incomplete results are rejected.
"""
import argparse
from collections import Counter
from pathlib import Path
import math
import numpy as np

from .run_application_gpu_recovery import registered_blocks, valid_block
from .tree_node_quality import array_digest, digest, environment, load, write


def checked_payload(path):
    value=load(path);payload=path.with_suffix('.npz')
    if not payload.exists() or digest(payload)!=value['payload_sha256']:raise ValueError('payload changed: '+str(path))
    with np.load(payload) as data:
        logits=data['logits'].copy();labels=data['labels'].copy();nll=data['nll'].copy()
    if array_digest(logits)!=value['logits_sha256']:raise ValueError('logit bytes changed')
    if not np.isfinite(logits).all() or not np.isfinite(nll).all():raise ValueError('nonfinite outputs')
    if len(labels)!=value['documents']:raise ValueError('document coverage changed')
    accuracy=float(np.mean(logits.argmax(1)==labels))
    if abs(accuracy-value['accuracy'])>1e-12 or abs(float(nll.mean())-value['mean_nll'])>1e-12:
        raise ValueError('reported quality differs from predictions')
    shifted=logits.astype(np.float64)-logits.max(axis=1,keepdims=True)
    expected=np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(logits)),labels]
    if not np.array_equal(expected,nll):raise ValueError('cross-entropy differs')
    return value,logits,labels


def audit(root,output,compact_output,datasets=('ag_news','dbpedia_14')):
    app=root/'application-frontier';selection=load(app/'source-selection.json')
    selection={**selection,'selected':[e for e in selection['selected'] if e['family'][0] in datasets]}
    source_records=[];models=[];timings=[];hashes={};checked={}
    def payload(path):
        key=str(path)
        if key not in checked:checked[key]=checked_payload(path);hashes[key]=digest(path)
        return checked[key]
    for entry in selection['selected']:
        name=Path(entry['training']).name;path=app/'source-quality'/(name+'-test.json')
        report,logits,labels=payload(path)
        job=app/'source-jobs'/(name+'.json')
        if report['status']!='complete' or report['job_sha256']!=digest(job):raise ValueError('source audit changed')
        source_records.append({'family':entry['family'],'seed':entry['seed'],'learning_rate':entry['learning_rate'],
            'rows':load(job)['rows'],'documents':len(labels),'accuracy':report['accuracy'],'mean_nll':report['mean_nll'],
            'complete_model_bytes':load(job)['complete_model_bytes'],'report_sha256':digest(path)})
    for entry in selection['selected']:
        if entry['family'][2]!='exact':continue
        model=app/'models'/Path(entry['training']).name;frozen=load(model/'selection.json');built=load(model/'built.json')
        if frozen['built_sha256']!=digest(model/'built.json') or frozen['test_used']:raise ValueError('frozen calibration changed')
        jobs={load(p)['name']:Path(p) for p in built['jobs']}
        reference,base,labels=payload(model/'quality/direct-fp32-test.json')
        quality=[];quality_by_name={}
        for name,path in sorted(jobs.items()):
            job=load(path);result,logits,actual_labels=payload(model/'quality'/(job['quality_reference']+'-test.json'))
            if not np.array_equal(labels,actual_labels):raise ValueError('official labels changed')
            leader=jobs[job['quality_reference']]
            if result['job_sha256']!=digest(leader):raise ValueError('quality-reference job changed')
            delta=(base.argmax(1)==labels).astype(np.int8)-(logits.argmax(1)==labels).astype(np.int8)
            record={'name':name,'complete_model_bytes':job['complete_model_bytes'],'accuracy':result['accuracy'],
                'mean_nll':result['mean_nll'],'accuracy_loss':float(delta.mean()),
                'prediction_disagreement_fraction':float(np.mean(base.argmax(1)!=logits.argmax(1))),
                'maximum_absolute_logit_error':float(np.max(np.abs(base.astype(np.float64)-logits.astype(np.float64)))),
                'byte_exact_source_rows':job['decoded_sha256']==load(jobs['direct-fp32'])['decoded_sha256']}
            quality.append(record);quality_by_name[name]=record
        choices=[]
        for budget,candidate in frozen['selection'].items():
            choices.append({'budget':float(budget),'selected':candidate,
                'heldout':quality_by_name[candidate['name']] if candidate else None})
        controls=[app/'source-jobs'/(Path(e['training']).name+'.json') for e in selection['selected']
            if e['family'][:2]==entry['family'][:2] and e['seed']==entry['seed']]
        blocks=registered_blocks(model,controls);full_gpu={};statuses=Counter()
        for path in [*(model/'gpu').glob('*.json'),*(model/'gpu-quality-retained').glob('*.json')]:
            value=load(path)
            if value.get('status')=='passed' and value.get('documents')==len(labels):
                full_gpu[(value['job_sha256'],value['mode'])]=path
        gpu_checks=[];seen=set()
        for block in blocks:
            if not valid_block(block):raise ValueError('incomplete paired block: '+block['name'])
            reports=[]
            for cell in block['cells']:
                path=Path(cell['output']);value=load(path);job=load(cell['job']);statuses[value['status']]+=1
                hashes[str(path)]=digest(path)
                if value['status']=='passed':
                    report,logits,actual_labels=payload(path)
                    expected_labels=labels[:len(actual_labels)]
                    if not np.array_equal(actual_labels,expected_labels):raise ValueError('GPU official test order changed')
                    if value['timed_documents']!=768 or len(value['samples_ms'])!=(768 if cell['batch_size']==1 else 24):
                        raise ValueError('fixed timing repeat schedule changed')
                    if not all(v>0 and math.isfinite(v) for v in value['samples_ms']):raise ValueError('invalid timings')
                    if abs(value['documents_per_second']-768/value['elapsed_wall_seconds'])>1e-9:raise ValueError('throughput count changed')
                    identity=(value['job_sha256'],value['mode'])
                    if identity not in seen:
                        if identity not in full_gpu:raise ValueError('full official GPU audit missing')
                        complete,full_logits,full_labels=payload(full_gpu[identity]);seen.add(identity)
                        baseline_path=app/'source-quality'/(Path(job['source']).name+'-test.json') if job['kind']=='source' else model/'quality'/(job['quality_reference']+'-test.json')
                        cpu,cpu_logits,cpu_labels=payload(baseline_path)
                        if not np.array_equal(full_labels,cpu_labels):raise ValueError('full GPU labels changed')
                        gpu_checks.append({'job':job['name'],'mode':value['mode'],'documents':len(full_labels),
                            'accuracy':complete['accuracy'],'cpu_accuracy':cpu['accuracy'],
                            'cpu_prediction_disagreement':int(np.sum(full_logits.argmax(1)!=cpu_logits.argmax(1))),
                            'maximum_absolute_logit_error_vs_cpu':float(np.max(np.abs(full_logits.astype(np.float64)-cpu_logits.astype(np.float64))))})
                reports.append({'job':job['name'],'kind':job['kind'],'mode':cell['mode'],'status':value['status'],
                    'p50_ms':value.get('p50_ms'),'p95_ms':value.get('p95_ms'),'p99_ms':value.get('p99_ms'),
                    'documents_per_second':value.get('documents_per_second'),'complete_model_bytes':value['complete_model_bytes'],
                    'resource_policy':value.get('resource_policy','exclusive'),'gpu_uuid':value['gpu_before']['gpu_uuid'],
                    'cpu_affinity':value['cpu_affinity'],'cuda_peak_reserved':value.get('cuda_peak_reserved'),
                    'host_after':value.get('host_after'),'report_sha256':digest(path)})
            timings.append({'model':model.name,'cohort':block['cohort'],'batch_size':block['batch_size'],
                'budget_gib':block['budget_gib'],'reports':reports})
        models.append({'model':model.name,'family':entry['family'],'seed':entry['seed'],'rows':load(jobs['direct-fp32'])['rows'],
            'official_test_documents':len(labels),'candidate_count':len(jobs),'quality':quality,'choices':choices,
            'gpu_cells':dict(statuses),'paired_blocks':len(blocks),'full_gpu_checks':gpu_checks,
            'built_sha256':digest(model/'built.json'),'selection_sha256':digest(model/'selection.json')})
    if len(models)!=6*len(datasets) or len(source_records)!=24*len(datasets):raise ValueError('complete registered source families required')
    result={'status':'audited-complete-public-application-grid','audit_environment':environment(),
        'datasets':list(datasets),'source_sha256':digest(__file__),'source_selection_sha256':digest(app/'source-selection.json'),
        'sources':source_records,'models':models,'timings':timings,'report_sha256':hashes,
        'scope':'Official AG News and DBpedia public classification. Allocator caps cover PyTorch CUDA allocations; host staging and host memory are outside them. Same-codec direct/native pairs and source/hash addressing controls remain distinct.'}
    write(output,result)
    compact={k:v for k,v in result.items() if k not in ('timings','report_sha256')}
    compact['models']=[{k:v for k,v in m.items() if k!='quality'} for m in models]
    compact.update(full_audit_sha256=digest(output),full_audit_artifact=str(output))
    write(compact_output,compact)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','compact-output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--datasets',nargs='+',choices=('ag_news','dbpedia_14'),default=['ag_news','dbpedia_14'])
    args=p.parse_args();audit(args.root,args.output,args.compact_output,args.datasets)

if __name__=='__main__':main()
