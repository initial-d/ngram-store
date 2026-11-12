"""Recompute official macro scores and paired seed/case uncertainty from scalars."""
import argparse
import json
from pathlib import Path

import numpy as np

from .counterfact_capacity import write,sha
from .counterfact_data import summarize

GROUPS=('rewrite','paraphrase','neighborhood')


def arrays(cases):
    cases=sorted((c for c in cases if c['split']=='test'),key=lambda c:c['case_index'])
    return [c['case_id'] for c in cases],np.array([[np.mean([p['success'] for p in c['groups'][g]]) for g in GROUPS] for c in cases])


def harmonic(values):
    values=np.asarray(values)
    return np.divide(3.,np.sum(np.divide(1.,values,out=np.full_like(values,np.inf),where=values>0),axis=-1))


def paired_interval(a,b,replicates=2000):
    if a.shape!=b.shape or a.ndim!=3:raise ValueError('paired seed/case arrays required')
    rng=np.random.default_rng(20261106);deltas=[];group_deltas=[]
    for _ in range(replicates):
        seeds=rng.integers(0,len(a),len(a));cases=rng.integers(0,a.shape[1],a.shape[1])
        av=a[seeds][:,cases].mean((0,1));bv=b[seeds][:,cases].mean((0,1))
        deltas.append(float(harmonic(av)-harmonic(bv)));group_deltas.append(av-bv)
    av=a.mean((0,1));bv=b.mean((0,1))
    return {'paired_seeds':len(a),'paired_cases':a.shape[1],
            'harmonic_delta':float(harmonic(av)-harmonic(bv)),
            'harmonic_delta_interval95':np.quantile(deltas,[.025,.975]).tolist(),
            'group_deltas':{g:{'delta':float((av-bv)[i]),'interval95':np.quantile(np.array(group_deltas)[:,i],[.025,.975]).tolist()} for i,g in enumerate(GROUPS)},
            'bootstrap':'Paired resampling of training seeds and official cases; 2000 replicates, fixed RNG',
            'limitations':'Three seeds cannot establish broad training variability; cases can share subjects and neighborhood prompts, so case resampling is not an independence guarantee'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();bykey={};items=[]
    for file in sorted(args.runs.glob('*/report.json')):
        r=json.loads(file.read_text());item={'run':file.parent.name,'report_sha256':sha(file),'status':r['status']}
        if r['status']!='complete':items.append(item);continue
        k=r.get('selected_top_k');case_file=file.parent/(f'case-scores-k{k}.json' if k else 'case-scores.json')
        cases=json.loads(case_file.read_text());metrics={s:summarize([c for c in cases if s=='all' or c['split']==s]) for s in ('all','calibration','test')}
        if metrics!=r['metrics']:raise AssertionError('reported metrics do not match per-case scalars: '+str(file))
        near=sum(abs(p['target_true']-p['target_new'])<=.0014 for c in cases for rows in c['groups'].values() for p in rows)
        safe=[p for c in cases if c['split']=='test' for p in c['groups']['neighborhood'] if not p['prompt_matches_an_edited_rewrite']]
        item.update(metrics=metrics,case_scores_sha256=sha(case_file),near_ties_at_0_0014_nll=near,
            unedited_neighbor_diagnostic={'prompts':len(safe),'success':float(np.mean([p['success'] for p in safe])) if safe else None},
            trainable_parameters=r.get('trainable_parameters'),memory_parameters=r.get('memory_parameters'),
            physical_rows=r.get('physical_rows'),training_seconds=r.get('training_seconds'),delta=r.get('delta'))
        items.append(item)
        if 'seed' in r:
            model='qwen35' if 'Qwen3.5' in r['base_model'] else 'qwen06'
            edits=r['protocol'].get('edit_counts');prepare=json.loads((args.runs.parent/f'prepared-{model}-{int(file.parent.name.split("-")[1])}'/'prepare.json').read_text())
            bykey[(model,prepare['edits'],r['variant'],r['seed'])]=arrays(cases)
    comparisons=[]
    for model in ('qwen06','qwen35'):
        for edits in (1024,20877):
            for control in ('small','mid','head-constant','off','adapter'):
                seeds=[s for s in (3,7,11) if (model,edits,'large',s) in bykey and (model,edits,control,s) in bykey]
                if not seeds:continue
                ids=bykey[(model,edits,'large',seeds[0])][0];a=[];b=[]
                for s in seeds:
                    ai,av=bykey[(model,edits,'large',s)];bi,bv=bykey[(model,edits,control,s)]
                    if ai!=ids or bi!=ids:raise ValueError('case pairing changed')
                    a.append(av);b.append(bv)
                comparisons.append({'model':model,'edits':edits,'control':control,'seeds':seeds,
                                    'complete_three_seed_pair':len(seeds)==3,**paired_interval(np.stack(a),np.stack(b))})
    trained_complete=len(bykey)==72
    write(args.output,{'status':'training-grid-complete' if trained_complete else 'partial',
        'completed_trained_runs':len(bykey),'expected_trained_runs':72,'runs':items,'large_vs_control':comparisons,
        'interpretation':'Partial pairs are provisional. Retain all official metrics and failures; no backend or algorithm dominance follows from these comparisons alone.'})


if __name__=='__main__':main()
