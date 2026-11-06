"""MEMIT MultiCounterFact cases, canonical target scoring and batching.

Keep every official rewrite/paraphrase/neighborhood prompt. Never train on
paraphrases or neighborhoods. The dataset contains counterfactual updates,
not independently verified contemporary facts.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

MEMIT_SHA='80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b'
DATA_SHA='977a6acce4705507b5fd6bfcea8f61cd78f9ed0f9cd9c9a6bcd6c8a3ed61c833'


def load_records(path,expected_sha=DATA_SHA):
    data=Path(path).read_bytes()
    if expected_sha and hashlib.sha256(data).hexdigest()!=expected_sha:raise ValueError('benchmark bytes changed')
    records=json.loads(data);keys=set();ids=set()
    for r in records:
        e=r['requested_rewrite'];key=(e['subject'],e['relation_id'])
        if key in keys or r['case_id'] in ids:raise ValueError('duplicate/conflicting batch edits; use official MultiCounterFact')
        keys.add(key);ids.add(r['case_id'])
        if not r['paraphrase_prompts'] or not r['neighborhood_prompts'] or e['target_new']['str']==e['target_true']['str']:
            raise ValueError('invalid official case')
    return records


def encode_choice(tokenizer,prefix,target,*,eos=False):
    prefix_ids=tokenizer(prefix,add_special_tokens=False).input_ids
    suffix_ids=tokenizer(' '+target,add_special_tokens=False).input_ids
    ids=tokenizer(prefix+' '+target,add_special_tokens=False).input_ids
    # The original oracle scores standalone suffix IDs at prefix length. Reject
    # a tokenizer boundary mismatch rather than silently score a different task.
    if ids!=prefix_ids+suffix_ids or not prefix_ids or not suffix_ids:
        raise ValueError('official prefix/suffix token boundary is not stable for this tokenizer')
    if eos:suffix_ids=suffix_ids+[tokenizer.eos_token_id];ids=prefix_ids+suffix_ids
    return {'ids':ids,'target_start':len(prefix_ids),'target_ids':suffix_ids}


def build_sequences(records,tokenizer,*,prefix_transform=None):
    sequences=[];lookup={};cases=[];train=[]
    def add(prefix,target,eos=False):
        if prefix_transform is not None:prefix=prefix_transform(prefix)
        item=encode_choice(tokenizer,prefix,target,eos=eos)
        key=(tuple(item['ids']),item['target_start'])
        if key not in lookup:lookup[key]=len(sequences);sequences.append(item)
        return lookup[key]
    all_rewrites={r['requested_rewrite']['prompt'].format(r['requested_rewrite']['subject']) for r in records}
    for index,r in enumerate(records):
        e=r['requested_rewrite'];prefix=e['prompt'].format(e['subject'])
        train.append(add(prefix,e['target_new']['str'],True))
        groups={}
        for name,prompts in [('rewrite',[prefix]),('paraphrase',r['paraphrase_prompts']),('neighborhood',r['neighborhood_prompts'])]:
            groups[name]=[{'new':add(q,e['target_new']['str']),'true':add(q,e['target_true']['str']),
                           'prompt_matches_an_edited_rewrite':q in all_rewrites} for q in prompts]
        cases.append({'case_id':r['case_id'],'case_index':index,'split':'calibration' if index%10==0 else 'test',
                      'groups':groups})
    return sequences,train,cases


def batch_indices(indices,sequences,batch_size,*,rng=None):
    groups=defaultdict(list)
    for i in indices:groups[len(sequences[i]['ids'])].append(i)
    result=[]
    for length in sorted(groups):
        group=groups[length]
        if rng is not None:group=[group[i] for i in rng.permutation(len(group))]
        result.extend(group[i:i+batch_size] for i in range(0,len(group),batch_size))
    if rng is not None:result=[result[i] for i in rng.permutation(len(result))]
    return result


def case_metrics(cases,scores):
    output=[]
    for c in cases:
        r={'case_id':c['case_id'],'case_index':c['case_index'],'split':c['split'],'groups':{}}
        for name,pairs in c['groups'].items():
            rows=[]
            for p in pairs:
                a,b=scores[str(p['new'])],scores[str(p['true'])]
                rows.append({'target_new':a['nll'],'target_true':b['nll'],
                             'correct':b['correct'] if name=='neighborhood' else a['correct'],
                             'success':b['nll']<a['nll'] if name=='neighborhood' else a['nll']<b['nll'],
                             'prompt_matches_an_edited_rewrite':p['prompt_matches_an_edited_rewrite']})
            r['groups'][name]=rows
        output.append(r)
    return output


def summarize(cases):
    import math
    result={}
    for group in ('rewrite','paraphrase','neighborhood'):
        percase=[sum(w['success'] for w in r['groups'][group])/len(r['groups'][group]) for r in cases]
        acc=[sum(w['correct'] for w in r['groups'][group])/len(r['groups'][group]) for r in cases]
        result[group]={'cases':len(cases),'prompts':sum(len(r['groups'][group]) for r in cases),
                       'success':math.fsum(percase)/len(cases),'teacher_forced_accuracy':math.fsum(acc)/len(cases)}
    values=[result[k]['success'] for k in ('rewrite','paraphrase','neighborhood')]
    result['harmonic_score']=len(values)/sum(1/v for v in values) if all(values) else 0.
    return result
