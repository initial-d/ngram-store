"""Observed prefix/suffix relations in actual trained hash tables.

Exclude rows with multiple observed n-gram keys within training windows. This
does not certify global collision freedom, equal incidence or co-training.
Compare to actual unrelated rows within the same head, never invented vectors.
No vocabulary, token IDs, row IDs or weights are written to the report.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
from .engram_row_adapter import file_sha256
from .tinyengram_checkpoint import load_upstream, source_revision
from .train_tinyengram_identity import sample_windows, tensor_sha, write_json


def relation_pairs(keys, side, rng, limit=10000):
    groups=defaultdict(list)
    for row,key in keys.items():
        groups[key[:-1] if side=='prefix' else key[1:]].append(row)
    groups=[sorted(rows) for rows in groups.values() if len(rows)>1]
    # One pair per group, deterministic group-balanced sampling.
    chosen=rng.permutation(len(groups))[:limit]
    pairs=[rng.choice(groups[i],2,replace=False).tolist() for i in chosen]
    return np.asarray(pairs,dtype=np.int64).reshape(-1,2),len(groups)


def pair_stats(rows,pairs):
    if not len(pairs):return {'pairs':0}
    a,b=rows[pairs[:,0]].astype(np.float64),rows[pairs[:,1]].astype(np.float64)
    na,nb=np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1)
    valid=(na>0)&(nb>0)
    a,b,na,nb=a[valid],b[valid],na[valid],nb[valid]
    cosine=np.einsum('ij,ij->i',a,b)/(na*nb)
    ratio=np.minimum(na,nb)/np.maximum(na,nb)
    return {'pairs':len(pairs),'nonzero_pairs':len(cosine),
            'cosine_quantiles':np.quantile(cosine,[0,.01,.05,.5,.95,.99,1]).tolist() if len(cosine) else [],
            'norm_ratio_quantiles':np.quantile(ratio,[0,.01,.05,.5,.95,.99,1]).tolist() if len(ratio) else [],
            'joint_fraction':{str(t):float(np.mean((cosine>=t)&(ratio>=t))) for t in (.90,.95,.97,.98,.99)} if len(cosine) else {}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-run','seed-directory','corpus','upstream-dir','report-out'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.report_out.exists():p.error('use a fresh report')
    start=time.perf_counter()
    train=json.loads(args.training_run.read_text()); protocol=train['protocol']
    seed=int(args.seed_directory.name.split('-')[-1]); memory=train['seeds'][str(seed)]['memory']
    if file_sha256(args.corpus)!=protocol['corpus_sha256']:raise ValueError('corpus changed')
    checkpoint=args.seed_directory/'checkpoint'
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
    text=args.corpus.read_text();text=text[:int(len(text)*protocol['training_fraction'])]
    ids=tokenizer(text,return_tensors='pt',add_special_tokens=False).input_ids
    if tensor_sha(ids)!=train['splits']['train']['token_ids_sha256']:raise ValueError('training tokens changed')
    windows,_=sample_windows(ids,protocol['window_tokens'],protocol['training_windows'])
    engram,_=load_upstream(args.upstream_dir)
    config=json.loads((checkpoint/'engram_config.json').read_text());config['tokenizer_name_or_path']=str(checkpoint)
    mapping=engram.NgramHashMapping(**{k:config[k] for k in ('engram_vocab_size','max_ngram_size','n_embed_per_ngram','n_head_per_ngram','layer_ids','tokenizer_name_or_path','pad_id','seed')})
    layer=int(train['seeds'][str(seed)]['layer_id'])
    weight_key=f'model.engram_layers.{layer}.multi_head_embedding.embedding.weight'
    with safe_open(checkpoint/'model.safetensors',framework='pt',device='cpu') as stream:weight=stream.get_tensor(weight_key)
    if tensor_sha(weight)!=memory['trained_weight_sha256']:raise ValueError('actual trained rows changed')
    rows=weight.numpy()
    report={'status':'passed','code_sha':source_revision(),'command':shlex.join(sys.orig_argv),
            'training_report_sha256':file_sha256(args.training_run),'memory':memory,'seed':seed,'heads':[],
            'sampling':'one pair per shared-prefix/suffix group, up to 10000 groups per head; unrelated actual-row pairs matched in count and head',
            'quantile_probabilities':[0,.01,.05,.5,.95,.99,1],
            'limitations':['finite observed training-key uniqueness is not globally collision-free addressing',
                           'hash-head retrieval and similar strings do not certify equal training incidence',
                           'TinyEngram position-wise gated concatenation differs from additive SGNS bags',
                           'trained rows retain their initialization; no claim all learned rows are >=99% similar',
                           'similarity thresholds are diagnostic; no quality or compression selection from this analysis'],
            'environment':train['environment']}
    hashed=mapping.hash(windows.numpy())[layer]
    compressed=mapping.compressed_tokenizer(windows.numpy())
    for head,offset in enumerate(memory['head_offsets']):
        order=head//protocol['n_head_per_ngram']+2
        unique={}; ambiguous=set();counts=defaultdict(int)
        for sample in range(len(windows)):
            for pos in range(order-1,protocol['window_tokens']):
                row=int(hashed[sample,pos,head])+offset
                key=tuple(int(x) for x in compressed[sample,pos-order+1:pos+1])
                counts[row]+=1
                if row in unique and unique[row]!=key:ambiguous.add(row)
                else:unique[row]=key
        keys={row:key for row,key in unique.items() if row not in ambiguous and counts[row]>=2}
        item={'head':head,'order':order,'observed_rows':len(unique),'observed_ambiguous_rows':len(ambiguous),'eligible_rows':len(keys)}
        rng=np.random.default_rng(seed*100+head)
        for side in ('prefix','suffix'):
            pairs,groups=relation_pairs(keys,side,rng)
            item[side]={'eligible_groups':groups,**pair_stats(rows,pairs)}
            eligible=np.array(sorted(keys),dtype=np.int64)
            unrelated=[]
            if len(eligible)>1:
                # Reject actual pairs sharing either relation; no random vectors.
                for _ in range(len(pairs)*20):
                    a,b=rng.choice(eligible,2,replace=False)
                    if keys[a][:-1]!=keys[b][:-1] and keys[a][1:]!=keys[b][1:]:unrelated.append([a,b])
                    if len(unrelated)>=len(pairs):break
            item[side+'-unrelated']=pair_stats(rows,np.array(unrelated,dtype=np.int64).reshape(-1,2))
        report['heads'].append(item)
    report['elapsed_seconds']=time.perf_counter()-start
    args.report_out.parent.mkdir(parents=True,exist_ok=True);write_json(args.report_out,report)


if __name__=='__main__':main()
