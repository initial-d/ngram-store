"""Build training-only identity vocabularies and verify paired real Qwen models."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .counterfact_capacity import environment,sha,write,Prefixes,batch_indices,filled,ids_for
from .counterfact_models import setup
from .late_memory_replay import load_engram,check_masked_replay,masked_loss
from .ngram_identity import Vocabulary,configure_memory,collision_audit


def build(args):
    if args.output.exists():raise ValueError('fresh vocabulary output required')
    args.output.mkdir(parents=True)
    meta=json.loads((args.prepared/'prepare.json').read_text())
    if sha(args.prepared/'sequences.json')!=meta['payload_sha256']:raise ValueError('prepared inputs changed')
    payload=json.loads((args.prepared/'sequences.json').read_text())
    config=json.loads((args.base/'config.json').read_text());text=config.get('text_config',config)
    layer=text['num_hidden_layers']-8;module=load_engram(args.upstream)
    source=module.NgramHashMapping([257,521,1031],4,64,2,[layer],str(args.base),
                                  __import__('transformers').AutoTokenizer.from_pretrained(args.base,local_files_only=True).pad_token_id,3)
    started=time.perf_counter();vocabulary=Vocabulary.build(payload,source,sha(args.prepared/'prepare.json'))
    vocabulary.save(args.output/'vocabulary.npz')
    coverage={}
    for split in ('calibration','test'):
        coverage[split]={}
        for group in ('rewrite','paraphrase','neighborhood'):
            counts=np.zeros(3,dtype=np.int64);total=0
            for c in payload['cases']:
                if c['split']!=split:continue
                for pair in c['groups'][group]:
                    sequence=payload['sequences'][pair['new']]
                    # Actual question prefix, excluding candidate answer tokens.
                    x=source.compressed_tokenizer(np.asarray([sequence['ids'][:sequence['target_start']]],dtype=np.int64))
                    known=vocabulary.known_ids(x)[0,:,::2]
                    counts+=(known!=0).sum(0);total+=len(known)
            coverage[split][group]={'question_prefix_positions':total,
                                    'known_by_order':{str(n):int(counts[n-2]) for n in (2,3,4)}}
    write(args.output/'report.json',{'status':'built','base':str(args.base),'prepared_sha256':sha(args.prepared/'prepare.json'),
        'vocabulary_sha256':sha(args.output/'vocabulary.npz'),'vocabulary_file_bytes':(args.output/'vocabulary.npz').stat().st_size,
        'metadata':vocabulary.metadata,'collision_audit_seed3':collision_audit(vocabulary,source),'question_prefix_coverage':coverage,
        'elapsed_seconds':time.perf_counter()-started,'code_sha':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'command':__import__('shlex').join(__import__('sys').orig_argv)})
    print(json.dumps({'status':'built','keys':{n:len(vocabulary.keys[n]) for n in (2,3,4)}}),flush=True)


def check(args):
    if args.output.exists():raise ValueError('fresh preflight output required')
    args.output.mkdir(parents=True);env=environment();vocabulary=Vocabulary.load(args.vocabulary)
    if vocabulary.metadata['prepared_sha256']!=sha(args.prepared/'prepare.json'):raise ValueError('vocabulary/data mismatch')
    payload=json.loads((args.prepared/'sequences.json').read_text());seq=payload['sequences'];batch=filled(batch_indices(payload['train'],seq,8)[0],8)
    ids=ids_for(batch,seq);starts=[seq[i]['target_start'] for i in batch];keep=max(len(seq[i]['target_ids']) for i in batch)+1
    objects=[];infos=[]
    try:
        for exact in (False,True):
            model,tok,replay,module,cfg,loading=setup(args.base,args.upstream,'full',[257,521,1031],3,train_decoder_suffix=False)
            info=configure_memory(replay,module,vocabulary,exact,3);objects.append((model,replay,module));infos.append(info)
        hm,hr,_=objects[0];em,er,_=objects[1]
        if infos[0]['head_sizes']!=infos[1]['head_sizes']:raise AssertionError('row budgets differ')
        if sum(p.numel() for p in hm.parameters() if p.requires_grad)!=sum(p.numel() for p in em.parameters() if p.requires_grad):
            raise AssertionError('trainable parameter budgets differ')
        with torch.no_grad():
            hs,es=hr.memory.multi_head_embedding,er.memory.multi_head_embedding
            for head,(hoff,eoff) in enumerate(zip(hs.offsets.tolist(),es.offsets.tolist())):
                keys=vocabulary.keys[2+head//2]
                local=hr.memory.hash_mapping.source._get_ngram_hashes(keys,hr.layer_id)[:,-1,head]+1
                a=hs.embedding.weight[torch.from_numpy(local+hoff).to(hr.device)]
                b=es.embedding.weight[eoff+1:eoff+1+len(keys)]
                if not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)):
                    raise AssertionError('initial effective vectors differ for a training key')
        with torch.no_grad():
            a=hr.complete(ids,True,keep);b=er.complete(ids,True,keep)
            if not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)):
                raise AssertionError('paired controls do not start with byte-exact effective logits')
        checked=[]
        with torch.no_grad():
            for c in payload['cases']:
                if c['split']!='test':continue
                for group in ('rewrite','paraphrase','neighborhood'):
                    for pair in c['groups'][group]:
                        s=seq[pair['new']];query=torch.tensor([s['ids'][:s['target_start']]])
                        a=hr.complete(query,True,1);b=er.complete(query,True,1)
                        if not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)):
                            raise AssertionError('paired question outputs differ at initialization')
                        checked.append({'case_id':c['case_id'],'group':group,'tokens':query.shape[1],'byte_exact':True})
                        if len(checked)==16:break
                    if len(checked)==16:break
                if len(checked)==16:break
        replay_checks=[];fingerprints=[]
        for model,replay,module in objects:
            prefixes=Prefixes(args.cache,args.prepared,replay,8)
            if not torch.equal(prefixes.get(batch,replay.device),replay.capture(ids)):
                raise AssertionError('cached prefixes differ')
            replay_checks.append(check_masked_replay(replay,ids,starts,True,keep))
            fingerprints.append((replay.frozen_hash(),replay.frozen_suffix_hash()))
        # Both loss graphs initially have identical effective vectors. Their
        # projection/gate gradients must agree, while hash row gradients pool
        # contributions from exact keys that share each bucket.
        if fingerprints[0]!=fingerprints[1]:raise AssertionError('paired immutable backbones differ')
        # check_masked_replay clears gradients; obtain them again for pooling.
        for model,replay,module in objects:
            masked_loss(replay.complete(ids,True,keep),ids,starts).backward()
        nonrow=True
        eg=dict(em.named_parameters());hg=dict(hm.named_parameters())
        for name,p in hg.items():
            if p.requires_grad and 'multi_head_embedding' not in name:
                a,b=p.grad,eg[name].grad
                if (a is None)!=(b is None) or a is not None and not torch.equal(a,b):nonrow=False
        if not nonrow:raise AssertionError('initial nonrow gradients differ')
        hs=hr.memory.multi_head_embedding;es=er.memory.multi_head_embedding
        target=hs.embedding.weight.grad.detach().cpu().double();pooled=torch.zeros_like(target)
        for head,(hoff,eoff) in enumerate(zip(hs.offsets.tolist(),es.offsets.tolist())):
            n=2+head//2;keys=vocabulary.keys[n]
            local=hr.memory.hash_mapping.source._get_ngram_hashes(keys,hr.layer_id)[:,-1,head]+1
            rows=es.embedding.weight.grad[eoff+1:eoff+1+len(keys)].detach().cpu().double()
            pooled.index_add_(0,torch.from_numpy(local+hoff),rows)
        delta=float((target-pooled).abs().max())
        if not torch.allclose(target,pooled,atol=1e-6,rtol=1e-4):raise AssertionError('hash gradients do not pool exact key gradients')
        for index,(model,replay,module) in enumerate(objects):
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001);optimizer.step()
            if (replay.frozen_hash(),replay.frozen_suffix_hash())!=fingerprints[index]:raise AssertionError('frozen backbone changed')
        write(args.output/'report.json',{'status':'passed','environment':env,'vocabulary_sha256':sha(args.vocabulary),
            'paired_training_logits_byte_exact':True,'question_requests':checked,'masked_replay':replay_checks,
            'initial_nonrow_gradients_byte_exact':True,'gradient_pooling_max_absolute_error':delta,
            'all_training_key_vectors_initially_byte_exact':True,'trainable_parameter_budget_matched':True,
            'frozen_prefix_and_head_sha256':fingerprints[0][0],'frozen_decoder_suffix_sha256':fingerprints[0][1],
            'gradient_pooling_tolerance':{'atol':1e-6,'rtol':1e-4},'backbone_unchanged_after_optimizer':True,
            'variants':infos,'scope':'Initialization/gradient fidelity only; no trained full-versus-hash quality or storage advantage'})
        print(json.dumps({'status':'passed','requests':len(checked),'gradient_pooling_max_absolute_error':delta}),flush=True)
    finally:
        for model,replay,module in objects:replay.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    for action in ('build','check'):
        p=sub.add_parser(action)
        for name in ('base','upstream','prepared','output'):p.add_argument('--'+name,type=Path,required=True)
        if action=='check':
            p.add_argument('--vocabulary',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args();globals()[args.action](args)


if __name__=='__main__':main()
