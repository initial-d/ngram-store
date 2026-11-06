"""Source-pinned MultiCounterFact preparation, exact prefix cache and editing.

Use task-local files only. Token IDs, model deltas and prefix states are private
run artifacts; public evidence contains scalar scores and hash manifests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoTokenizer

from .counterfact_data import load_records,build_sequences,batch_indices,case_metrics,summarize
from .counterfact_models import setup
from .counterfact_models import load_model
from .counterfact_retrieval import EditFactBM25
from .late_memory_replay import check_masked_replay,masked_loss


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,default=lambda x:sorted(x) if isinstance(x,set) else str(x))+'\n')
    tmp.replace(path)


def environment():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    e={'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,
       'numpy':np.__version__,'os':platform.platform(),'cuda':torch.version.cuda,'tf32':False,
       'command':shlex.join(sys.orig_argv),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'source_hashes':{p.name:sha(p) for p in [*Path(__file__).parent.glob('counterfact*.py'),Path(__file__).parent/'late_memory_replay.py']}}
    if torch.cuda.is_available():
        p=torch.cuda.get_device_properties(0)
        e.update(gpu=p.name,gpu_uuid=str(p.uuid),gpu_memory=p.total_memory)
    return e


def filled(batch,size):return batch+[batch[0]]*(size-len(batch))


def ids_for(batch,sequences):return torch.tensor([sequences[i]['ids'] for i in batch],dtype=torch.int64)


def prepare(args):
    if args.output.exists():raise ValueError('fresh output required')
    args.output.mkdir(parents=True)
    tok=AutoTokenizer.from_pretrained(args.base,local_files_only=True,trust_remote_code=False)
    records=load_records(args.data)
    # Official order gives nested edit sets, shared by all seeds/architectures.
    records=records[:args.edits]
    if len(records)!=args.edits:raise ValueError('requested edit count unavailable')
    sequences,train,cases=build_sequences(records,tok)
    payload={'sequences':sequences,'train':train,'cases':cases}
    write(args.output/'sequences.json',payload)
    batches=batch_indices(range(len(sequences)),sequences,args.batch)
    write(args.output/'batches.json',batches)
    tokens=sum(len(s['ids']) for s in sequences)
    report={'status':'prepared','data_sha256':sha(args.data),'base':str(args.base),'edits':args.edits,
            'batch_size':args.batch,'unique_sequences':len(sequences),'tokens':tokens,'batches':len(batches),
            'token_boundary_mismatches':0,'payload_sha256':sha(args.output/'sequences.json'),
            'batches_sha256':sha(args.output/'batches.json'),
            'training_sequences':len(train),'filler_sequence_exposures':sum(args.batch-len(b) for b in batches),
            'edited_neighborhood_prompts':sum(p['prompt_matches_an_edited_rewrite'] for c in cases for p in c['groups']['neighborhood']),
            'split_counts':{s:sum(c['split']==s for c in cases) for s in ('calibration','test')}}
    write(args.output/'prepare.json',report);print(json.dumps(report),flush=True)


def cache(args):
    env=environment();meta=json.loads((args.prepared/'prepare.json').read_text())
    if sha(args.prepared/'sequences.json')!=meta['payload_sha256']:raise ValueError('prepared IDs changed')
    seq=json.loads((args.prepared/'sequences.json').read_text())['sequences']
    batches=json.loads((args.prepared/'batches.json').read_text())[args.shard::args.shards]
    args.output.mkdir(parents=True,exist_ok=True)
    stem=args.output/f'shard-{args.shard}'
    if stem.with_suffix('.json').exists():raise ValueError('cache shard already exists')
    model,tok,replay,module,cfg,loading=setup(args.base,args.upstream,'off',[257,521,1031],3,suffix_layers=args.suffix)
    hdim=replay.text.config.hidden_size;count=sum(len(seq[i]['ids']) for b in batches for i in b)
    array=np.memmap(stem.with_suffix('.f32'),mode='w+',dtype='<f4',shape=(count,hdim))
    started=time.perf_counter();index={};offset=0
    for n,b in enumerate(batches):
        ids=ids_for(filled(b,meta['batch_size']),seq)
        h=replay.capture(ids).cpu().numpy()
        for j,i in enumerate(b):
            length=len(seq[i]['ids']);array[offset:offset+length]=h[j]
            index[str(i)]=[offset,length];offset+=length
        if n%100==0:
            print(json.dumps({'stage':'cache','shard':args.shard,'done':n,'batches':len(batches),
                              'elapsed':time.perf_counter()-started}),flush=True)
    array.flush();del array
    report={'status':'complete','shard':args.shard,'shards':args.shards,'sequences':index,'hidden_size':hdim,
            'tokens':count,'bytes':stem.with_suffix('.f32').stat().st_size,'sha256':sha(stem.with_suffix('.f32')),
            'prepared_sha256':sha(args.prepared/'prepare.json'),'batch_size':meta['batch_size'],
            'frozen_weight_sha256':replay.frozen_hash(),'environment':env,'loading':loading,
            'elapsed_seconds':time.perf_counter()-started}
    write(stem.with_suffix('.json'),report);print(json.dumps({k:v for k,v in report.items() if k!='sequences'}),flush=True)


class Prefixes:
    def __init__(self,path,prepared,replay,batch_size):
        self.index={};self.arrays=[];self.hashes=[]
        frozen_hash=replay.frozen_hash();prepared_hash=sha(prepared/'prepare.json')
        for file in sorted(path.glob('shard-*.json')):
            meta=json.loads(file.read_text());raw=file.with_suffix('.f32')
            if meta['status']!='complete' or meta['prepared_sha256']!=prepared_hash or meta['batch_size']!=batch_size:
                raise ValueError('prefix cache protocol mismatch')
            if meta['frozen_weight_sha256']!=frozen_hash:raise ValueError('prefix weights changed')
            if raw.stat().st_size!=meta['bytes'] or sha(raw)!=meta['sha256']:raise ValueError('prefix cache changed')
            a=np.memmap(raw,mode='r',dtype='<f4',shape=(meta['tokens'],meta['hidden_size']))
            for i,entry in meta['sequences'].items():
                if int(i) in self.index:raise ValueError('duplicate prefix')
                self.index[int(i)]=(len(self.arrays),*entry)
            self.arrays.append(a);self.hashes.append({'file':str(raw),'sha256':meta['sha256']})
    def get(self,batch,device):
        h=[]
        for i in batch:
            shard,offset,length=self.index[i];h.append(np.array(self.arrays[shard][offset:offset+length],copy=True))
        return torch.from_numpy(np.stack(h)).to(device)


def choice_scores(logits,batch,sequences,valid):
    scores={}
    for j,i in enumerate(batch[:valid]):
        s=sequences[i];targets=torch.tensor(s['target_ids'],device=logits.device)
        n=len(targets);x=logits[j,-n-1:-1].float()
        nll=-x.log_softmax(-1).gather(-1,targets[:,None]).squeeze(-1)
        # Match the official scorer's sequential FP32 accumulator.
        total=np.float32(0.)
        for value in nll.tolist():total=np.float32(total+value)
        scores[str(i)]={'nll':float(np.float32(total/n)),'correct':bool(torch.equal(x.argmax(-1),targets))}
    return scores


def train(args):
    env=environment();protocol=json.loads(args.protocol.read_text())
    if args.output.exists():raise ValueError('fresh run output required')
    args.output.mkdir(parents=True)
    started=time.perf_counter();report={'status':'running','environment':env,'seed':args.seed,'variant':args.variant,
        'protocol':protocol,'protocol_sha256':sha(args.protocol),'prepared_sha256':sha(args.prepared/'prepare.json')}
    def save():
        report['elapsed_seconds']=time.perf_counter()-started;write(args.output/'report.json',report)
    save()
    try:
        if args.seed not in protocol['seeds']:raise ValueError('unregistered seed')
        variant=protocol['variants'][args.variant]
        meta=json.loads((args.prepared/'prepare.json').read_text())
        if meta['batch_size']!=protocol['batch_size'] or meta['edits'] not in protocol['edit_counts']:raise ValueError('unregistered data')
        if sha(args.prepared/'sequences.json')!=meta['payload_sha256']:raise ValueError('IDs changed')
        payload=json.loads((args.prepared/'sequences.json').read_text());seq=payload['sequences'];samples=payload['train']
        control=variant['control'];enabled=control!='off'
        model,tok,replay,module,cfg,loading=setup(args.base,args.upstream,control,variant['vocab_sizes'],args.seed,
            suffix_layers=protocol['suffix_layers'],train_decoder_suffix=protocol.get('train_decoder_suffix',True))
        report.update(loading=loading,base_model=str(args.base),frozen_weight_sha256=replay.frozen_hash(),
            frozen_decoder_suffix_sha256=replay.frozen_suffix_hash(),
            trained_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad],
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            memory_parameters=sum(p.numel() for p in replay.memory.parameters()),
            physical_rows=len(replay.memory.multi_head_embedding.embedding.weight) if hasattr(replay.memory,'multi_head_embedding') else 0)
        prefixes=Prefixes(args.cache,args.prepared,replay,protocol['batch_size'])
        if len(prefixes.index)!=len(seq):raise ValueError('incomplete cache')
        report['prefix_files']=prefixes.hashes
        first=batch_indices(samples,seq,protocol['batch_size'])[0]
        b=filled(first,protocol['batch_size']);ids=ids_for(b,seq);hidden=prefixes.get(b,replay.device)
        if not torch.equal(hidden,replay.capture(ids)):raise AssertionError('rebatched frozen cache changed prefix bytes')
        starts=[seq[i]['target_start'] for i in b]
        report['replay_check']=check_masked_replay(replay,ids,starts,enabled,max(len(seq[i]['target_ids']) for i in b)+1)
        suffix=[p for n,p in model.named_parameters() if p.requires_grad and n in replay.suffix_names]
        branch=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('ngram_memory.') and 'multi_head_embedding' not in n]
        rows=[p for n,p in model.named_parameters() if p.requires_grad and 'ngram_memory.multi_head_embedding' in n]
        groups=[{'params':suffix,'lr':protocol['suffix_lr']}] if suffix else []
        if branch:groups.append({'params':branch,'lr':protocol['branch_lr']})
        if rows:groups.append({'params':rows,'lr':protocol['row_lr']})
        if not groups:raise ValueError('no trainable parameters; use baseline action for an unedited frozen model')
        optimizer=torch.optim.Adam(groups)
        rng=np.random.default_rng(args.seed);steps=0;report['epochs']=[];save()
        training_started=time.perf_counter()
        for epoch in range(protocol['epochs']):
            losses=[];exposures=0;epoch_started=time.perf_counter()
            for n,valid in enumerate(batch_indices(samples,seq,protocol['batch_size'],rng=rng)):
                b=filled(valid,protocol['batch_size']);ids=ids_for(b,seq)
                starts=[seq[i]['target_start'] for i in valid]+[ids.shape[1]]*(len(b)-len(valid))
                k=max(len(seq[i]['target_ids']) for i in valid)+1
                optimizer.zero_grad(set_to_none=True)
                logits=replay.forward(ids,prefixes.get(b,replay.device),enabled,k)
                loss=masked_loss(logits,ids,starts)
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],protocol['clip_norm'])
                optimizer.step();steps+=1;module.set_global_step(steps)
                losses.append(float(loss.detach()));exposures+=len(valid)
                if n%100==0:print(json.dumps({'stage':'train','epoch':epoch,'batch':n,'loss':losses[-1],'steps':steps}),flush=True)
            report['epochs'].append({'epoch':epoch,'mean_batch_loss':float(np.mean(losses)),
                'actual_sequences':exposures,'steps':steps,'elapsed_seconds':time.perf_counter()-epoch_started});save()
        report['training_seconds']=time.perf_counter()-training_started
        if replay.frozen_hash()!=report['frozen_weight_sha256']:raise AssertionError('frozen weights changed')
        if replay.frozen_suffix_hash()!=report['frozen_decoder_suffix_sha256']:raise AssertionError('frozen decoder suffix changed')
        # A delta plus its immutable shared base is the complete deployment recipe.
        state={n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
        save_file(state,str(args.output/'delta.safetensors'));del state
        report['delta']={'bytes':(args.output/'delta.safetensors').stat().st_size,'sha256':sha(args.output/'delta.safetensors')}
        report['status']='evaluating';save();scores={};evaluation_started=time.perf_counter()
        all_eval=sorted({p[k] for c in payload['cases'] for pairs in c['groups'].values() for p in pairs for k in ('new','true')})
        with torch.no_grad():
            for n,valid in enumerate(batch_indices(all_eval,seq,protocol['batch_size'])):
                b=filled(valid,protocol['batch_size']);ids=ids_for(b,seq);k=max(len(seq[i]['target_ids']) for i in b)+1
                logits=replay.forward(ids,prefixes.get(b,replay.device),enabled,k)
                scores.update(choice_scores(logits,b,seq,len(valid)))
                if n%200==0:print(json.dumps({'stage':'eval','batches':n,'choices':len(scores)}),flush=True)
        cases=case_metrics(payload['cases'],scores)
        write(args.output/'choice-scores.json',scores);write(args.output/'case-scores.json',cases)
        report['evaluation_seconds']=time.perf_counter()-evaluation_started
        report['metrics']={s:summarize([c for c in cases if s=='all' or c['split']==s]) for s in ('all','calibration','test')}
        report['score_artifacts']={p:{'sha256':sha(args.output/p),'bytes':(args.output/p).stat().st_size} for p in ('choice-scores.json','case-scores.json')}
        report.update(status='complete',optimizer_steps=steps,peak_cuda_allocated=torch.cuda.max_memory_allocated());save()
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());save();raise


def retrieval(args):
    env=environment();protocol=json.loads(args.protocol.read_text());started=time.perf_counter()
    if args.output.exists():raise ValueError('fresh output required')
    args.output.mkdir(parents=True)
    records=load_records(args.data)[:args.edits]
    tok=AutoTokenizer.from_pretrained(args.base,local_files_only=True,trust_remote_code=False)
    index=EditFactBM25(records);model,loading=load_model(args.base);model.to('cuda').eval()
    report={'status':'running','environment':env,'base_model':str(args.base),'edits':args.edits,'loading':loading,
            'protocol_sha256':sha(args.protocol),'data_sha256':sha(args.data),'candidates':{}}
    def save():
        report['elapsed_seconds']=time.perf_counter()-started;write(args.output/'report.json',report)
    save()
    try:
        for k in (1,3):
            retrieval_started=time.perf_counter();lookup={}
            def transform(prefix):
                if prefix not in lookup:lookup[prefix]=index.prefix(prefix,k)[0]
                return lookup[prefix]
            seq,unused,cases=build_sequences(records,tok,prefix_transform=transform)
            report['candidates'][str(k)]={'queries':len(lookup),'index_and_tokenization_seconds':time.perf_counter()-retrieval_started}
            all_eval=sorted({p[x] for c in cases for pairs in c['groups'].values() for p in pairs for x in ('new','true')})
            scores={};eval_started=time.perf_counter()
            with torch.no_grad():
                for n,valid in enumerate(batch_indices(all_eval,seq,protocol['batch_size'])):
                    b=filled(valid,protocol['batch_size']);ids=ids_for(b,seq)
                    keep=max(len(seq[i]['target_ids']) for i in b)+1
                    logits=model(input_ids=ids.to('cuda'),use_cache=False,logits_to_keep=keep).logits
                    scores.update(choice_scores(logits,b,seq,len(valid)))
                    if n%100==0:print(json.dumps({'stage':'rag-eval','top_k':k,'batches':n,'choices':len(scores)}),flush=True)
            evaluated=case_metrics(cases,scores);write(args.output/f'case-scores-k{k}.json',evaluated)
            item=report['candidates'][str(k)]
            item.update(evaluation_seconds=time.perf_counter()-eval_started,
                total_input_tokens=sum(len(seq[i]['ids']) for i in all_eval),
                metrics={s:summarize([c for c in evaluated if s=='all' or c['split']==s]) for s in ('all','calibration','test')},
                case_scores_sha256=sha(args.output/f'case-scores-k{k}.json'));save()
        selected=max((1,3),key=lambda k:(report['candidates'][str(k)]['metrics']['calibration']['harmonic_score'],-k))
        report.update(status='complete',selected_top_k=selected,selection_fold='calibration',
            metrics=report['candidates'][str(selected)]['metrics'],peak_cuda_allocated=torch.cuda.max_memory_allocated());save()
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());save();raise


def baseline(args):
    env=environment();protocol=json.loads(args.protocol.read_text());started=time.perf_counter()
    if args.output.exists():raise ValueError('fresh output required')
    args.output.mkdir(parents=True)
    meta=json.loads((args.prepared/'prepare.json').read_text())
    payload=json.loads((args.prepared/'sequences.json').read_text());seq=payload['sequences']
    model,tok,replay,module,cfg,loading=setup(args.base,args.upstream,'off',[257,521,1031],3,suffix_layers=protocol['suffix_layers'])
    for p in model.parameters():p.requires_grad_(False)
    prefixes=Prefixes(args.cache,args.prepared,replay,protocol['batch_size'])
    scores={};all_eval=sorted({p[k] for c in payload['cases'] for pairs in c['groups'].values() for p in pairs for k in ('new','true')})
    with torch.no_grad():
        for n,valid in enumerate(batch_indices(all_eval,seq,protocol['batch_size'])):
            b=filled(valid,protocol['batch_size']);ids=ids_for(b,seq);k=max(len(seq[i]['target_ids']) for i in b)+1
            logits=replay.forward(ids,prefixes.get(b,replay.device),False,k)
            scores.update(choice_scores(logits,b,seq,len(valid)))
            if n%200==0:print(json.dumps({'stage':'base-eval','batches':n,'choices':len(scores)}),flush=True)
    cases=case_metrics(payload['cases'],scores);write(args.output/'case-scores.json',cases)
    write(args.output/'report.json',{'status':'complete','variant':'unedited-base','environment':env,'loading':loading,
        'base_model':str(args.base),'edits':meta['edits'],'prepared_sha256':sha(args.prepared/'prepare.json'),
        'protocol_sha256':sha(args.protocol),'metrics':{s:summarize([c for c in cases if s=='all' or c['split']==s]) for s in ('all','calibration','test')},
        'case_scores_sha256':sha(args.output/'case-scores.json'),'elapsed_seconds':time.perf_counter()-started})


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--base',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--edits',type=int,required=True);p.add_argument('--batch',type=int,default=8)
    for name in ('cache','train','baseline'):
        p=sub.add_parser(name)
        for x in ('base','upstream','prepared','output'):p.add_argument('--'+x,type=Path,required=True)
        if name=='cache':
            p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True);p.add_argument('--suffix',type=int,default=8)
        else:
            p.add_argument('--cache',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
            if name=='train':p.add_argument('--seed',type=int,required=True);p.add_argument('--variant',required=True)
    p=sub.add_parser('retrieval')
    for x in ('base','data','protocol','output'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--edits',type=int,required=True)
    args=parser.parse_args();globals()[args.action](args)


if __name__=='__main__':main()
