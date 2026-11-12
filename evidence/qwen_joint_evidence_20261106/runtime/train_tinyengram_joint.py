"""Joint late-decoder/memory training against separately trained strong controls.

Frozen prefix, tied LM head, tokenizer and addressing remain unchanged. Train
the last decoder layers with full memory, head constants, or no memory under a
fixed common protocol. Final checkpoints only; no test-based selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer,Qwen3Config,Qwen3ForCausalLM
import transformers

from .engram_row_adapter import file_sha256
from .frozen_tinyengram_prefix import FrozenTinyEngramPrefix,check_reuse
from .lm_quality_metrics import next_token_metrics,summarize_windows,memory_qualification
from .tinyengram_checkpoint import load_upstream,source_revision,export_tinyengram_checkpoint,UPSTREAM_SHA
from .train_tinyengram_identity import sample_windows,heldout_windows,tensor_sha,write_json,evaluate_seed


class HeadConstantEmbedding(nn.Module):
    """One trainable vector per original head; retain the logical hash layout."""
    def __init__(self,source):
        super().__init__()
        self.num_heads=source.num_heads;self.embedding_dim=source.embedding_dim
        self.register_buffer('offsets',source.offsets.detach().clone())
        offsets=source.offsets.tolist()+[len(source.embedding.weight)]
        self.sizes=[b-a for a,b in zip(offsets,offsets[1:])]
        self.embedding=nn.Embedding(self.num_heads,self.embedding_dim,device=source.embedding.weight.device)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.stack([source.embedding.weight[a:b].mean(0) for a,b in zip(offsets,offsets[1:])]))

    def forward(self,ids):
        if ids.shape[-1]!=self.num_heads:raise ValueError('head layout changed')
        heads=torch.arange(self.num_heads,device=ids.device).expand_as(ids)
        return self.embedding(heads)

    def expanded(self):
        return torch.repeat_interleave(self.embedding.weight,torch.tensor(self.sizes,device=self.embedding.weight.device),dim=0)


def score_off(model,reuse,windows,prefixes,qwen):
    result={s:[] for s in windows}
    with torch.no_grad():
        for split,samples in windows.items():
            for i,ids in enumerate(samples):
                ids=ids.unsqueeze(0)
                hidden=torch.from_numpy(np.array(prefixes[split][i:i+1],copy=True)).to(reuse.device)
                output=reuse.forward(ids,hidden,use_memory=False)
                if i==0:
                    qwen.set_skip_engram(True)
                    try:
                        if not torch.equal(model(ids.to(reuse.device),use_cache=False).logits.view(torch.uint8),output.contiguous().view(torch.uint8)):
                            raise AssertionError('plain baseline prefix changed full logits')
                    finally:qwen.set_skip_engram(False)
                m=next_token_metrics(output,output,ids.to(reuse.device));m['window_index']=i
                result[split].append(m)
    return {'windows':{'source':result},'summary':{s:{'source':summarize_windows(v)} for s,v in result.items()}}


def save_plain_baseline(model, tokenizer, checkpoint, windows, reuse, qwen):
    """Remove the unused branch and check the saved ordinary Qwen on each split."""
    plain=Qwen3ForCausalLM(model.config)
    plain.load_state_dict({n:v for n,v in model.state_dict().items() if 'engram_layers' not in n},strict=True)
    plain.save_pretrained(checkpoint,safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    del plain
    plain=Qwen3ForCausalLM.from_pretrained(checkpoint,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='eager').to(reuse.device).eval()
    checks={}
    with torch.no_grad():
        for split,samples in windows.items():
            ids=samples[:1].to(reuse.device)
            qwen.set_skip_engram(True)
            try:expected=model(ids,use_cache=False).logits
            finally:qwen.set_skip_engram(False)
            actual=plain(ids,use_cache=False).logits
            if not torch.equal(expected.contiguous().view(torch.uint8),actual.contiguous().view(torch.uint8)):
                raise AssertionError('saved ordinary Qwen changed memory-off logits: '+split)
            checks[split]={'logits_byte_exact':True,'targets':ids.shape[1]-1}
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base-model','corpus','upstream-dir','protocol','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--native-library',type=Path,required=True)
    p.add_argument('--control',choices=('full','head-constant','off'),required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--prefix-from',type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=p.parse_args();protocol=json.loads(args.protocol.read_text())
    if (protocol['schema_version']!=1 or args.output_dir.exists() or args.seed not in protocol['seeds']
            or not protocol['train_decoder_suffix'] or args.control not in protocol['controls']
            or protocol['training_windows']%protocol['batch_size'] or protocol['epochs']<1
            or not 0<protocol['training_fraction']<protocol['test_start_fraction']<1):
        p.error('fresh output, registered seed and joint-decoder protocol required')
    args.output_dir.mkdir(parents=True)
    started=time.perf_counter()
    report={'schema_version':1,'status':'running','control':args.control,'seed':args.seed,'protocol':protocol,
            'protocol_sha256':file_sha256(args.protocol),'code_sha':source_revision(),'command':shlex.join(sys.orig_argv),
            'upstream_sha':UPSTREAM_SHA,'source_file_hashes':{name:file_sha256(Path(__file__).parent/name) for name in
                ('train_tinyengram_joint.py','frozen_tinyengram_prefix.py','train_tinyengram_identity.py','tinyengram_checkpoint.py','lm_quality_metrics.py')},
            'environment':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,
                           'numpy':np.__version__,'os':platform.platform(),'device':args.device,'threads':protocol['threads']},
            'limitations':['exploratory fixed protocol after earlier frozen-backbone failures; one cached corpus and partially adapted Qwen',
                           'same sampling/epochs/optimizer schedule, different parameter counts; not equal-parameter architecture dominance',
                           'head constants initialized from the corresponding source head means; the full model starts from distinct rows',
                           'prefix cache/dense-source quality oracle costs are not serving memory or throughput',
                           'held-out overlap with original Qwen pretraining is unknown; paired-window bootstrap is not population confidence']}
    def save():
        report['elapsed_seconds']=time.perf_counter()-started;write_json(args.output_dir/'report.json',report)
    try:
        if file_sha256(args.corpus)!=protocol['corpus_sha256']:raise ValueError('corpus changed')
        torch.set_num_threads(protocol['threads']);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        device=torch.device(args.device)
        if device.type=='cuda':
            device=torch.device('cuda',torch.cuda.current_device());properties=torch.cuda.get_device_properties(device)
            report['environment'].update(gpu=properties.name,gpu_uuid=str(properties.uuid),cuda=torch.version.cuda,tf32=False)
        torch.manual_seed(args.seed)
        tokenizer=AutoTokenizer.from_pretrained(args.base_model,local_files_only=True,trust_remote_code=False)
        text=args.corpus.read_text();a,b=int(len(text)*protocol['training_fraction']),int(len(text)*protocol['test_start_fraction'])
        encoded={s:tokenizer(t,return_tensors='pt',add_special_tokens=False).input_ids for s,t in
                 [('train',text[:a]),('calibration',text[a:b]),('test',text[b:])]}
        report['splits']={s:{'tokens':v.shape[1],'token_ids_sha256':tensor_sha(v)} for s,v in encoded.items()}
        length,batch=protocol['window_tokens'],protocol['batch_size']
        training,selected=sample_windows(encoded['train'],length,protocol['training_windows'])
        windows={s:heldout_windows(v,length,protocol[s+'_windows'],protocol.get(s+'_window_start',0)) for s,v in encoded.items() if s!='train'}
        report['sampled_train_windows']={'token_ids_sha256':tensor_sha(training),'window_indices_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),'count':len(training)}
        report['training_targets']=len(training)*(length-1)*protocol['epochs']
        engram,qwen=load_upstream(args.upstream_dir)
        config=Qwen3Config.from_pretrained(args.base_model,local_files_only=True)
        layer_id=config.num_hidden_layers-protocol['layer_from_end']
        ecfg=engram.EngramConfig(tokenizer_name_or_path=str(args.base_model),engram_vocab_size=protocol['ngram_vocab_sizes'],
            max_ngram_size=protocol['max_ngram_size'],n_embed_per_ngram=protocol['n_embed_per_ngram'],
            n_head_per_ngram=protocol['n_head_per_ngram'],layer_ids=[layer_id],pad_id=tokenizer.pad_token_id,
            kernel_size=protocol['kernel_size'],seed=args.seed)
        engram.set_global_step(0)
        model,loading=qwen.EngramQwenForCausalLM.from_pretrained(args.base_model,engram_cfg=ecfg,local_files_only=True,
            torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
        missing={n for n in model.state_dict() if 'engram_layers' in n}
        if set(loading['missing_keys'])!=missing or any(loading[k] for k in ('unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('backbone load changed: '+repr(loading))
        branch=model.model.engram_layers[str(layer_id)]
        original_embedding=branch.multi_head_embedding
        source_initial=tensor_sha(original_embedding.embedding.weight)
        if args.control=='head-constant':branch.multi_head_embedding=HeadConstantEmbedding(original_embedding)
        suffix_names=tuple(f'model.layers.{i}.' for i in range(layer_id,config.num_hidden_layers))
        for name,param in model.named_parameters():
            param.requires_grad_(name.startswith(suffix_names) or 'engram_layers' in name and args.control!='off')
        model.to(device).eval();reuse=FrozenTinyEngramPrefix(model,train_decoder_suffix=True)
        frozen_hash=reuse.frozen_weight_sha256()
        memory=branch.multi_head_embedding.embedding.weight
        initial=memory.detach().clone()
        report.update(layer_id=layer_id,initial_loading=loading,initial_full_memory_sha256=source_initial,
            frozen_weight_sha256=frozen_hash,trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
            trained_parameter_names=[n for n,v in model.named_parameters() if v.requires_grad],
            memory_trainable_parameters=memory.numel() if memory.requires_grad else 0,
            logical_memory_rows=len(original_embedding.embedding.weight),head_offsets=original_embedding.offsets.tolist(),
            initial_decoder_sha256={n:tensor_sha(v) for n,v in model.named_parameters() if n.startswith(suffix_names)},
            reuse_check=check_reuse(model,reuse,training[:batch],gradients=True,use_memory=args.control!='off',set_skip_memory=qwen.set_skip_engram))
        save()
        prefixes={}
        if args.prefix_from:
            prior=json.loads((args.prefix_from/'report.json').read_text())
            if (prior['protocol']!=protocol or prior['frozen_weight_sha256']!=frozen_hash or prior['splits']!=report['splits']
                    or prior['sampled_train_windows']!=report['sampled_train_windows'] or prior['environment']['device']!=args.device):
                raise ValueError('prefix protocol/data/frozen weights changed')
            report['prefix_files']=prior['prefix_files'];report['prefix_source_report_sha256']=file_sha256(args.prefix_from/'report.json')
            for split,item in prior['prefix_files'].items():
                path=args.prefix_from/(split+'-prefix.npy')
                if file_sha256(path)!=item['sha256']:raise ValueError('cached prefix bytes changed')
                prefixes[split]=np.load(path,mmap_mode='r',allow_pickle=False)
        else:
            report['prefix_files']={}
            for split,samples in {'train':training,**windows}.items():
                path=args.output_dir/(split+'-prefix.npy')
                array=np.lib.format.open_memmap(path,mode='w+',dtype='<f4',shape=(len(samples),length,config.hidden_size))
                step_batch=batch if split=='train' else 1
                for i in range(0,len(samples),step_batch):array[i:i+step_batch]=reuse.capture(samples[i:i+step_batch]).cpu().numpy()
                array.flush();array._mmap.close()
                report['prefix_files'][split]={'bytes':path.stat().st_size,'sha256':file_sha256(path)}
                prefixes[split]=np.load(path,mmap_mode='r',allow_pickle=False)
        if not torch.equal(reuse.capture(training[:batch]).contiguous().view(torch.uint8),
                           torch.from_numpy(np.array(prefixes['train'][:batch],copy=True)).to(device).contiguous().view(torch.uint8)):
            raise AssertionError('serialized prefix does not match current frozen model')
        params=[v for v in model.parameters() if v.requires_grad]
        suffix=[v for n,v in model.named_parameters() if n.startswith(suffix_names)]
        other=[v for n,v in model.named_parameters() if 'engram_layers' in n and v.requires_grad and v is not memory]
        groups=[{'params':suffix,'lr':protocol['decoder_learning_rate']}]
        if args.control!='off':groups.extend([{'params':[memory],'lr':protocol['memory_learning_rate']},{'params':other,'lr':protocol['branch_learning_rate']}])
        optimizer=torch.optim.Adam(groups)
        report['losses']=[];step=0;rng=np.random.default_rng(args.seed)
        training_started=time.perf_counter()
        save()
        for epoch in range(protocol['epochs']):
            order=rng.permutation(len(training))
            for start in range(0,len(training),batch):
                idx=order[start:start+batch];samples=training[torch.from_numpy(idx)]
                hidden=torch.from_numpy(np.array(prefixes['train'][idx],copy=True)).to(device)
                optimizer.zero_grad(set_to_none=True);engram.set_global_step(step)
                logits=reuse.forward(samples,hidden,use_memory=args.control!='off')
                loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),samples[:,1:].to(device).reshape(-1))
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,protocol['gradient_clip']);optimizer.step()
                report['losses'].append(float(loss.detach()));step+=1
                del logits,loss,hidden
                if step%32==0:
                    report['step']=step;save();print(args.control,args.seed,epoch+1,step,report['losses'][-1],flush=True)
        if reuse.frozen_weight_sha256()!=frozen_hash:raise AssertionError('frozen prefix/head changed during joint training')
        report.update(steps=step,training_seconds=time.perf_counter()-training_started,
                      final_reuse_check=check_reuse(model,reuse,training[:batch],gradients=True,use_memory=args.control!='off',set_skip_memory=qwen.set_skip_engram),
                      frozen_weight_hash_unchanged=True,trained_decoder_sha256={n:tensor_sha(v) for n,v in model.named_parameters() if n.startswith(suffix_names)},
                      changed_physical_memory_rows=int((memory.detach()!=initial).any(1).sum()))
        engram.set_global_step(step)
        if args.control=='head-constant':
            expanded=branch.multi_head_embedding.expanded().detach()
            original_embedding.to(device)
            with torch.no_grad():original_embedding.embedding.weight.copy_(expanded)
            branch.multi_head_embedding=original_embedding
            report['constant_vector_payload_bytes']=memory.numel()*4
        if args.control=='off':
            quality=score_off(model,reuse,windows,prefixes,qwen)
            report['plain_reload_checks']=save_plain_baseline(model,tokenizer,args.output_dir/'checkpoint',windows,reuse,qwen)
            report['checkpoint_contract']='ordinary Qwen checkpoint, unused memory branch removed without changing trained baseline outputs'
            report['plain_baseline_sha256']=file_sha256(args.output_dir/'checkpoint/model.safetensors')
        else:
            checkpoint=args.output_dir/'checkpoint';bundle=args.output_dir/'bundle'
            model.save_pretrained(checkpoint,safe_serialization=True);tokenizer.save_pretrained(checkpoint)
            write_json(checkpoint/'engram_config.json',ecfg.to_dict());write_json(checkpoint/'trainer_state.json',{'global_step':step,'corpus_sha256':protocol['corpus_sha256']})
            export_tinyengram_checkpoint(checkpoint,bundle,upstream_dir=args.upstream_dir,block_rows=64,predictor='xor')
            ws=evaluate_seed(model,reuse,windows,prefixes,bundle,args.native_library)
            summaries={s:{name:summarize_windows(v[s]) for name,v in ws.items()} for s in windows}
            quality={'windows':ws,'summary':summaries,'memory_qualified_on_calibration':memory_qualification(summaries['calibration'],minimum_ppl_effect=protocol['minimum_ppl_effect']),
                     'test_confirmation':memory_qualification(summaries['test'],minimum_ppl_effect=protocol['minimum_ppl_effect'])}
            report.update(trained_memory_sha256=tensor_sha(branch.multi_head_embedding.embedding.weight),
                complete_bundle_bytes=sum(v.stat().st_size for v in bundle.rglob('*') if v.is_file()),bundle_manifest_sha256=file_sha256(bundle/'bundle.json'))
        write_json(args.output_dir/'quality.json',quality)
        report.update(status='passed',quality_sha256=file_sha256(args.output_dir/'quality.json'),quality={k:v for k,v in quality.items() if k!='windows'},
                      complete_checkpoint_bytes=sum(v.stat().st_size for v in (args.output_dir/'checkpoint').rglob('*') if v.is_file()))
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:save();print('Report:',args.output_dir/'report.json',flush=True)


if __name__=='__main__':main()
