"""Export a completed Qwen editing delta and verify real native model requests.

This checks the frozen-table boundary; a trained model passing this command
does not demonstrate memory necessity or a native serving speed advantage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors.torch import load_file,save_file

from .counterfact_capacity import sha,write,environment
from .counterfact_models import setup
from .device_row_adapter import StagedMultiHeadEmbedding
from .engram_row_adapter import export_multihead


class NativeHashMemory(torch.nn.Module):
    """Keep token/hash/identity addressing on CPU; stage rows to the model."""
    def __init__(self,source):super().__init__();self.source=source
    def forward(self,hidden_states,input_ids):
        return self.source(hidden_states=hidden_states,input_ids=input_ids.cpu())


def configure_identity(replay,module,folder,spec,*,initialize_rows=True):
    """Reconstruct trained local-head IDs before loading weights or native rows."""
    if not spec['protocol'].get('identity_control'):
        if spec.get('native_oov',False):raise ValueError('native OOV requires a trained identity hash control')
        return
    from .ngram_identity import Vocabulary,IdentityMapping,configure_memory
    path=folder/spec['vocabulary']
    if sha(path)!=spec['vocabulary_sha256']:raise ValueError('identity vocabulary checksum changed')
    vocabulary=Vocabulary.load(path)
    if spec.get('prepared_sha256',vocabulary.metadata['prepared_sha256'])!=vocabulary.metadata['prepared_sha256']:
        raise ValueError('identity vocabulary/data mismatch')
    variant=spec['protocol']['variants'][spec['variant']]['identity']
    if variant not in ('exact','hash'):raise ValueError('unsupported identity variant')
    if initialize_rows:
        info=configure_memory(replay,module,vocabulary,variant=='exact',spec['seed'])
    else:
        mapping=IdentityMapping(replay.memory.hash_mapping,vocabulary,variant=='exact')
        replay.memory.hash_mapping=mapping
        heads=[n for order in mapping.vocab_size_across_layers[replay.memory.layer_id] for n in order]
        info={'addressing':'exact-canonical-key' if variant=='exact' else 'shared-hash-bucket',
              'head_sizes':heads,'physical_rows':sum(heads),'row_bytes':sum(heads)*32*4,
              'unique_keys_per_order':{str(n):len(vocabulary.keys[n]) for n in (2,3,4)}}
    for key in ('addressing','head_sizes','physical_rows','row_bytes','unique_keys_per_order'):
        if info[key]!=spec['identity'][key]:raise ValueError('identity layout changed: '+key)
    if spec.get('native_oov',False):
        if variant!='hash':raise ValueError('native OOV cannot change an exact-key model')
        replay.memory.hash_mapping.native_oov=True


def restore(run,base,upstream,*,device='cuda',native_oov=False):
    r=json.loads((run/'report.json').read_text());p=r['protocol'];v=p['variants'][r['variant']]
    if r['status']!='complete' or v['control']!='full' or sha(run/'delta.safetensors')!=r['delta']['sha256']:
        raise ValueError('a completed, hash-verified full-row source run is required')
    model,tok,replay,module,cfg,loading=setup(base,upstream,'full',v['vocab_sizes'],r['seed'],
        device=device,suffix_layers=p['suffix_layers'],train_decoder_suffix=p.get('train_decoder_suffix',True))
    if p.get('identity_control'):
        recipe=json.loads((run/'model-recipe.json').read_text())
        for key,value in (('protocol',p),('seed',r['seed']),('variant',r['variant']),
                          ('delta_sha256',r['delta']['sha256']),('identity',r['identity'])):
            if recipe[key]!=value:raise ValueError('source identity recipe changed: '+key)
        if recipe['vocabulary_sha256']!=r['identity']['vocabulary_sha256']:
            raise ValueError('source identity vocabulary changed')
        configure_identity(replay,module,run,{**recipe,'prepared_sha256':r['prepared_sha256'],'native_oov':native_oov})
    elif native_oov:
        raise ValueError('native OOV requires a trained identity hash control')
    state=load_file(str(run/'delta.safetensors'),device='cpu')
    expected={n:p for n,p in model.named_parameters() if p.requires_grad}
    if state.keys()!=expected.keys():raise ValueError('delta does not contain exactly the registered trained parameters')
    with torch.no_grad():
        for n,p in expected.items():
            x=state[n]
            if p.shape!=x.shape or p.dtype!=x.dtype:raise ValueError('delta shape/dtype changed: '+n)
            p.copy_(x)
    if replay.frozen_hash()!=r['frozen_weight_sha256']:raise ValueError('source immutable backbone changed')
    if 'frozen_decoder_suffix_sha256' in r and replay.frozen_suffix_hash()!=r['frozen_decoder_suffix_sha256']:
        raise ValueError('source immutable decoder suffix changed')
    module.set_global_step(r['optimizer_steps']);model.requires_grad_(False).eval()
    return r,model,replay,state


def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def restore_independent_control(run, upstream, *, device='cuda'):
    """Reopen trained constants/adapters without an unused large key registry."""
    from .counterfact_models import ConstantAddressMapping
    from .late_memory_replay import DenseMemoryAdapter
    run = Path(run); report = json.loads((run / 'report.json').read_text())
    protocol = report['protocol']; variant = protocol['variants'][report['variant']]
    control = variant['control']
    if (report['status'] != 'complete' or not protocol.get('independent_identity_controls')
            or control not in ('head-constant', 'adapter')
            or sha(run / 'delta.safetensors') != report['delta']['sha256']):
        raise ValueError('complete independently trained control required')
    model, tokenizer, replay, module, config, loading = setup(report['base_model'], upstream,
        'head-constant' if control == 'head-constant' else 'full', variant['vocab_sizes'], report['seed'],
        device=device, suffix_layers=protocol['suffix_layers'],
        train_decoder_suffix=protocol.get('train_decoder_suffix', True))
    try:
        if control == 'head-constant':
            replay.memory.hash_mapping = ConstantAddressMapping(replay.layer_id, replay.memory.multi_head_embedding.num_heads)
        else:
            branch = DenseMemoryAdapter(replay.text.config.hidden_size, report['identity']['target_parameter_budget']).to(replay.device)
            replay.memory = branch; model.ngram_memory = branch
        state = load_file(str(run / 'delta.safetensors'), device='cpu')
        expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if state.keys() != expected.keys(): raise ValueError('independent delta parameter domain changed')
        with torch.no_grad():
            for name, parameter in expected.items():
                value = state[name]
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError('independent trained parameter shape/dtype changed')
                parameter.copy_(value)
        if sum(p.numel() for p in expected.values()) != report['trainable_parameters']:
            raise ValueError('independent parameter count changed')
        if replay.frozen_hash() != report['frozen_weight_sha256']:
            raise ValueError('independent frozen backbone changed')
        if replay.frozen_suffix_hash() != report['frozen_decoder_suffix_sha256']:
            raise ValueError('independent frozen decoder changed')
        module.set_global_step(report['optimizer_steps']); model.requires_grad_(False).eval()
        return report, model, replay
    except BaseException:
        replay.close(); raise


def check_native_layout(staged,heads,dim,dtype=torch.float32):
    offsets=[sum(heads[:i]) for i in range(len(heads))]
    meta=staged.gather.metadata
    if (meta['shape']!=[sum(heads),dim] or meta['head_sizes']!=heads
            or meta['offsets']!=offsets or staged.dtype!=dtype):
        staged.close();raise ValueError('native local-head layout differs from configured branch')


def attach_native(model,replay,bundle,native_library,device):
    branch=replay.memory;source=branch.multi_head_embedding
    offsets=source.offsets.detach().cpu().tolist()+[len(source.embedding.weight)]
    heads=[b-a for a,b in zip(offsets,offsets[1:])]
    staged=StagedMultiHeadEmbedding(bundle/'memory',device=device,native_library=native_library,
        native_cache_bytes=1024**2,native_cache_policy='rows',staging_rows=4096,staging_slots=2)
    check_native_layout(staged,heads,source.embedding_dim,source.embedding.weight.dtype)
    branch.multi_head_embedding=staged;wrapped=NativeHashMemory(branch).eval()
    model.ngram_memory=wrapped;replay.memory=wrapped
    return staged


def reopen(bundle,upstream,native_library,*,device='cuda'):
    recipe=json.loads((bundle/'model-recipe.json').read_text());p=recipe['protocol'];v=p['variants'][recipe['variant']]
    if 'memory_manifest_sha256' in recipe and sha(bundle/'memory/embedding.json')!=recipe['memory_manifest_sha256']:
        raise ValueError('native memory manifest changed')
    path=bundle/recipe['nonrow_delta']
    if sha(path)!=recipe['nonrow_delta_sha256']:raise ValueError('nonrow delta changed')
    staged=None;replay=None
    def factory(list_of_N,D):
        nonlocal staged
        staged=StagedMultiHeadEmbedding(bundle/'memory',device=device,native_library=native_library,
            native_cache_bytes=1024**2,native_cache_policy='rows',staging_rows=4096,staging_slots=2)
        if len(list_of_N)!=staged.num_heads or D!=staged.embedding_dim:
            raise ValueError('native local-head layout differs from configured branch')
        return staged
    try:
        model,tok,replay,module,cfg,loading=setup(recipe['base_model'],upstream,'full',v['vocab_sizes'],recipe['seed'],
            device=device,suffix_layers=p['suffix_layers'],train_decoder_suffix=p.get('train_decoder_suffix',True),
            embedding_factory=factory)
        configure_identity(replay,module,bundle,recipe,initialize_rows=False)
        heads=[n for order in replay.memory.hash_mapping.vocab_size_across_layers[replay.memory.layer_id] for n in order]
        check_native_layout(staged,heads,32)
        state=load_file(str(path),device='cpu')
        expected={n:p for n,p in model.named_parameters() if p.requires_grad}
        if state.keys()!=expected.keys():raise ValueError('native bundle is missing nonrow parameters')
        with torch.no_grad():
            for n,p in expected.items():
                x=state[n]
                if p.shape!=x.shape or p.dtype!=x.dtype:raise ValueError('nonrow shape/dtype changed')
                p.copy_(x)
        if 'frozen_weight_sha256' in recipe and replay.frozen_hash()!=recipe['frozen_weight_sha256']:
            raise ValueError('native immutable backbone changed')
        if 'frozen_decoder_suffix_sha256' in recipe and replay.frozen_suffix_hash()!=recipe['frozen_decoder_suffix_sha256']:
            raise ValueError('native immutable decoder suffix changed')
        model.requires_grad_(False).eval();module.set_global_step(recipe['global_step'])
        wrapped=NativeHashMemory(replay.memory).eval();model.ngram_memory=wrapped;replay.memory=wrapped
        return model,replay,staged
    except BaseException:
        if staged is not None:staged.close()
        if replay is not None:replay.close()
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('run','base','upstream','prepared','output','native-library'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--requests',type=int,default=16)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--native-oov',action='store_true',help='Use ordinary open-vocabulary lookup for an identity hash checkpoint')
    args=parser.parse_args()
    if not 1<=args.requests<=64:raise ValueError('request check count must be between 1 and 64')
    if args.output.exists():raise ValueError('fresh output required')
    args.output.mkdir(parents=True);started=time.perf_counter();env=environment()
    r,model,replay,state=restore(args.run,args.base,args.upstream,device=args.device,native_oov=args.native_oov)
    prepared=json.loads((args.prepared/'prepare.json').read_text())
    if sha(args.prepared/'prepare.json')!=r['prepared_sha256'] or sha(args.prepared/'sequences.json')!=prepared['payload_sha256']:
        raise ValueError('source prepared requests changed')
    payload=json.loads((args.prepared/'sequences.json').read_text());seq=payload['sequences'];requests=[]
    for c in payload['cases']:
        if c['split']!='test':continue
        for g in ('rewrite','paraphrase','neighborhood'):
            for pair in c['groups'][g]:
                s=seq[pair['new']];ids=torch.tensor([s['ids'][:s['target_start']]],dtype=torch.int64)
                requests.append((c['case_id'],g,ids))
                if len(requests)==args.requests:break
            if len(requests)==args.requests:break
        if len(requests)==args.requests:break
    if len(requests)!=args.requests:raise ValueError('insufficient real request prefixes')
    references=[]
    with torch.no_grad():
        for _,_,ids in requests:references.append(replay.complete(ids,True,1).detach().cpu())
    branch=replay.memory;source=branch.multi_head_embedding;weight=source.embedding.weight
    offsets=source.offsets.cpu().tolist()+[len(weight)];heads=[b-a for a,b in zip(offsets,offsets[1:])]
    export_started=time.perf_counter()
    packed=export_multihead(weight,heads,args.output/'memory',block_rows=32,backend='native',deduplicate=False,predictor='xor')
    staged=attach_native(model,replay,args.output,args.native_library,args.device)
    try:
        for start in range(0,len(weight),256):
            end=min(start+256,len(weight))
            if staged.gather.table.rows(range(start,end))!=weight[start:end].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes():
                raise AssertionError('native original row bytes changed')
        report={'status':'checking-requests','environment':env,'source_run_report_sha256':sha(args.run/'report.json'),
            'source_delta_sha256':sha(args.run/'delta.safetensors'),'native_library_sha256':sha(args.native_library),
            'protocol_sha256':r['protocol_sha256'],'all_original_rows_byte_exact':True,'table':packed,
            'export_and_row_verification_seconds':time.perf_counter()-export_started,'requests':[],
            'limitations':['Interface/fidelity only, not memory necessity, comparison with editing methods, or serving superiority',
                'Resident and native modules coexist during validation; no comparative peak-memory or latency claim',
                'No KV cache, padding, distributed serving or generated-answer quality is measured here']}
        with torch.no_grad():
            for (case,g,ids),expected in zip(requests,references):
                actual=replay.complete(ids,True,1).detach().cpu()
                if digest(actual)!=digest(expected):raise AssertionError('native full model request changed next-token logit bytes')
                report['requests'].append({'case_id':case,'group':g,'input_tokens':ids.shape[1],
                    'input_ids_sha256':digest(ids),'next_token_logits_sha256':digest(actual),'byte_exact':True})
        row_name='ngram_memory.multi_head_embedding.embedding.weight'
        if row_name not in state:raise AssertionError('source row parameter missing')
        del state[row_name];save_file(state,str(args.output/'nonrow-delta.safetensors'))
        recipe={'base_model':str(args.base),'seed':r['seed'],'variant':r['variant'],'protocol':r['protocol'],
            'global_step':r['optimizer_steps'],'nonrow_delta':'nonrow-delta.safetensors','memory':'memory/embedding.json',
            'nonrow_delta_sha256':sha(args.output/'nonrow-delta.safetensors'),
            'memory_manifest_sha256':sha(args.output/'memory/embedding.json'),
            'frozen_weight_sha256':r['frozen_weight_sha256'],
            'contract':'Immutable original base plus nonrow delta and native original-ID rows; no original row delta needed'}
        if 'frozen_decoder_suffix_sha256' in r:recipe['frozen_decoder_suffix_sha256']=r['frozen_decoder_suffix_sha256']
        if r['protocol'].get('identity_control'):
            shutil.copyfile(args.run/'vocabulary.npz',args.output/'vocabulary.npz')
            if sha(args.output/'vocabulary.npz')!=r['identity']['vocabulary_sha256']:
                raise ValueError('identity vocabulary changed during export')
            recipe.update(vocabulary='vocabulary.npz',vocabulary_sha256=r['identity']['vocabulary_sha256'],
                identity=r['identity'],prepared_sha256=r['prepared_sha256'],native_oov=args.native_oov)
        write(args.output/'model-recipe.json',recipe)
        independent,independent_replay,independent_staged=reopen(args.output,args.upstream,args.native_library,device=args.device)
        try:
            with torch.no_grad():
                for (_,_,ids),expected in zip(requests,references):
                    if digest(independent_replay.complete(ids,True,1))!=digest(expected):
                        raise AssertionError('independently reopened native model changed logit bytes')
            report['independent_native_bundle_reopen_byte_exact']=True
        finally:independent_staged.close();independent_replay.close()
        report.update(status='passed',bundle_bytes=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()),
            source_delta_bytes=r['delta']['bytes'],staging_stats=staged.stats,elapsed_seconds=time.perf_counter()-started,
            execution_device=args.device,identity=r.get('identity'),native_oov=args.native_oov,
            native_reopen_constructs_dense_rows=False,
            startup_limitation='Native reopen creates no dense memory row parameter. Base loading, Python key dictionaries, file-backed metadata and allocator RSS remain separate; no total startup memory cap.')
        write(args.output/'report.json',report);print(json.dumps(report),flush=True)
    finally:staged.close();replay.close()


if __name__=='__main__':main()
