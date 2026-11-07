"""Export a completed Qwen editing delta and verify real native model requests.

This checks the frozen-table boundary; a trained model passing this command
does not demonstrate memory necessity or a native serving speed advantage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file,save_file

from .counterfact_capacity import sha,write,environment
from .counterfact_models import setup
from .device_row_adapter import StagedMultiHeadEmbedding
from .engram_row_adapter import export_multihead


class NativeHashMemory(torch.nn.Module):
    """Let the unchanged Engram hash on CPU and return native rows on CUDA."""
    def __init__(self,source):super().__init__();self.source=source
    def forward(self,hidden_states,input_ids):
        return self.source(hidden_states=hidden_states,input_ids=input_ids.cpu())


def restore(run,base,upstream):
    r=json.loads((run/'report.json').read_text());p=r['protocol'];v=p['variants'][r['variant']]
    if r['status']!='complete' or v['control']!='full' or sha(run/'delta.safetensors')!=r['delta']['sha256']:
        raise ValueError('a completed, hash-verified full-row source run is required')
    model,tok,replay,module,cfg,loading=setup(base,upstream,'full',v['vocab_sizes'],r['seed'],
        suffix_layers=p['suffix_layers'],train_decoder_suffix=p.get('train_decoder_suffix',True))
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


def reopen(bundle,upstream,native_library):
    recipe=json.loads((bundle/'model-recipe.json').read_text());p=recipe['protocol'];v=p['variants'][recipe['variant']]
    model,tok,replay,module,cfg,loading=setup(recipe['base_model'],upstream,'full',v['vocab_sizes'],recipe['seed'],
        suffix_layers=p['suffix_layers'],train_decoder_suffix=p.get('train_decoder_suffix',True))
    path=bundle/recipe['nonrow_delta']
    if sha(path)!=recipe['nonrow_delta_sha256']:raise ValueError('nonrow delta changed')
    state=load_file(str(path),device='cpu')
    expected={n:p for n,p in model.named_parameters() if p.requires_grad and n!='ngram_memory.multi_head_embedding.embedding.weight'}
    if state.keys()!=expected.keys():raise ValueError('native bundle is missing nonrow parameters')
    with torch.no_grad():
        for n,p in expected.items():
            x=state[n]
            if p.shape!=x.shape or p.dtype!=x.dtype:raise ValueError('nonrow shape/dtype changed')
            p.copy_(x)
    model.requires_grad_(False).eval();module.set_global_step(recipe['global_step'])
    staged=StagedMultiHeadEmbedding(bundle/'memory',device='cuda',native_library=native_library,
        native_cache_bytes=1024**2,native_cache_policy='rows',staging_rows=4096,staging_slots=2)
    branch=replay.memory
    if (len(branch.multi_head_embedding.embedding.weight),branch.multi_head_embedding.embedding_dim)!=tuple(staged.gather.metadata['shape']):
        staged.close();raise ValueError('native row shape differs from configured branch')
    branch.multi_head_embedding=staged;wrapped=NativeHashMemory(branch);model.ngram_memory=wrapped;replay.memory=wrapped
    return model,replay,staged


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('run','base','upstream','prepared','output','native-library'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--requests',type=int,default=16);args=parser.parse_args()
    if not 1<=args.requests<=64:raise ValueError('request check count must be between 1 and 64')
    if args.output.exists():raise ValueError('fresh output required')
    args.output.mkdir(parents=True);started=time.perf_counter();env=environment()
    r,model,replay,state=restore(args.run,args.base,args.upstream)
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
    staged=StagedMultiHeadEmbedding(args.output/'memory',device='cuda',native_library=args.native_library,
        native_cache_bytes=1024**2,native_cache_policy='rows',staging_rows=4096,staging_slots=2)
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
        branch.multi_head_embedding=staged;wrapped=NativeHashMemory(branch)
        model.ngram_memory=wrapped;replay.memory=wrapped
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
            'contract':'Immutable original base plus nonrow delta and native original-ID rows; no original row delta needed'}
        write(args.output/'model-recipe.json',recipe)
        independent,independent_replay,independent_staged=reopen(args.output,args.upstream,args.native_library)
        try:
            with torch.no_grad():
                for (_,_,ids),expected in zip(requests,references):
                    if digest(independent_replay.complete(ids,True,1))!=digest(expected):
                        raise AssertionError('independently reopened native model changed logit bytes')
            report['independent_native_bundle_reopen_byte_exact']=True
        finally:independent_staged.close();independent_replay.close()
        report.update(status='passed',bundle_bytes=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()),
            source_delta_bytes=r['delta']['bytes'],staging_stats=staged.stats,elapsed_seconds=time.perf_counter()-started,
            startup_limitation='Current reopen constructs a dense temporary row module before replacing it; no bounded native-startup claim')
        write(args.output/'report.json',report);print(json.dumps(report),flush=True)
    finally:staged.close();replay.close()


if __name__=='__main__':main()
