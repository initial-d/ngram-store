"""Real distinct-request Qwen cache correctness; shared GPUs are allowed.

This checker measures no latency or throughput. It compares the actual trained
GPU source with an original-ID byte-preserving native reader under the same
complete cached generation schedule.
"""
import argparse
import gc
from pathlib import Path

import torch

from .benchmark_llm_codec_serving import batches_for
from .cached_memory_decode import generate_cached
from .counterfact_native import restore, digest as tensor_digest
from .llm_codec_frontier import reopen_codec
from .tree_node_quality import digest, load, write


def check(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    common=args.model/'common';metadata=load(common/'metadata.json');recipe=load(common/'model-recipe.json')
    job=load(args.model/'jobs/native-none-fp32.json')
    if load(job['report'])['decoded_row_sha256']!=metadata['source_row_sha256']:
        raise ValueError('Source-byte-preserving native candidate required')
    schedules={batch:batches_for(common,'test',args.requests,batch) for batch in (1,8)}
    reference={};results=[]
    report={'status':'running','purpose':'Shared-device interface correctness only; no performance result',
            'source_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'checker_source_sha256':digest(__file__),
            'cached_decode_source_sha256':digest(Path(__file__).with_name('cached_memory_decode.py')),
            'codec_adapter_source_sha256':digest(Path(__file__).with_name('codec_memory_adapter.py')),
            'model':args.model.name,'source_checkpoint_sha256':metadata['source_checkpoint_sha256'],
            'decoded_row_sha256':metadata['source_row_sha256'],'device':args.device,'checks':results}
    try:
        for mode in ('source','native'):
            staged=None
            if mode=='source':
                _,model,replay,state=restore(Path(metadata['source_run']),Path(recipe['base_model']),args.upstream,device=args.device)
                del state
            else:model,replay,staged=reopen_codec(job,args.upstream,device=args.device)
            try:
                with torch.no_grad():
                    for batch,requests in schedules.items():
                        for index,request in enumerate(requests):
                            ids=torch.tensor([row['ids'] for row in request],dtype=torch.int64)
                            generated,values=generate_cached(replay,ids,steps=16,global_step=recipe['global_step'])
                            direct=replay.complete(ids,True,1).cpu()
                            actual={'input_ids_sha256':tensor_digest(ids),'batch':batch,'prefix_tokens':ids.shape[1],
                                    'generated_ids_sha256':tensor_digest(generated),'cached_logits_sha256':[tensor_digest(v) for v in values],
                                    'cached_vs_full_prefill_max_absolute':float((direct-values[0]).abs().max()),
                                    'cached_vs_full_prefill_argmax_changes':int((direct.argmax(-1)!=values[0].argmax(-1)).sum())}
                            key=(batch,index)
                            if mode=='source':reference[key]=actual
                            else:
                                for field in ('input_ids_sha256','generated_ids_sha256','cached_logits_sha256'):
                                    if actual[field]!=reference[key][field]:raise AssertionError('Actual source/native batch differs: '+field)
                                results.append({'batch':batch,'index':index,'source':reference[key],'native':actual,'byte_exact_source_native':True})
                                write(args.output,report)
            finally:
                if staged is not None:staged.close()
                replay.close();del model,replay;gc.collect()
                if args.device=='cuda':torch.cuda.empty_cache()
        report.update(status='passed',matched_complete_cached_logits=sum(len(r['source']['cached_logits_sha256']) for r in results),
                      matched_generated_sequences=sum(r['batch'] for r in results))
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:write(args.output,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','upstream','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--requests',type=int,default=2)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    if args.requests<1 or args.output.exists():parser.error('Fresh output and positive request count required')
    check(args)


if __name__=='__main__':main()
