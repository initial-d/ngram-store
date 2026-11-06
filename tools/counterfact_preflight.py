"""Real-weight replay gradients and unchanged MEMIT scoring oracle parity."""
import argparse
import ast
import typing
import time
import traceback

import numpy as np
import torch

from .counterfact_capacity import environment,sha,write
from .counterfact_models import setup
from .counterfact_data import load_records,encode_choice
from .late_memory_replay import check_masked_replay

ORACLE_SHA='b00d9810305580dddfeb5e65129db4d56bcbfee1edf020605e21bf4f5033aeb7'


def main():
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','upstream','data','oracle','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter();report={'status':'running','environment':environment(),'base':str(args.base)}
    try:
        if sha(args.oracle)!=ORACLE_SHA:raise ValueError('official MEMIT scorer source changed')
        model,tok,replay,module,cfg,loading=setup(args.base,args.upstream,'full',[257,521,1031],3)
        records=load_records(args.data);e=records[0]['requested_rewrite']
        sample=encode_choice(tok,e['prompt'].format(e['subject']),e['target_new']['str'],eos=True)
        ids=torch.tensor([sample['ids']])
        check=check_masked_replay(replay,ids,[sample['target_start']],logits_to_keep=len(sample['target_ids'])+1)
        parsed=ast.parse(args.oracle.read_text())
        node=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='test_batch_prediction')
        namespace={'torch':torch,'np':np,'typing':typing}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned MEMIT oracle>','exec'),namespace)
        class Wrapper(torch.nn.Module):
            def forward(self,input_ids,**kwargs):
                with replay.context(input_ids.cpu(),True):return model(input_ids=input_ids,**kwargs,use_cache=False)
        tok.padding_side='right';parity=[]
        with torch.no_grad():
            for r in records[:4]:
                e=r['requested_rewrite'];prefixes=[e['prompt'].format(e['subject']),*r['paraphrase_prompts'],*r['neighborhood_prompts']]
                correct=[0]*(1+len(r['paraphrase_prompts']))+[1]*len(r['neighborhood_prompts'])
                expected,accuracy=namespace['test_batch_prediction'](Wrapper(),tok,prefixes,correct,e['target_new']['str'],e['target_true']['str'])
                for i,prefix in enumerate(prefixes):
                    for choice in ('target_new','target_true'):
                        x=encode_choice(tok,prefix,e[choice]['str']);a=torch.tensor([x['ids']]);k=len(x['target_ids'])+1
                        logits=replay.complete(a,True,k);labels=torch.tensor(x['target_ids'],device='cuda')
                        values=-logits[0,:-1].float().log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
                        delta=abs(float(values.mean())-expected[i][choice])
                        if delta>1e-4:raise AssertionError(('official scorer mismatch',r['case_id'],choice,delta))
                        actual=bool(torch.equal(logits[0,:-1].argmax(-1),labels))
                        if (choice=='target_new')==(correct[i]==0) and actual!=bool(accuracy[i]):raise AssertionError('official target accuracy mismatch')
                        parity.append(delta)
        report.update(status='passed',loading=loading,masked_replay=check,official_oracle_max_nll_delta=max(parity),
            official_oracle_choices=len(parity),official_oracle_sha256=sha(args.oracle),
            physical_rows=len(replay.memory.multi_head_embedding.embedding.weight),frozen_weight_sha256=replay.frozen_hash(),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;write(args.output,report);print(report,flush=True)


if __name__=='__main__':main()
