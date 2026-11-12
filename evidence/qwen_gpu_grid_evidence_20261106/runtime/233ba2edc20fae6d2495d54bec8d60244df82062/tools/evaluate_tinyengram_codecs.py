"""Same-source frozen-row codecs on actual trained TinyEngram tables.

This is a quality oracle: candidate rows are reconstructed into a dense GPU
table to compare the SAME decoder schedule. It makes no serving-memory claim.
Codec fitting uses only rows queried by training windows; no test selection.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys
import time

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from .engram_row_adapter import file_sha256
from .faiss_row_codec import FaissRows, fit_codec
from .frozen_tinyengram_prefix import FrozenTinyEngramPrefix
from .lm_quality_metrics import next_token_metrics, summarize_windows
from .quantized_memory import QuantizedRows, pack_quantized
from .tinyengram_checkpoint import load_upstream, source_revision
from .train_tinyengram_identity import sample_windows, tensor_sha, write_json

CODECS = ('fp16', 'int8', 'sq8', 'sq4', 'pq-8x8', 'cluster-8', 'opq-8x8')


def close_reader(reader):
    if isinstance(reader, QuantizedRows):
        reader.close()
    else:
        reader.codes._mmap.close()


def read_rows(reader, ids):
    return reader.gather(ids) if isinstance(reader, QuantizedRows) else reader.rows(ids)


def decode_complete(reader, shape):
    result = np.empty(shape, dtype='<f4')
    for start in range(0, shape[0], 4096):
        stop = min(start+4096, shape[0])
        result[start:stop] = read_rows(reader, np.arange(start,stop))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training-run','seed-directory','prefix-root','corpus','upstream-dir','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--native-library',type=Path,required=True)
    p.add_argument('--codecs',default=','.join(CODECS))
    args = p.parse_args()
    names = args.codecs.split(',')
    if args.output_dir.exists() or not set(names) <= set(CODECS) or len(names)!=len(set(names)):
        p.error('fresh output and distinct supported codecs required')
    args.output_dir.mkdir(parents=True)
    train = json.loads(args.training_run.read_text())
    protocol = train['protocol']
    seed = int(args.seed_directory.name.split('-')[-1])
    source = train['seeds'][str(seed)]
    if train['status'] != 'passed' or source['status'] != 'passed':
        raise ValueError('complete successful training artifact required (quality qualification may fail)')
    if file_sha256(args.corpus) != protocol['corpus_sha256']:
        raise ValueError('corpus changed')
    report = {'status':'running','code_sha':source_revision(),'command':shlex.join(sys.orig_argv),
              'training_report_sha256':file_sha256(args.training_run), 'seed':seed,
              'protocol':protocol,'codecs':{},'calibration_quality_budgets':[0.001,0.01],
              'quality_gate':'upper paired-window 95% relative PPL bound <= budget; select smallest backend+codec bytes on calibration only',
              'environment':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,
                             'numpy':np.__version__,'os':platform.platform()},
              'limitations':['quality oracle uses dense reconstructed candidate rows and cached frozen prefix; no serving resource claim',
                             'codec fit uses sampled actually trained rows, encodes every row including unused initialization',
                             'models failing memory-off qualification cannot support necessary-backend claims',
                             'positive and negative quantization PPL changes are both retained; no compression retraining',
                             'one fixed corpus, frozen backbone and late branch; not released DeepSeek trained weights']}
    start = time.perf_counter()
    model = None
    try:
        import faiss
        faiss.omp_set_num_threads(8)
        torch.set_num_threads(8); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        report['environment'].update(faiss=faiss.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_properties(0).name,gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),tf32=False)
        checkpoint=args.seed_directory/'checkpoint'
        engram,qwen=load_upstream(args.upstream_dir)
        config=json.loads((checkpoint/'engram_config.json').read_text())
        config['tokenizer_name_or_path']=str(checkpoint)
        engram.set_global_step(source['steps'])
        model,loading=qwen.EngramQwenForCausalLM.from_pretrained(checkpoint,
            engram_cfg=engram.EngramConfig(**config),local_files_only=True,torch_dtype=torch.float32,
            attn_implementation='eager',output_loading_info=True)
        if any(loading.values()):
            raise ValueError('incomplete checkpoint restore: '+repr(loading))
        model.requires_grad_(False).cuda().eval()
        reuse=FrozenTinyEngramPrefix(model)
        if reuse.frozen_weight_sha256()!=source['frozen_weight_sha256']:
            raise ValueError('frozen backbone changed')
        weight=reuse.branch.multi_head_embedding.embedding.weight
        if tensor_sha(weight)!=source['memory']['trained_weight_sha256']:
            raise ValueError('trained table changed')
        original=weight.detach().clone()
        rows=original.cpu().numpy()
        report['memory']=source['memory']
        tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
        text=args.corpus.read_text()
        a,b=int(len(text)*protocol['training_fraction']),int(len(text)*protocol['test_start_fraction'])
        tokens={s:tokenizer(t,return_tensors='pt',add_special_tokens=False).input_ids
                for s,t in [('train',text[:a]),('calibration',text[a:b]),('test',text[b:])]}
        if any(tensor_sha(ids)!=train['splits'][s]['token_ids_sha256'] for s,ids in tokens.items()):
            raise ValueError('tokenized split changed')
        training,_=sample_windows(tokens['train'],protocol['window_tokens'],protocol['training_windows'])
        queried=[]
        for chunk in training.split(64):
            ids=torch.from_numpy(reuse.branch.hash_mapping.hash(chunk.numpy())[reuse.layer_id])
            queried.append((ids+reuse.branch.multi_head_embedding.offsets.cpu()).flatten().numpy())
        fit_ids=np.unique(np.concatenate(queried))
        fit_ids=np.sort(np.random.default_rng(seed).choice(fit_ids,min(16384,len(fit_ids)),replace=False))
        report['fit_original_ids_sha256']=hashlib.sha256(fit_ids.astype('<i8').tobytes()).hexdigest()
        length=protocol['window_tokens']
        windows={s:ids[0,length*protocol.get(s+'_window_start',0):length*(protocol.get(s+'_window_start',0)+protocol[s+'_windows'])].reshape(-1,length)
                 for s,ids in tokens.items() if s!='train'}
        prefixes={}
        for s,ids in windows.items():
            path=args.prefix_root/(s+'-prefix.npy')
            if file_sha256(path)!=train['prefix_files'][s]['sha256']:
                raise ValueError('prefix bytes changed')
            prefixes[s]=np.load(path,mmap_mode='r',allow_pickle=False)
        for name in names:
            directory=args.output_dir/name
            readers=[]
            try:
                if name in ('fp16','int8'):
                    pack_quantized(rows,directory/'raw',name,native=False)
                    pack_quantized(rows,directory/'native',name,native=True,max_block_bytes=8192)
                    readers=[QuantizedRows(directory/'raw',args.native_library),QuantizedRows(directory/'native',args.native_library)]
                    candidate=decode_complete(readers[0],rows.shape)
                    if candidate.tobytes()!=decode_complete(readers[1],rows.shape).tobytes():
                        raise AssertionError('native encoding changed quantized decoded row bytes')
                    storage={k:sum(f.stat().st_size for f in (directory/k).iterdir() if f.is_file()) for k in ('raw','native')}
                    metadata={k:r.metadata for k,r in zip(('raw','native'),readers)}
                else:
                    metadata=fit_codec(directory,rows,name,rows[fit_ids],seed=seed,iterations=20,opq_iterations=10)
                    readers=[FaissRows(directory)]
                    candidate=decode_complete(readers[0],rows.shape)
                    storage={'faiss':sum(f.stat().st_size for f in directory.iterdir() if f.is_file())}
                item={'storage_bytes':storage,'codec_metadata':metadata,'decoded_rows_sha256':hashlib.sha256(candidate.tobytes()).hexdigest(),
                      'native_matches_raw_decoded_bytes':True if name in ('fp16','int8') else None}
                decoded=torch.from_numpy(candidate).cuda()
                metrics={s:[] for s in windows}
                with torch.no_grad():
                    for split,samples in windows.items():
                        for i,sample in enumerate(samples):
                            sample=sample.unsqueeze(0)
                            hidden=torch.from_numpy(np.array(prefixes[split][i:i+1],copy=True)).cuda()
                            weight.copy_(original)
                            reference=reuse.forward(sample,hidden)
                            if i==0 and not torch.equal(model(sample.cuda(),use_cache=False).logits.view(torch.uint8),reference.contiguous().view(torch.uint8)):
                                raise AssertionError('codec source prefix differs from complete forward')
                            weight.copy_(decoded)
                            output=reuse.forward(sample,hidden)
                            m=next_token_metrics(reference,output,sample.cuda()); m['window_index']=i
                            metrics[split].append(m)
                            del reference,output,hidden
                item['summary']={s:summarize_windows(m) for s,m in metrics.items()}
                target=args.output_dir/(name+'-windows.json.gz')
                with gzip.open(target,'wt') as stream:
                    json.dump(metrics,stream,allow_nan=False)
                item['windows_sha256']=file_sha256(target)
                report['codecs'][name]=item
                write_json(args.output_dir/'report.json',report)
                print(name,{s:round(m['relative_ppl_change'],6) for s,m in item['summary'].items()},flush=True)
                del candidate,decoded
            finally:
                weight.copy_(original)
                for reader in readers: close_reader(reader)
        report['selection']={}
        for budget in report['calibration_quality_budgets']:
            candidates=[(size,name,backend) for name,item in report['codecs'].items()
                        if item['summary']['calibration']['bootstrap']['relative_ppl_change_95ci'][1]<=budget
                        for backend,size in item['storage_bytes'].items()]
            chosen=min(candidates) if candidates else None
            report['selection'][str(budget)]={'choice':chosen,'test_confirms':None if chosen is None else
                report['codecs'][chosen[1]]['summary']['test']['bootstrap']['relative_ppl_change_95ci'][1]<=budget}
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=repr(error)); raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        write_json(args.output_dir/'report.json',report)


if __name__=='__main__':
    main()
