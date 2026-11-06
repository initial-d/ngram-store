"""Diagnostic unadapted reference; never used to select any trained model."""
import pathlib,json,hashlib,sys,time,platform,shlex
import numpy,torch,transformers
from transformers import AutoTokenizer,Qwen3ForCausalLM
from tools.train_tinyengram_identity import tensor_sha,heldout_windows,write_json
from tools.lm_quality_metrics import next_token_metrics,summarize_windows
from tools.tinyengram_checkpoint import source_revision
root=pathlib.Path('/llm-align/duyimin/ngram-qwen-joint-20261106')
base=pathlib.Path('/llm-align/open_models/Qwen3/Qwen3-0.6B')
protocol=json.loads((root/'source/docs/qwen_joint_gpu_protocol_20261106.json').read_text())
full=json.loads((root/'full-seed3/report.json').read_text())
started=time.perf_counter()
torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=Qwen3ForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval().to('cuda')
state=model.state_dict()
assert {n:tensor_sha(state[n]) for n in full['initial_decoder_sha256']}==full['initial_decoder_sha256']
h=hashlib.sha256()
for n,p in state.items():
 if n not in full['initial_decoder_sha256']:
  h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
assert h.hexdigest()==full['frozen_weight_sha256']
path=pathlib.Path('/llm-align/duyimin/meft-cuda-20261030/delta/corpus.txt')
assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['corpus_sha256']
text=path.read_text();a,b=int(len(text)*protocol['training_fraction']),int(len(text)*protocol['test_start_fraction'])
tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
encoded={s:tok(t,return_tensors='pt',add_special_tokens=False).input_ids for s,t in [('train',text[:a]),('calibration',text[a:b]),('test',text[b:])]}
assert {s:{'tokens':v.shape[1],'token_ids_sha256':tensor_sha(v)} for s,v in encoded.items()}==full['splits']
windows={s:heldout_windows(v,protocol['window_tokens'],protocol[s+'_windows'],protocol[s+'_window_start']) for s,v in encoded.items() if s!='train'}
result={}
with torch.no_grad():
 for split,samples in windows.items():
  result[split]=[]
  for i,ids in enumerate(samples):
   ids=ids.unsqueeze(0).to('cuda');output=model(ids,use_cache=False).logits
   m=next_token_metrics(output,output,ids);m['window_index']=i;result[split].append(m)
report={'schema_version':1,'status':'passed','source_sha':source_revision(),'protocol':protocol,'splits':full['splits'],
 'initial_decoder_sha256':full['initial_decoder_sha256'],'frozen_weight_sha256':h.hexdigest(),
 'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'command':shlex.join(sys.orig_argv),
 'windows':result,'summary':{s:summarize_windows(v) for s,v in result.items()},
 'environment':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__,
                'os':platform.platform(),'gpu':torch.cuda.get_device_name(),'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid)},
 'elapsed_seconds':time.perf_counter()-started,
 'limitations':['post-run diagnostic unadapted pretrained Qwen on identical windows; no selection or new qualification gate',
                'complete eager forwards; shared device is allowed; elapsed time is not an isolated serving benchmark',
                'pretraining overlap unknown; only contextualizes late-decoder adaptation, not n-gram memory value']}
write_json(root/'original-qwen.json',report)
print(report['summary'])
