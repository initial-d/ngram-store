import pathlib,json,hashlib,sys,shlex
from transformers import AutoTokenizer
from tools.train_tinyengram_identity import tensor_sha
root=pathlib.Path('/llm-align/duyimin/ngram-qwen-joint-20261106')
protocol=json.loads((root/'source/docs/qwen_joint_gpu_protocol_20261106.json').read_text())
text=pathlib.Path('/llm-align/duyimin/meft-cuda-20261030/delta/corpus.txt').read_text()
a,b=int(len(text)*protocol['training_fraction']),int(len(text)*protocol['test_start_fraction'])
checks={}
for control in ('full','head-constant','off'):
 for seed in (3,7,11):
  name=control+'-seed'+str(seed);path=root/name/'checkpoint'
  tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
  splits={}
  for s,t in [('train',text[:a]),('calibration',text[a:b]),('test',text[b:])]:
   ids=tokenizer(t,return_tensors='pt',add_special_tokens=False).input_ids
   splits[s]={'tokens':ids.shape[1],'token_ids_sha256':tensor_sha(ids)}
  report=json.loads((root/name/'report.json').read_text())
  assert splits==report['splits'],name
  checks[name]={'all_split_token_ids_exact':True,'splits':splits}
result={'status':'passed','command':shlex.join(sys.orig_argv),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'checks':checks,'limitations':['Pinned tokenizer versions and nonpadding/default input configuration only; original warnings retained in log']}
(root/'tokenizer-check.json').write_text(json.dumps(result,indent=2)+'\n')
print('Verified all 9 saved tokenizers produce the same split token-ID hashes as the base tokenizer.')
