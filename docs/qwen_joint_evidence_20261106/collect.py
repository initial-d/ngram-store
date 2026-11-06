import pathlib,json,gzip,shutil,hashlib,subprocess
root=pathlib.Path('/llm-align/duyimin/ngram-qwen-joint-20261106')
out=root/'public-evidence'
out.mkdir()
(out/'training').mkdir();(out/'runtime').mkdir()
for name in ('environment.json','summary.json','joint-fixture-tests.log','run.sh','launcher.log','released-occupiers.jsonl','release-helper.log','experiment.exit-code','tokenizer-check.json','tokenizer-check.log','original-qwen.log','evaluate-original.py','verify-saved-tokenizers.py'):
 shutil.copy2(root/name,out/name)
shutil.copy2(root/'source/docs/qwen_joint_gpu_protocol_20261106.json',out/'protocol.json')
names=('train_tinyengram_joint.py','frozen_tinyengram_prefix.py','train_tinyengram_identity.py','tinyengram_checkpoint.py','lm_quality_metrics.py','analyze_tinyengram_joint.py','engram_row_adapter.py','device_row_adapter.py')
for name in names:shutil.copy2(root/'source/tools'/name,out/'runtime'/name)
shutil.copy2(root/'source/tests/test_tinyengram_checkpoint.py',out/'runtime/test_tinyengram_checkpoint.py')
inventory={}
for control in ('full','head-constant','off'):
 for seed in (3,7,11):
  name=control+'-seed'+str(seed);path=root/name;target=out/'training'/name;target.mkdir()
  r=json.loads((path/'report.json').read_text())
  assert r['status']=='passed'
  shutil.copy2(path/'report.json',target/'report.json')
  with (target/'quality.json.gz').open('wb') as f:
   with gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as z:z.write((path/'quality.json').read_bytes())
  for ext in ('.log','.exit-code'):shutil.copy2(root/(name+ext),out/'training'/(name+ext))
  inventory[name]={}
  for kind in ('checkpoint','bundle'):
   d=path/kind
   if not d.exists():continue
   files=[]
   for p in sorted(d.rglob('*')):
    if not p.is_file():continue
    h=hashlib.sha256()
    with p.open('rb') as f:
     for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    files.append({'path':str(p.relative_to(d)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
   inventory[name][kind]=files
with (out/'original-qwen.json.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as z:z.write((root/'original-qwen.json').read_bytes())
(out/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
print(out)
