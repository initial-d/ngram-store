#!/bin/bash
set -eu
TASK=${RUN_ROOT}
PY=${ENV_ROOT}/bin/python
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MEFT_SOURCE_COMMIT=b58c3fa
cd "$TASK/frontier-source"
"$PY" - <<'PYCODE'
import pathlib,json,subprocess,sys
root=pathlib.Path('${RUN_ROOT}')
model=root/'frontier/legacy-order-4-seed-3'
selected=['raw-fp32','native-none-fp32','tree-datrie-c0.99-n0.99-fp32','tree-reference-c0.99-n0.99-fp16','faiss-sq8','dedup-sq8','constant-mean']
for name in selected:
 d=json.loads((model/'jobs'/f'{name}.json').read_text())
 d['directory']=str(root/'preflight/bundles'/name);d['report']=str(root/'preflight/reports'/f'{name}.json')
 if name=='dedup-sq8':d['direct_directory']=str(root/'preflight/bundles/faiss-sq8')
 pathlib.Path(d['directory']).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(d['report']).parent.mkdir(parents=True,exist_ok=True)
 p=root/'preflight/jobs'/f'{name}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
 subprocess.run([sys.executable,'-m','tools.tree_node_quality','build','--job',str(p)],check=True)
 subprocess.run([sys.executable,'-m','tools.tree_node_quality','audit','--job',str(p),'--cached','0'],check=True)
 subprocess.run([sys.executable,'-m','tools.tree_node_quality','audit','--job',str(p),'--cached','1'],check=True)
 print(json.dumps({'preflight_passed':name}),flush=True)
PYCODE
printf '%s\n' complete > "$TASK/preflight-complete"
