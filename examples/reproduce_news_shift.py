"""Test frozen AG News models on independent, naturally newer public news.

Requires three existing source training checkpoints and their complete exports.
An explicit download occurs only if --huffpost is omitted. No GPU or training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--existing-serving',type=Path,required=True,help='Parent of seed-3, seed-7 and seed-11 directories.')
    p.add_argument('--ag-data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--huffpost',type=Path)
    cfg=p.parse_args()
    cfg.output=cfg.output.resolve();cfg.output.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
    def run(args):
        subprocess.run(args,cwd=ROOT,env=env,check=True)
    # Importing only NumPy/C-side tools; no Torch required for this follow-up.
    sys.path.insert(0,str(ROOT))
    from tools.benchmark_news_shift import HUFFPOST_URL,HUFFPOST_SHA256
    huffpost=cfg.huffpost.resolve() if cfg.huffpost else cfg.output/'huffpost.jsonl'
    if not cfg.huffpost:
        digest=hashlib.sha256()
        with urllib.request.urlopen(HUFFPOST_URL,timeout=120) as response,huffpost.open('wb') as stream:
            while chunk:=response.read(1<<20):
                stream.write(chunk);digest.update(chunk)
        if digest.hexdigest()!=HUFFPOST_SHA256:
            raise ValueError('pinned independent news data changed')
    run(['make','native'])
    library=ROOT/'build'/('libmeft_rows.dylib' if sys.platform=='darwin' else 'libmeft_rows.so')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    requests=cfg.output/'requests'
    run([sys.executable,'-m','tools.benchmark_news_shift','prepare','--huffpost',str(huffpost),
        '--ag-train',str(cfg.ag_data.resolve()/'train.csv'),'--training',str(cfg.training.resolve()),'--output',str(requests)])
    for seed in (3,7,11):
        hybrids=cfg.output/'hybrids'/f'seed-{seed}'
        existing=cfg.existing_serving.resolve()/f'seed-{seed}'
        run([sys.executable,'-m','tools.export_news_hybrids','--training',str(cfg.training.resolve()),
            '--existing-serving',str(existing),'--output',str(hybrids),'--seed',str(seed)])
        run([sys.executable,'-m','tools.benchmark_news_shift','run','--serving',str(existing),
            '--hybrids',str(hybrids),'--requests',str(requests),'--output',str(cfg.output/'results'/f'seed-{seed}'),
            '--native-library',str(library),'--source-commit',commit,'--seed',str(seed)])
    run([sys.executable,'-m','tools.bootstrap_news_shift','--results',str(cfg.output/'results'),
        '--output',str(cfg.output/'bootstrap.json')])
    run([sys.executable,'-m','tools.analyze_news_shift','--results',str(cfg.output/'results'),
        '--bootstrap',str(cfg.output/'bootstrap.json'),'--output',str(cfg.output/'summary.json')])
    from tools.export_news_hybrids import AUDIT_HYBRIDS
    from tools.analyze_news_hybrid_audit import summarize as summarize_audit
    primary, audits = [], []
    for seed in (3,7,11):
        hybrids=cfg.output/'audit-hybrids'/f'seed-{seed}'
        report=cfg.output/'results'/f'seed-{seed}'/'report.json'
        output=cfg.output/'audit-results'/f'seed-{seed}'
        run([sys.executable,'-m','tools.export_news_hybrids','--training',str(cfg.training.resolve()),
            '--existing-serving',str(cfg.existing_serving.resolve()/f'seed-{seed}'),
            '--output',str(hybrids),'--seed',str(seed),'--backends',','.join(AUDIT_HYBRIDS)])
        run([sys.executable,'-m','tools.benchmark_news_hybrid_audit','--primary',str(report),
            '--hybrids',str(hybrids),'--requests',str(requests),'--output',str(output),
            '--native-library',str(library),'--source-commit',commit])
        primary.append(json.loads(report.read_text()))
        audits.append(json.loads((output/'report.json').read_text()))
    (cfg.output/'audit-summary.json').write_text(json.dumps(summarize_audit(primary,audits),indent=2)+'\n')
    print('Complete independent evaluation:',cfg.output/'summary.json')
    print('Article text and model weights stay in this output or their original training directory.')


if __name__=='__main__':
    main()
