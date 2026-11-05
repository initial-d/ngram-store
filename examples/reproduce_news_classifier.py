"""Download AG News and build a usable CPU n-gram news classifier.

Run from a checkout with NumPy/PyTorch and a C compiler installed. This explicit
command downloads research data; ordinary classification never fetches anything.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/555590db4219b1243abb1918effd6a7425a2d75f/data/ag_news_csv/'
HASHES = {'train.csv':'76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e',
          'test.csv':'521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699',
          'readme.txt':'779c2e6bbf93dd810d9c7b318006daef7a4bfc34062002461d251dc10fb9f58c'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=3)
    p.add_argument('--full-controls',action='store_true',help='Also build FAISS/quantization controls; requires faiss-cpu.')
    cfg = p.parse_args()
    cfg.output = cfg.output.resolve()
    cfg.output.mkdir(parents=True,exist_ok=False)
    data = cfg.output/'data'
    data.mkdir()
    for name,expected in HASHES.items():
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL+name,timeout=120) as response, (data/name).open('wb') as stream:
            while block := response.read(1<<20):
                stream.write(block);digest.update(block)
        if digest.hexdigest()!=expected:
            raise ValueError('public data hash changed: '+name)
    env = {**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    def run(args):
        subprocess.run(args,cwd=ROOT,env=env,check=True)
    run(['make','native'])
    # Mirrors the built-in source/wheel library resolution on Linux/macOS.
    library = ROOT/'build'/('libmeft_rows.dylib' if sys.platform=='darwin' else 'libmeft_rows.so')
    env['MEFT_ROWS_LIBRARY'] = str(library)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    training = cfg.output/'training'
    run([sys.executable,'-m','tools.train_news_classifier','--data',str(data),'--output',str(training),
         '--source-commit',commit,'--seeds',str(cfg.seed),'--activation','relu'])
    serving = cfg.output/'serving'
    command = [sys.executable,'-m','tools.export_news_classifier','--training',str(training),'--output',str(serving),
               '--seed',str(cfg.seed),'--native-library',str(library)]
    if not cfg.full_controls:
        command.extend(['--backends','source-mmap,int8,ngram-bounded,ngram-int8,precomputed-words'])
    run(command)
    requests = cfg.output/'requests'
    run([sys.executable,'-m','tools.benchmark_news_classifier','prepare','--data',str(data),
         '--training',str(training),'--output',str(requests)])
    results = cfg.output/'results'
    run([sys.executable,'-m','tools.benchmark_news_classifier','run','--serving',str(serving),
         '--requests',str(requests),'--output',str(results),'--seed',str(cfg.seed),
         '--source-commit',commit,'--native-library',str(library)])
    run([sys.executable,'-m','ngram_store','classify','--store',str(serving/'ngram-int8'),
         '--text','The team won the final championship match.','Scientists announced a new computer chip.'])
    print('Classifier bundle:',serving/'ngram-int8')
    print('Validation-first complete test report:',results/'report.json')
    print('Data and weights stay in this output directory; source code uses the pinned research dataset.')


if __name__=='__main__':main()
