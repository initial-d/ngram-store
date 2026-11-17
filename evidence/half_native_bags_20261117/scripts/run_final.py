"""Final branchless FP16/BF16 kernel on the exact already frozen full-table grid.

Inherits initial full input hashes; checks trace hashes and all source features
again. Preserves the earlier branching kernel's measurements as an ablation.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

WORK = Path('/llm-align/duyimin/ngram-half-bags-20261117')
PRIOR = Path('/llm-align/duyimin/ngram-sparse-xor-20261115')
COMMIT = 'd80f44c6b597a78d29215f12bb284175c1c94c45'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    results_dir = WORK / 'final-results'
    results_dir.mkdir(exist_ok=False)
    candidate = WORK / 'lib/candidate-final.so'
    source = WORK / 'final-runtime/native/meft_rows.c'
    flags = ['-I' + str(PRIOR / 'include'), '-DMEFT_USE_LZ4', '-DMEFT_USE_ZSTD',
        '-DMEFT_USE_LIBDEFLATE', '-l:liblz4.so.1', '-l:libzstd.so.1', '-l:libdeflate.so.0']
    command = ['cc', '-std=c11', '-O3', '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared',
        str(source), '-lz', *flags, '-fopt-info-vec-optimized=' + str(results_dir / 'vectorization.log'), '-o', str(candidate)]
    subprocess.run(command, check=True)
    manifest = {'commit': COMMIT, 'command': command, 'source_sha256': digest(source),
        'library_sha256': digest(candidate), 'compiler': subprocess.check_output(['cc', '--version'], text=True),
        'initial_build_manifest_sha256': digest(WORK / 'results/build.json'),
        'input_scope': 'Exactly the initial frozen tables/codes/traces; full hashes inherited, feature and trace checks repeated.'}
    (results_dir / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = dict(os.environ, PYTHONPATH=str(WORK / 'final-runtime'), CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    initial = json.loads((WORK / 'results/run-status.json').read_text())
    if any(r['returncode'] for r in initial):
        raise ValueError('initial run must pass before final input hashes are reused')
    jobs = []
    for entry in initial:
        command = entry['command'].copy()
        parent = Path(command[command.index('--output') + 1])
        command[command.index('--output') + 1] = str(results_dir / parent.name)
        command[command.index('--library') + 1] = str(candidate)
        command[command.index('--commit') + 1] = COMMIT
        command += ['--hashes-from', str(parent)]
        jobs.append((entry['label'], command))
    def run(job):
        label, command = job
        with (results_dir / (label + '.log')).open('w') as out:
            result = subprocess.run(command, cwd=WORK / 'final-runtime', env=env, stdout=out, stderr=subprocess.STDOUT)
        print('benchmarked-final', label, result.returncode, flush=True)
        return {'label': label, 'returncode': result.returncode, 'command': command}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(run, jobs))
    (results_dir / 'run-status.json').write_text(json.dumps(results, indent=2) + '\n')
    if any(r['returncode'] for r in results):
        raise RuntimeError('final benchmark failed; logs preserved')


if __name__ == '__main__':
    main()
