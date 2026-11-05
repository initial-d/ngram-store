"""Export the frozen real news task to complete same-source backend bundles."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .budgeted_bags import build_error_bounds
from .certified_store import compact_native
from .cotraining_groups import BagPartition
from .external_cotraining import file_hash
from .faiss_row_codec import fit_codec
from .quantized_memory import pack_quantized_file
from .text_classifier import _RawRows, reduce_words
from .train_news_classifier import CLASSES, write_json


BACKENDS = ['source-resident','source-mmap','fp16','int8','faiss-sq4','faiss-sq8',
            'faiss-pq-8x8','faiss-cluster-8','ngram-bounded','ngram-int8','precomputed-words']


def copy_file(source, target):
    # Each directory is standalone; count its complete logical bytes even if
    # local experiment files share an inode. Exported bundles have no fallback.
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def finalize(root, backend, data, training, started):
    metadata = {'schema_version':1,'kind':'ngram-linear-text','backend':backend,
        'classes':training.get('classes',CLASSES),'rows':data['rows'],'dim':data['dim'],
        'trained_source_sha256':training['source_sha256'],'source_commit':training['source_commit'],
        'head_sha256':training['head_sha256'],'seed':training['seed'],'selected_epoch':training['selected_epoch'],
        'source_row_byte_preserving':backend.startswith('source-'),
        'prediction_preservation_certificate':backend == 'ngram-bounded',
        'contract':'Frozen supervised reference; all requests predicted; source-real-model preservation flags only for bounded FP32.',
        'build_seconds':time.perf_counter()-started,
        'files_sha256':{str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    write_json(root/'classifier.json',metadata)
    return {'backend':backend,'complete_model_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
            'build_seconds':metadata['build_seconds'],'classifier_sha256':file_hash(root/'classifier.json'),
            'files_sha256':metadata['files_sha256']}


def export_seed(common, checkpoint, output, *, library=None, backends=BACKENDS):
    data = json.loads((common/'data.json').read_text())
    training = json.loads((checkpoint/'training.json').read_text())
    if training['status'] != 'passed' or file_hash(checkpoint/'source.f32') != training['source_sha256']:
        raise ValueError('passed frozen trained source required')
    if file_hash(checkpoint/'head.npz') != training['head_sha256']:
        raise ValueError('source head changed')
    output.mkdir(parents=True,exist_ok=False)
    protected = output/'protected.u64'
    np.arange(data['whole_word_rows'],dtype='<u8').tofile(protected)
    source = np.memmap(checkpoint/'source.f32',mode='r',dtype='<f4',shape=(data['rows'],data['dim']))
    # Freeze actual codebook training rows before looking at validation metrics.
    rng = np.random.default_rng(42)
    sample = np.sort(rng.choice(len(source),size=min(32768,len(source)),replace=False))
    train_rows = np.asarray(source[sample],'<f4')
    lexicon = json.loads((common/'lexicon.json').read_text())
    report = {'status':'running','seed':training['seed'],'source_sha256':training['source_sha256'],
              'codebook_training_ids_sha256':__import__('hashlib').sha256(sample.astype('<u8').tobytes()).hexdigest(),
              'variants':[]}
    bounded = None
    for backend in backends:
        started = time.perf_counter()
        root = output/backend
        root.mkdir()
        copy_file(checkpoint/'head.npz',root/'head.npz')
        if backend == 'precomputed-words':
            slim = {**lexicon,'grams':[],'whole_words':[],'whole_word_rows':0}
            write_json(root/'lexicon.json',slim)
            offsets = np.memmap(common/'word-offsets.u64',mode='r',dtype='<u8')
            ids = np.memmap(common/'word-ids.u64',mode='r',dtype='<u8')
            words = np.memmap(root/'word-vectors.f32',mode='w+',dtype='<f4',shape=(len(offsets)-1,data['dim']))
            reader = _RawRows(checkpoint/'source.f32',source.shape)
            for start in range(0,len(words),1024):
                end = min(len(words),start+1024)
                bags = [ids[int(offsets[i]):int(offsets[i+1])] for i in range(start,end)]
                words[start:end] = reduce_words(reader,bags,data['dim'])
            words.flush(); words._mmap.close(); reader.close(); offsets._mmap.close(); ids._mmap.close()
        else:
            for name in ('lexicon.json','word-ids.u64','word-offsets.u64'):
                copy_file(common/name,root/name)
            if backend.startswith('source-'):
                copy_file(checkpoint/'source.f32',root/'source.f32')
            elif backend in ('fp16','int8'):
                pack_quantized_file(checkpoint/'source.f32',root/'memory',data['dim'],backend,native=False)
            elif backend.startswith('faiss-'):
                fit_codec(root/'memory',source,backend[len('faiss-'):],train_rows,seed=42,iterations=20)
            elif backend == 'ngram-bounded':
                compact_native(checkpoint/'source.f32',data['dim'],common/'word-ids.u64',common/'word-offsets.u64',
                               root/'memory',protected_ids=protected if data['whole_word_rows'] else None,block_rows=8)
                build_error_bounds(checkpoint/'source.f32',root/'memory',root/'bounds',native_library=library)
                bounded = root
            elif backend == 'ngram-int8':
                partition = BagPartition.load(bounded/'memory/domain.npz')
                means = partition.means(source)
                staging = output/'projected.f32'
                with staging.open('wb') as stream:
                    for start in range(0,len(source),4096):
                        stream.write(means[partition.mapping[start:start+4096]].astype('<f4').tobytes())
                pack_quantized_file(staging,root/'memory',data['dim'],'int8',native=True,
                    representatives=partition.representatives().tolist(),deduplicate=False,max_block_bytes=8192)
                staging.unlink()
        report['variants'].append(finalize(root,backend,data,training,started))
        write_json(output/'export.json',report)
        print(json.dumps({'seed':training['seed'],'backend':backend,
            'complete_model_bytes':report['variants'][-1]['complete_model_bytes'],
            'build_seconds':report['variants'][-1]['build_seconds']}),flush=True)
    source._mmap.close(); protected.unlink()
    report['status'] = 'passed'
    write_json(output/'export.json',report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--native-library',type=Path)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU-only export requires empty CUDA_VISIBLE_DEVICES')
    import faiss
    faiss.omp_set_num_threads(1)
    export_seed(cfg.training/'common',cfg.training/f'seed-{cfg.seed}',cfg.output,native_library=cfg.native_library)


if __name__ == '__main__':
    main()
