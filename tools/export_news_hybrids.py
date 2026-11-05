"""Precompute source training words, retaining a complete unknown-word backend.

The four controls have identical known-word FP32 features. Only their n-gram
fallback differs. No evaluation words, input text or labels enter this export.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .cotraining_groups import BagPartition
from .export_news_classifier import copy_file, finalize
from .external_cotraining import file_hash
from .quantized_memory import pack_quantized_file
from .train_news_classifier import write_json


HYBRIDS = ('precomputed-hybrid-source', 'precomputed-hybrid-int8',
           'precomputed-hybrid-faiss-sq4', 'precomputed-hybrid-ngram-int8')


def export_hybrids(common, checkpoint, existing, output, backends=HYBRIDS):
    common, checkpoint, existing, output = map(Path, (common, checkpoint, existing, output))
    if not backends or any(name not in HYBRIDS for name in backends):
        raise ValueError('nonempty known hybrid backends required')
    data = json.loads((common/'data.json').read_text())
    training = json.loads((checkpoint/'training.json').read_text())
    if training['status'] != 'passed' or file_hash(checkpoint/'source.f32') != training['source_sha256']:
        raise ValueError('unchanged passed source checkpoint required')
    if file_hash(checkpoint/'head.npz') != training['head_sha256']:
        raise ValueError('source head changed')
    precomputed = existing/'precomputed-words'
    base = json.loads((precomputed/'classifier.json').read_text())
    if base['trained_source_sha256'] != training['source_sha256']:
        raise ValueError('precomputed words use a different trained source')
    for name, digest in base['files_sha256'].items():
        if file_hash(precomputed/name) != digest:
            raise ValueError('precomputed source file changed')
    output.mkdir(parents=True, exist_ok=False)
    lexicon = json.loads((common/'lexicon.json').read_text())
    prefix = data['whole_word_rows']
    gram_count = data['rows']-prefix
    if gram_count != len(lexicon['grams']) or gram_count <= 0:
        raise ValueError('nonempty contiguous n-gram suffix required')
    source = np.memmap(checkpoint/'source.f32',mode='r',dtype='<f4',shape=(data['rows'],data['dim']))
    report = {'status':'running','source_sha256':training['source_sha256'], 'seed':training['seed'],
              'precomputed_word_sha256':base['files_sha256']['word-vectors.f32'],
              'original_whole_word_rows_removed':prefix, 'variants':[]}
    try:
        for backend in backends:
            started = time.perf_counter()
            root = output/backend
            root.mkdir()
            copy_file(checkpoint/'head.npz',root/'head.npz')
            copy_file(precomputed/'word-vectors.f32',root/'word-vectors.f32')
            write_json(root/'lexicon.json',{**lexicon,'whole_words':[],'whole_word_rows':0})
            fallback = backend[len('precomputed-hybrid-'):]
            if fallback in ('source','int8','ngram-int8'):
                staging = root/'source.f32'
                partition = means = representatives = None
                if fallback == 'ngram-int8':
                    bounded = json.loads((existing/'ngram-bounded/classifier.json').read_text())
                    if bounded['trained_source_sha256'] != training['source_sha256']:
                        raise ValueError('partition uses a different trained source')
                    domain = existing/'ngram-bounded/memory/domain.npz'
                    if file_hash(domain) != bounded['files_sha256']['memory/domain.npz']:
                        raise ValueError('training-incidence partition changed')
                    partition = BagPartition.load(domain)
                    if len(partition.mapping) != len(source):
                        raise ValueError('partition source shape mismatch')
                    representatives = partition.representatives()[prefix:] - prefix
                    if np.any(representatives < 0) or np.any(representatives >= gram_count):
                        raise ValueError('n-gram group incorrectly shares a whole-word row')
                    means = partition.means(source)
                with staging.open('wb') as stream:
                    for start in range(prefix,len(source),4096):
                        rows = (source[start:start+4096] if means is None else
                                means[partition.mapping[start:start+4096]])
                        stream.write(rows.astype('<f4').tobytes())
                if fallback != 'source':
                    options = {'native':False} if fallback=='int8' else {
                        'native':True,'representatives':representatives.tolist(),
                        'deduplicate':False,'max_block_bytes':8192}
                    pack_quantized_file(staging,root/'memory',data['dim'],'int8',**options)
                    staging.unlink()
            else:
                codec_root = existing/'faiss-sq4'
                bound = json.loads((codec_root/'classifier.json').read_text())
                if bound['trained_source_sha256'] != training['source_sha256']:
                    raise ValueError('SQ4 uses a different trained source')
                for name,digest in bound['files_sha256'].items():
                    if name.startswith('memory/') and file_hash(codec_root/name) != digest:
                        raise ValueError('existing SQ4 codec changed')
                memory = root/'memory'
                memory.mkdir()
                copy_file(codec_root/'memory/codec.faiss',memory/'codec.faiss')
                codes = np.load(codec_root/'memory/codes.npy',mmap_mode='r',allow_pickle=False)
                slim = np.lib.format.open_memmap(memory/'codes.npy',mode='w+',dtype=codes.dtype,
                                                shape=(gram_count,codes.shape[1]))
                try:
                    for start in range(0,gram_count,4096):
                        slim[start:start+4096] = codes[prefix+start:prefix+start+4096]
                    slim.flush()
                finally:
                    slim._mmap.close(); codes._mmap.close()
                codec = json.loads((codec_root/'memory/codec.json').read_text())
                write_json(memory/'codec.json',{**codec,'rows':gram_count,'reused_source_codec':True,
                    'source_codec_sha256':file_hash(codec_root/'memory/codec.faiss'),
                    'source_codes_sha256':file_hash(codec_root/'memory/codes.npy'),
                    'whole_word_prefix_removed':prefix})
            variant = finalize(root,backend,{**data,'rows':gram_count},training,started)
            report['variants'].append(variant)
            write_json(output/'export.json',report)
            print(json.dumps({'backend':backend,'complete_model_bytes':variant['complete_model_bytes']}),flush=True)
    finally:
        source._mmap.close()
    report['status'] = 'passed'
    write_json(output/'export.json',report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--existing-serving',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU-only export requires empty CUDA_VISIBLE_DEVICES')
    export_hybrids(cfg.training/'common',cfg.training/f'seed-{cfg.seed}',cfg.existing_serving,cfg.output)


if __name__ == '__main__':
    main()
