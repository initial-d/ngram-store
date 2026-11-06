"""Audit same-seed joint-memory controls using paired held-out windows.

Standard-library only. All final seeds are reported. Independent controls are
paired by protocol, data/prefix hashes and initial decoder/memory hashes, not
merely by comparable headline PPL. No checkpoint or test-based selection.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random


CONTROLS=('full','head-constant','off')


def read_json(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as f:return json.load(f)
    return json.loads(path.read_text())


def quality_path(directory):
    path=directory/'quality.json'
    return path if path.exists() else directory/'quality.json.gz'


def quantile(values,q):
    pos=(len(values)-1)*q;lo=int(pos);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)


def paired_nll(full,baseline,*,seed=731,resamples=10000):
    if len(full)!=len(baseline) or not full:raise ValueError('paired windows required')
    counts=[];sums=[]
    for i,(a,b) in enumerate(zip(full,baseline)):
        if (a['window_index']!=i or b['window_index']!=i or a['targets']!=b['targets']
                or len(a['source_nll'])!=a['targets'] or len(b['source_nll'])!=b['targets']):
            raise ValueError('window/target pairing changed')
        if not all(math.isfinite(x) for x in a['source_nll']+b['source_nll']):raise ValueError('finite NLL required')
        counts.append(a['targets']);sums.append(math.fsum(b['source_nll'])-math.fsum(a['source_nll']))
    targets=sum(counts)
    source=math.fsum(x for w in full for x in w['source_nll'])/targets
    candidate=math.fsum(x for w in baseline for x in w['source_nll'])/targets
    rng=random.Random(seed);values=[]
    for _ in range(resamples):
        draw=[rng.randrange(len(full)) for _ in full]
        values.append(math.fsum(sums[i] for i in draw)/sum(counts[i] for i in draw))
    values.sort();bounds=[quantile(values,q) for q in (.025,.975)]
    return {'windows':len(full),'targets':targets,'full_ppl':math.exp(source),'baseline_ppl':math.exp(candidate),
            'baseline_minus_full_nll':candidate-source,'baseline_relative_ppl_excess':math.expm1(candidate-source),
            'baseline_minus_full_nll_95ci':bounds,
            'bootstrap':{'rng':'Python random.Random / randrange','seed':seed,'resamples':resamples,
                         'quantile':'linear interpolation','unit':'paired nonoverlapping token window'}}


def source_windows(quality,control,split):
    return quality['windows']['source' if control=='off' else 'native-fp32'][split]


def analyze(root,protocol):
    seeds={};hashes={}
    for seed in protocol['seeds']:
        reports={};quality={}
        for control in CONTROLS:
            directory=root/(control+'-seed'+str(seed));r=read_json(directory/'report.json')
            if r['status']!='passed' or r['seed']!=seed or r['control']!=control or r['protocol']!=protocol:
                raise ValueError('missing/failed/mismatched registered control')
            if r['steps']!=protocol['epochs']*protocol['training_windows']//protocol['batch_size']:
                raise ValueError('training budget changed')
            if not r['frozen_weight_hash_unchanged'] or not r['final_reuse_check']['all_trainable_gradients_exact']:
                raise ValueError('joint training contract failed')
            qpath=quality_path(directory)
            raw=gzip.decompress(qpath.read_bytes()) if qpath.suffix=='.gz' else qpath.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=r['quality_sha256']:raise ValueError('quality bytes changed')
            reports[control]=r;quality[control]=json.loads(raw)
            hashes[control+'-seed'+str(seed)]={'report_sha256':hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
                                             'quality_sha256':hashlib.sha256(raw).hexdigest()}
        full=reports['full']
        for control,r in reports.items():
            for name in ('code_sha','source_file_hashes','splits','sampled_train_windows','frozen_weight_sha256',
                         'prefix_files','initial_full_memory_sha256','initial_decoder_sha256','head_offsets'):
                if r[name]!=full[name]:raise ValueError('independent-control pairing changed: '+name)
            if r['initial_decoder_sha256']==r['trained_decoder_sha256']:raise ValueError('decoder did not adapt')
        if reports['off']['memory_trainable_parameters']!=0 or reports['off']['changed_physical_memory_rows']!=0:
            raise ValueError('off control trained unused memory')
        if not all(x['logits_byte_exact'] for x in reports['off']['plain_reload_checks'].values()):
            raise ValueError('ordinary baseline checkpoint changed logits')
        entry={'trainable_parameters':{c:r['trainable_parameters'] for c,r in reports.items()},
               'training_seconds':{c:r['training_seconds'] for c,r in reports.items()},'splits':{}}
        for split in ('calibration','test'):
            source=source_windows(quality['full'],'full',split)
            if len(source)!=protocol[split+'_windows']:raise ValueError('held-out budget changed')
            pairs={c:paired_nll(source,source_windows(quality[c],c,split)) for c in ('head-constant','off')}
            ablations=quality['full']['summary'][split]
            if not ablations['native-fp32']['all_logits_byte_exact']:raise ValueError('native source bytes changed')
            gate=protocol['minimum_ppl_effect']
            independent=all(p['baseline_relative_ppl_excess']>=gate and p['baseline_minus_full_nll_95ci'][0]>0 for p in pairs.values())
            own=all(ablations[c]['relative_ppl_change']>=gate and ablations[c]['bootstrap']['delta_nll_95ci'][0]>0 for c in ('memory-off','head-mean'))
            entry['splits'][split]={'full_ppl':pairs['off']['full_ppl'],'independent_controls':pairs,
                'same_checkpoint_ablations':{c:ablations[c] for c in ('memory-off','head-mean')},
                'native_byte_exact':True,'independent_controls_qualified':independent,
                'same_checkpoint_ablations_qualified':own,'qualified':independent and own}
        seeds[str(seed)]=entry
    return {'schema_version':1,'protocol':protocol,'source_hashes':hashes,'seeds':seeds,
        'all_seeds_qualified_on_calibration':all(s['splits']['calibration']['qualified'] for s in seeds.values()),
        'all_seeds_confirmed_on_test':all(s['splits']['test']['qualified'] for s in seeds.values()),
        'limitations':['fixed exploratory late-decoder adaptation, same sampling/optimizer schedule but different parameter counts',
                       'shared text windows and three seeds; bootstrap is not population/model-family confidence',
                       'unweighted trained constants and unweighted head-mean ablation are distinct controls',
                       'dense/prefix-cached quality oracle; no isolated serving or whole-model savings conclusion']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=analyze(args.training_root,read_json(args.protocol))
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
