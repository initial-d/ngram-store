"""Regenerate compact fixed-model summaries from archived raw reports."""
import gzip
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def read(path):
    path=ROOT/path
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:return json.load(stream)


def main():
    inventory=read('inventory.json')
    result={'scope':'Frozen Qwen3-0.6B, one late TinyEngram branch, fixed local TinyShakespeare; not released DeepSeek weights or necessary-backend evidence',
            'checkpoint_target_evaluations':146304,'unique_original_heldout_targets':16256,
            'supplemental_checkpoint_target_evaluations':48768,'additional_unique_test_targets':8128,
            'orders':{},'primary_codecs':{},'cache_cases':{},'diagnostic':{}}
    for order in (3,4,6):
        runs=[read(Path('training')/f'order{order}-seed{seed}'/'report.json.gz')['seeds'][str(seed)] for seed in (3,7,11)]
        raw=runs[0]['memory']['raw_bytes']
        result['orders'][str(order)]={'rows':runs[0]['memory']['rows'],'dim':32,'raw_bytes':raw,
            'native_memory_bytes_range':[min(sum(f['bytes'] for f in inventory[f'order{order}-seed{seed}']['memory_files'].values()) for seed in (3,7,11)),
                                         max(sum(f['bytes'] for f in inventory[f'order{order}-seed{seed}']['memory_files'].values()) for seed in (3,7,11))],
            'complete_checkpoint_bytes_range':[min(r['complete_checkpoint_bytes'] for r in runs),max(r['complete_checkpoint_bytes'] for r in runs)],
            'complete_bundle_bytes_range':[min(r['complete_bundle_bytes'] for r in runs),max(r['complete_bundle_bytes'] for r in runs)],
            'final32_training_ce_range':[min(statistics.mean(r['losses'][-32:]) for r in runs),max(statistics.mean(r['losses'][-32:]) for r in runs)],
            'source_test_ppl_range':[min(r['quality']['summary']['test']['native-fp32']['candidate_ppl'] for r in runs),max(r['quality']['summary']['test']['native-fp32']['candidate_ppl'] for r in runs)],
            'off_test_ppl':runs[0]['quality']['summary']['test']['memory-off']['candidate_ppl'],
            'head_mean_test_ppl_range':[min(r['quality']['summary']['test']['head-mean']['candidate_ppl'] for r in runs),max(r['quality']['summary']['test']['head-mean']['candidate_ppl'] for r in runs)],
            'qualified_seeds':sum(r['quality']['memory_qualified_on_calibration'] for r in runs)}
    for seed in (3,7,11):
        r=read(Path('diagnostic')/f'low-lr-seed{seed}'/'report.json.gz')['seeds'][str(seed)]
        result['diagnostic'][str(seed)]=r['quality']['summary']
    runs=[read(Path('codecs')/f'order4-seed{seed}'/'report.json.gz') for seed in (3,7,11)]
    for codec in runs[0]['codecs']:
        items=[r['codecs'][codec] for r in runs]
        summaries=[r['summary']['test'] for r in items]
        result['primary_codecs'][codec]={'backend_bytes_median':{backend:statistics.median(r['storage_bytes'][backend] for r in items) for backend in items[0]['storage_bytes']},
            'relative_test_ppl_change_range':[min(r['relative_ppl_change'] for r in summaries),max(r['relative_ppl_change'] for r in summaries)],
            'test_mean_kl_range':[min(r['mean_kl'] for r in summaries),max(r['mean_kl'] for r in summaries)],
            'test_top1_change_fraction_range':[min(r['top1_changes']/r['targets'] for r in summaries),max(r['top1_changes']/r['targets'] for r in summaries)]}
    cache=[{name:read(Path('cache-serving')/f'seed{seed}'/(name+'.json.gz')) for name in ('resident','native-blocks','native-rows')} for seed in (3,7,11)]
    for i,c in enumerate(cache[0]['resident']['cases']):
        result['cache_cases'][c['key']]={'median_seconds':{name:statistics.median(r[name]['cases'][i]['median_seconds'] for r in cache) for name in ('resident','native-blocks','native-rows')},
            'row_speedup_range':[min(r['native-blocks']['cases'][i]['median_seconds']/r['native-rows']['cases'][i]['median_seconds'] for r in cache),
                                 max(r['native-blocks']['cases'][i]['median_seconds']/r['native-rows']['cases'][i]['median_seconds'] for r in cache)],
            'gpu_allocator_saving_bytes_range':[min(r['resident']['cases'][i]['gpu_peak_allocated_bytes']-r['native-rows']['cases'][i]['gpu_peak_allocated_bytes'] for r in cache),
                                                max(r['resident']['cases'][i]['gpu_peak_allocated_bytes']-r['native-rows']['cases'][i]['gpu_peak_allocated_bytes'] for r in cache)]}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
