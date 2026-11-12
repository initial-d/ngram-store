"""Dependency-free audit of actual GPU training, codecs and serving evidence."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def read(path):
    path=ROOT/path
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:return json.load(stream)


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)


def quantile(values,q):
    values=sorted(values);at=(len(values)-1)*q
    lo,hi=math.floor(at),math.ceil(at)
    return values[lo]+(values[hi]-values[lo])*(at-lo)


def metrics(windows,summary):
    n=sum(w['targets'] for w in windows)
    assert len(windows)==64 and n==8128
    assert [w['window_index'] for w in windows]==list(range(64))
    for w in windows:
        assert w['targets']==127
        for key in ('source_nll','candidate_nll','source_to_candidate_kl'):
            assert len(w[key])==127 and all(math.isfinite(x) and x>=0 for x in w[key])
        assert all(0<=w[k]<=127 for k in ('source_correct','candidate_correct','top1_changes'))
    a=math.fsum(x for w in windows for x in w['source_nll'])/n
    b=math.fsum(x for w in windows for x in w['candidate_nll'])/n
    kl=[x for w in windows for x in w['source_to_candidate_kl']]
    for key,value in {'source_nll':a,'candidate_nll':b,'delta_nll':b-a,
                      'source_ppl':math.exp(a),'candidate_ppl':math.exp(b),
                      'relative_ppl_change':math.expm1(b-a),'mean_kl':math.fsum(kl)/n,
                      'p99_kl':quantile(kl,.99),'max_kl':max(kl)}.items():close(summary[key],value)
    for key in ('top1_changes','source_correct','candidate_correct'):
        assert summary[key]==sum(w[key] for w in windows)
    assert summary['all_logits_byte_exact']==all(w['logits_byte_exact'] for w in windows)
    boot=summary['bootstrap'];assert (boot['seed'],boot['resamples'])==(731,10000)
    lo,hi=boot['delta_nll_95ci'];assert math.isfinite(lo) and math.isfinite(hi) and lo<=hi
    for x,y in zip(boot['relative_ppl_change_95ci'],(math.expm1(lo),math.expm1(hi))):close(x,y)


def gate(summary):
    return all(summary[name]['relative_ppl_change']>=.01 and summary[name]['bootstrap']['delta_nll_95ci'][0]>0 for name in ('memory-off','head-mean'))


def main():
    manifest=read('manifest.json')
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    assert 'Ran 26 tests' in (ROOT/'controls/fixed-tests.log').read_text()
    assert 'OK (skipped=1)' in (ROOT/'controls/fixed-tests.log').read_text()
    inventory=read('inventory.json');models={}
    for order in (3,4,6):
        for seed in (3,7,11):
            name=f'order{order}-seed{seed}'
            prefix=Path('training')/name
            r=read(prefix/'report.json.gz');p=r['protocol'];s=r['seeds'][str(seed)]
            for filename,digest in r['source_file_hashes'].items():
                assert hashlib.sha256((ROOT/'runtime'/r['code_sha']/'tools'/filename).read_bytes()).hexdigest()==digest
            assert r['status']==s['status']=='passed'
            assert r['executed_seeds']==[seed] and p['seeds']==[3,7,11]
            assert not r['all_seeds_qualified_on_calibration'] and not r['all_seeds_confirmed_on_test']
            assert p['max_ngram_size']==order and p['epochs']==4 and s['steps']==1024
            assert len(s['losses'])==1024 and r['training_targets_per_seed']==520192
            assert r['environment']['device']=='cuda' and not r['environment']['tf32']
            assert s['frozen_weight_hash_unchanged'] and s['reuse_check']['logits_byte_exact'] and s['reuse_check']['branch_gradients_exact']
            memory=s['memory'];assert memory['raw_bytes']==memory['rows']*memory['dim']*4
            assert memory['dim']==32 and 0<memory['changed_rows']<=memory['queried_rows']<=memory['rows']
            q=read(prefix/f'seed-{seed}/quality.json.gz')
            assert {k:v for k,v in q.items() if k!='windows'}==s['quality']
            assert hashlib.sha256(gzip.decompress((ROOT/prefix/f'seed-{seed}/quality.json.gz').read_bytes())).hexdigest()==s['quality_sha256']
            assert (ROOT/'training'/f'{name}.exit-code').read_text().strip()=='0'
            for split,summary in q['summary'].items():
                for control,ss in summary.items():
                    ws=q['windows'][control][split];metrics(ws,ss)
                    assert all(w['source_nll']==q['windows']['memory-off'][split][i]['source_nll'] for i,w in enumerate(ws))
                    if control=='native-fp32':
                        assert ss['all_logits_byte_exact'] and ss['top1_changes']==0 and ss['max_kl']==0
                assert not gate(summary)
            assert not q['memory_qualified_on_calibration'] and not q['test_confirmation']
            assert inventory[name]['complete_bundle_bytes']==s['complete_bundle_bytes']
            models[name]=q
            codecs=read(Path('codecs')/name/'report.json.gz')
            assert codecs['status']=='passed' and codecs['memory']==memory
            assert codecs['protocol']==p and codecs['training_report_sha256']==hashlib.sha256(gzip.decompress((ROOT/prefix/'report.json.gz').read_bytes())).hexdigest()
            assert set(codecs['codecs'])=={'fp16','int8','sq8','sq4','pq-8x8','cluster-8','opq-8x8'}
            for codec,item in codecs['codecs'].items():
                path=Path('codecs')/name/(codec+'-windows.json.gz')
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==item['windows_sha256']
                windows=read(path)
                for split,summary in item['summary'].items():
                    metrics(windows[split],summary)
                    assert all(w['source_nll']==q['windows']['native-fp32'][split][i]['source_nll'] for i,w in enumerate(windows[split]))
                if codec in ('fp16','int8'):
                    assert item['native_matches_raw_decoded_bytes']
                    assert item['codec_metadata']['raw']['decoded_float32_sha256']==item['codec_metadata']['native']['decoded_float32_sha256']==item['decoded_rows_sha256']
            for budget,selection in codecs['selection'].items():
                candidates=[(size,codec,backend) for codec,item in codecs['codecs'].items()
                            if item['summary']['calibration']['bootstrap']['relative_ppl_change_95ci'][1]<=float(budget)
                            for backend,size in item['storage_bytes'].items()]
                chosen=min(candidates) if candidates else None
                assert selection['choice']==(list(chosen) if chosen else None)
                assert selection['test_confirms']==(codecs['codecs'][chosen[1]]['summary']['test']['bootstrap']['relative_ppl_change_95ci'][1]<=float(budget) if chosen else None)
    for seed in (3,7,11):
        name=f'low-lr-seed{seed}'
        r=read(Path('diagnostic')/name/'report.json.gz');s=r['seeds'][str(seed)]
        assert r['status']==s['status']=='passed' and s['steps']==256 and len(s['losses'])==256
        assert r['protocol']['epochs']==1 and r['protocol']['test_window_start']==64
        q=read(Path('diagnostic')/name/f'seed-{seed}/quality.json.gz')
        for split,summary in q['summary'].items():
            for control,ss in summary.items():metrics(q['windows'][control][split],ss)
            assert not gate(summary)
    count=0
    for folder,expected in [('serving',60),('cache-serving',9)]:
        reports=[p for p in (ROOT/folder).glob('*/*.json.gz') if '-attempt' not in p.name]
        assert len(reports)==expected,(folder,len(reports))
        for path in reports:
            r=read(path.relative_to(ROOT));assert r['status']=='passed'
            assert not r['gpu_before']['compute_pids'] and r['gpu_before']['utilization_percent']<=5
            assert len(r['cases'])==6
            assert r['settings']['repeat']==7 and r['settings']['decode_tokens']==16
            reference=read(path.parent.relative_to(ROOT)/'resident.json.gz')
            assert reference['inputs']==r['inputs'] and reference['settings']==r['settings'] and reference['software']==r['software']
            assert reference['gpu_before']['gpu_uuid']==r['gpu_before']['gpu_uuid']
            for filename,digest in r['source_hashes'].items():
                assert hashlib.sha256((ROOT/'runtime'/r['commit']/'tools'/filename).read_bytes()).hexdigest()==digest
            for c,oracle in zip(r['cases'],reference['cases']):
                assert c['key']==oracle['key'] and c['input_ids_sha256']==oracle['input_ids_sha256']
                assert len(c['seconds'])==7 and all(t>0 for t in c['seconds'])
                close(c['median_seconds'],statistics.median(c['seconds']))
                assert 'generated_ids' not in c
                if r['mode'] in ('native','dense-offload'):assert c['same_schedule_byte_exact'] and c['output_sha256']==oracle['output_sha256']
            count+=1
    assert len(list((ROOT/'relations').glob('*.json.gz')))==9
    for path in (ROOT/'relations').glob('*.json.gz'):
        r=read(path.relative_to(ROOT));assert r['status']=='passed'
        for head in r['heads']:
            assert head['eligible_rows']<=head['observed_rows']-head['observed_ambiguous_rows']
            for side in ('prefix','suffix'):
                value=head[side]
                if value['pairs']:
                    assert all(-1.00000001<=x<=1.00000001 for x in value['cosine_quantiles'])
                    assert value['joint_fraction']['0.99']==0
    assert read('superseded/training/report.json.gz')['status']=='failed'
    assert read('superseded/order4-seed3/report.json.gz')['status']=='failed'
    print(f'Verified 12 trained checkpoints, 63 codec comparisons, {count} isolated serving workers, source pairing and negative qualification results.')


if __name__=='__main__':main()
