"""Stdlib verification of complete finite memory-application evidence.

Checks public artifact hashes, candidate counts, all quality arithmetic, frozen
choices, full-request provenance, paired intervals and serial serving samples.
Private model execution is represented by immutable reports and source hashes.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

BUDGETS = (.0001, .001, .005, .01, .02)
STRATA = ('all', 'known', 'oov_nonempty', 'oov_empty')
GROUPS = ('rewrite', 'paraphrase', 'neighborhood')


def load(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10), (a, b)


def quantile(values, p):
    values = sorted(values); at = (len(values) - 1) * p
    low, high = math.floor(at), math.ceil(at)
    return values[low] + (at - low) * (values[high] - values[low])


def source(environment, inventory):
    assert environment['python'] and environment['torch'] and environment['os']
    commit = environment['commit']; assert commit in inventory
    assert environment['source_sha256'] == inventory[commit]['files']['tools/tree_node_quality.py']['original_sha256']
    assert '/' + 'llm-align/' not in environment['command']


def qwen_source(environment, inventory):
    assert environment['python'] and environment['torch'] and environment['transformers'] and environment['os']
    assert environment['tf32'] is False and '/' + 'llm-align/' not in environment['command']
    files = inventory[environment['code_sha']]['files']
    for name, expected in environment['source_hashes'].items():
        assert files['tools/' + name]['original_sha256'] == expected


def gate(quality, budget):
    return all(quality[s]['pairs'] > 0 and quality[s]['loss_relative_change'] <= budget
               and quality[s]['discrimination_change'] >= -.005 for s in ('all', 'oov_nonempty'))


def paired(blocks, quality, total):
    assert set(blocks) == set(STRATA)
    for name, block in blocks.items():
        ids = block['block_ids']; assert ids == sorted(set(ids))
        assert all(len(v) == len(ids) for v in block.values())
        assert all(0 <= i <= (total['tokens'] - 1) // 128 for i in ids)
        count = sum(block['count']); q = quality[name]
        assert count == q['pairs'] and all(isinstance(v, int) and v > 0 for v in block['count'])
        if count:
            candidate, baseline = sum(block['candidate_loss']), sum(block['baseline_loss'])
            win, refwin = sum(block['candidate_discrimination']), sum(block['baseline_discrimination'])
            close(q['sgns_loss_per_pair'], candidate / count)
            close(q['baseline_sgns_loss_per_pair'], baseline / count)
            close(q['context_discrimination'], win / count)
            close(q['baseline_context_discrimination'], refwin / count)
            close(q['loss_relative_change'], candidate / baseline - 1)
            close(q['discrimination_change'], (win - refwin) / count)
        else:
            assert q['loss_relative_change'] is q['discrimination_change'] is None
    assert quality['all']['pairs'] == total['pairs']
    for field in ('count', 'candidate_loss', 'baseline_loss', 'candidate_discrimination', 'baseline_discrimination'):
        parts = {}
        for name in STRATA[1:]:
            for key, value in zip(blocks[name]['block_ids'], blocks[name][field]):
                parts[key] = parts.get(key, 0) + value
        assert set(parts) == set(blocks['all']['block_ids'])
        for key, value in zip(blocks['all']['block_ids'], blocks['all'][field]):
            close(value, parts[key])


def intervals(value):
    if value['observed_blocks'] == 0:
        assert value['interval'] is None
        return
    assert value['repeats'] == len(value['samples']) == 2000 and value['block_tokens'] == 128 and value['seed'] == 42
    for index, field in enumerate(('loss_relative_change_95_percent_interval', 'discrimination_change_95_percent_interval')):
        for endpoint, probability in enumerate((.025, .975)):
            close(value[field][endpoint], quantile([row[index] for row in value['samples']], probability))


def metrics(cases):
    result = {}
    assert len({c['case_id'] for c in cases}) == len(cases)
    for name in GROUPS:
        success = []; accuracy = []
        for case in cases:
            rows = case['groups'][name]; assert rows
            for row in rows:
                assert row['success'] == (row['target_true'] < row['target_new'] if name == 'neighborhood' else row['target_new'] < row['target_true'])
                assert math.isfinite(row['target_new']) and math.isfinite(row['target_true'])
            success.append(sum(r['success'] for r in rows) / len(rows))
            accuracy.append(sum(r['correct'] for r in rows) / len(rows))
        result[name] = {'cases': len(cases), 'prompts': sum(len(c['groups'][name]) for c in cases),
                        'success': math.fsum(success) / len(cases), 'teacher_forced_accuracy': math.fsum(accuracy) / len(cases)}
    values = [result[name]['success'] for name in GROUPS]
    result['harmonic_score'] = 3 / sum(1 / v for v in values) if all(values) else 0.
    return result


def check_metrics(actual, expected):
    close(actual['harmonic_score'], expected['harmonic_score'])
    for name in GROUPS:
        assert actual[name]['cases'] == expected[name]['cases'] and actual[name]['prompts'] == expected[name]['prompts']
        close(actual[name]['success'], expected[name]['success'])
        close(actual[name]['teacher_forced_accuracy'], expected[name]['teacher_forced_accuracy'])


def metric_values(value):
    return [value['harmonic_score']] + [value[name]['success'] for name in GROUPS]


def cpu_projection(report):
    keys = ('decoded_row_sha256', 'cpu_affinity', 'cpu_lane', 'cpu_parallel_plan_sha256',
            'measurement_cohort', 'worker_source_sha256', 'environment', 'measurement_host',
            'sample_started','sample_finished')
    result = {key: report[key] for key in keys if key in report}
    fields = ('output_sha256', 'input_hashes', 'output_hashes', 'complete_p50_ms', 'p50_ms')
    result['query'] = {kind: {k: v for k, v in value.items() if k in fields}
                       for kind, value in report['query'].items()}
    return result


def verify_cpu_schedule(root, controller, frozen, records, phase, base, inventory, private):
    assert controller['status'] == 'complete'
    if controller['workers'] == 1:
        return
    plan = load(root / 'cpu-serving-parallel-plan.json')
    assert controller['cpu_parallel_plan_sha256'] == private['cpu-serving-parallel-plan.json']
    assert controller['workers'] == plan['workers'] == 8
    concurrent=root/'cpu-parallel-reference-plan.json'
    if concurrent.exists():
        reference_plan=load(concurrent)
        assert controller['serial_reference_workers']==reference_plan['workers']==8
        assert controller['reference_execution']=='parallel-fixed-core'
        assert controller['parallel_reference_plan_sha256']==private['cpu-parallel-reference-plan.json']
        assert reference_plan['original_cpu_plan_sha256']==private['cpu-serving-parallel-plan.json']
    else:assert controller['serial_reference_workers']==1
    assert controller['model_lanes'] == plan['model_lanes'][phase] and controller['lanes'] == plan['lanes']
    if plan.get('measurement_host'):
        assert controller['measurement_host']==plan['measurement_host']
        assert controller['parallel_with_gpu_shards']==plan['parallel_with_gpu_shards']
    assert set(controller['model_lanes']) == {m['name'] for m in frozen['models']}
    logical = set(); physical = set()
    for lane in plan['lanes'].values():
        cpus = set(lane['cpus']); cores = {tuple(v) for v in lane['physical_cores']}
        assert cpus and cores and not logical.intersection(cpus) and not physical.intersection(cores)
        logical.update(cpus); physical.update(cores)
    originals = {json.dumps(r['request'], sort_keys=True): r for r in records}
    assert len(originals) == len(controller['requests']) == len(controller['complete'])
    overlap=root/'cpu-phase-overlap-plan.json'
    overlap_finished=None
    if overlap.exists():
        execution=load(root/'diagnostics/cpu-phase-overlap-controller.json');overlap_plan=load(overlap)
        assert execution['status']=='complete' and execution['failures']==[] and execution['workers']==6
        assert execution['plan_sha256']==private['cpu-phase-overlap-plan.json']
        assert overlap_plan['original_cpu_plan_sha256']==private['cpu-serving-parallel-plan.json']
        files=inventory[execution['environment']['commit']]['files']
        assert execution['phase_source_sha256']==files['tools/cpu_phase_overlap.py']['original_sha256']
        assert execution['started']<=execution['finished']
        overlap_finished=execution['finished']
        if phase=='retime-llm-cpu':
            assert overlap_plan['requests']==execution['requests']==controller['requests']
            precalculated={json.dumps(v['request'],sort_keys=True):v for v in execution['complete']}
            assert len(precalculated)==len(execution['complete'])==len(originals)
            assert set(precalculated)==set(originals)
            for key,item in precalculated.items():
                assert item['sha256']==originals[key]['private_sha256']
                value=originals[key]['report']
                assert execution['started']<=value['sample_started']<=value['sample_finished']<=execution['finished']
        else:
            assert max(v['report']['sample_finished'] for v in originals.values())<=execution['started']
    names = {}
    for model in frozen['models']:
        selected = set()
        for choice in model['choices']:
            primary = choice.get('budget_pp') == .5 if phase == 'retime-llm-cpu' else choice.get('budget') == .001 and choice['cached'] == 0
            if primary and choice['name']:
                selected.add(choice['name'])
        names[model['name']] = selected | {'raw-fp32', 'raw-fp16', 'raw-int8', 'dense-fp32',
                                         'native-none-fp32', 'native-none-fp16', 'native-none-int8'}
    expected = [r for r in controller['requests'] if r['cache'] == 0 and r['name'] in names[r['model']] and (
        phase == 'retime-llm-cpu' or r['cached'] == 0 and (
            r['index'] == 'datrie' or r['name'] == 'raw-fp32' or r['name'].startswith('tree-datrie-')))]
    assert controller['reference_requests'] == expected
    assert len(controller['serial_references']) == len(expected)
    refs = {}
    for model in frozen['models']:
        folder = root / base / model['name']
        for item in load(folder / 'serial-references.json.gz'):
            key = json.dumps(item['request'], sort_keys=True)
            assert key not in refs
            refs[key] = item
            report = item['report']
            source(report['environment'], inventory)
            if plan.get('measurement_host'):
                marker=load(root/'local-gpu-finished.json')
                assert report['measurement_host']==plan['measurement_host']==marker['measurement_host']
                assert report['local_gpu_gate_sha256']==private['local-gpu-finished.json']
                assert report['sample_started']>=marker['finished']
            if overlap_finished is not None:assert report['sample_started']>=overlap_finished
            if phase == 'retime-long':
                long_timing(report, load(folder / 'metadata.json'))
            else:
                llm_timing(report, 'cpu')
    assert set(refs) == {json.dumps(r, sort_keys=True) for r in expected}
    for key, item in originals.items():
        report = item['report']; request = item['request']; lane = controller['model_lanes'][request['model']]
        assert report['measurement_cohort'] == 'parallel-shared-host'
        assert report['cpu_lane'] == lane and report['cpu_affinity'] == plan['lanes'][lane]['cpus']
        assert report['cpu_parallel_plan_sha256'] == controller['cpu_parallel_plan_sha256']
        if plan.get('measurement_host'):assert report['measurement_host']==plan['measurement_host']
        files = inventory[report['environment']['commit']]['files']
        assert report['worker_source_sha256'] == files['tools/parallel_cpu_serving.py']['original_sha256']
        if report.get('process_launch'):
            launch=report['process_launch'];assert launch['kind']=='fresh-fork'
            assert launch['parent_commit']==report['environment']['commit']
            assert launch['source_sha256']==files['tools/cpu_fork_serving.py']['original_sha256']
    for item, request in zip(controller['serial_references'], expected):
        assert item['request'] == request
        key = json.dumps(request, sort_keys=True); reference = refs[key]; original = originals[key]
        a, b = original['report'], reference['report']
        assert item['sha256'] == reference['private_sha256']
        assert item['parallel_output_sha256'] == original['private_sha256']
        assert b['measurement_cohort'] == 'serial-reference' and b['cpu_affinity'] == a['cpu_affinity']
        assert b['cpu_lane'] == a['cpu_lane'] and b['decoded_row_sha256'] == a['decoded_row_sha256']
        assert b['cpu_parallel_plan_sha256'] == a['cpu_parallel_plan_sha256']
        files = inventory[b['environment']['commit']]['files']
        assert b['worker_source_sha256'] == files['tools/parallel_cpu_serving.py']['original_sha256']
        if concurrent.exists():
            launch=b['reference_execution'];assert launch['kind']=='parallel-fixed-core' and launch['workers']==8
            assert launch['plan_sha256']==private['cpu-parallel-reference-plan.json']
            assert launch['source_sha256']==files['tools/cpu_parallel_references.py']['original_sha256']
        for kind, value in a['query'].items():
            actual = b['query'][kind]
            if 'output_sha256' in value:
                assert value['output_sha256'] == actual['output_sha256']
                ratio = value['complete_p50_ms'] / actual['complete_p50_ms']
            else:
                assert value['input_hashes'] == actual['input_hashes'] and value['output_hashes'] == actual['output_hashes']
                ratio = value['p50_ms'] / actual['p50_ms']
            close(reference['parallel_to_serial_p50_ratios'][kind], ratio)
            close(item['parallel_to_serial_p50_ratios'][kind], ratio)


def long_timing(report, metadata):
    assert report['status'] == 'complete' and report['all_original_rows_verified'] == metadata['shape'][0]
    assert report['load_seconds'] >= 0 and report['memory_after_query']['peak_bytes'] > 0
    for name, result in report['query'].items():
        assert result['inputs'] == metadata['query_types'][name]
        repeats = 1 if report.get('measurement_purpose') == 'quality-and-reopening-only' else 15
        assert len(result['complete_samples_ms']) == len(result['replay_samples_ms']) == repeats
        assert all(v > 0 for v in result['complete_samples_ms'])
        close(result['complete_p50_ms'], statistics.median(result['complete_samples_ms']))
        close(result['complete_p95_ms'], quantile(result['complete_samples_ms'], .95))
        close(result['replay_p50_ms'], statistics.median(result['replay_samples_ms']))
        close(result['inputs_per_second'], result['inputs'] * 1000 / result['complete_p50_ms'])


def llm_timing(report, phase, *, request_count=32, repeat=3):
    assert report['status'] == 'passed' and report['open_seconds'] >= 0
    if phase == 'cpu':
        for category in ('query', 'row_only'):
            for split in ('calibration', 'test'):
                result = report[category][split]
                values = result.get('warm_samples_ms', result.get('samples_ms'))
                assert values and all(v > 0 for v in values)
                close(result['p50_ms'], statistics.median(values))
                close(result['p95_ms'], quantile(values, .95))
    else:
        if report.get('measurement_mode')=='matched-prompt-shards':
            shards=report['shards'];cursor=0
            for record in shards:
                child=record['report'];start,count=child['request_slice'];assert start==cursor
                llm_timing(child,'gpu',request_count=count);cursor+=count
            assert cursor==32 and len(report['requests'])==32 and len(report['samples'])==96
            assert report['requests']==[v for record in shards for v in record['report']['requests']]
            assert report['samples']==[v for record in shards for v in record['report']['samples']]
            decode=[d for s in report['samples'] for d in s['decode_ms']];prefill=[s['prefill_ms'] for s in report['samples']]
            close(report['prefill_p50_ms'],statistics.median(prefill));close(report['prefill_p95_ms'],quantile(prefill,.95))
            close(report['decode_p50_ms'],statistics.median(decode));close(report['decode_p95_ms'],quantile(decode,.95))
            close(report['decode_tokens_per_second'],report['batch_size']*1000/statistics.mean(decode))
            return
        assert report['gpu_before']['compute_pids'] == report['gpu_before']['external_compute_pids'] == []
        assert report['gpu_before']['utilization_percent'] <= 5
        assert report['gpu_samples'] and all(not v['external_compute_pids'] and len(v['compute_pids']) == 1 for v in report['gpu_samples'])
        assert report['batch_size'] in (1, 8) and report['decode_steps'] == 16
        assert report['batch_contract'].startswith('Each batch uses distinct real official prompts')
        prompts=[]
        for request in report['requests']:
            assert len(request['prompt_hashes']) == len(request['batch_case_ids']) == report['batch_size']
            if report['batch_size']==8: assert len(set(request['prompt_hashes']))==8
            prompts.extend(request['prompt_hashes'])
        assert len(set(prompts))==report['unique_prompt_count']
        if report['batch_size']==8:assert report['unique_prompt_count']==request_count*8
        prefill = [v['prefill_ms'] for v in report['samples']]
        decode = [d for v in report['samples'] for d in v['decode_ms']]
        assert all(len(v['decode_ms']) == 16 for v in report['samples'])
        assert len(report['samples']) == len(report['requests']) * repeat and len(report['requests']) == request_count
        close(report['prefill_p50_ms'], statistics.median(prefill)); close(report['prefill_p95_ms'], quantile(prefill, .95))
        close(report['decode_p50_ms'], statistics.median(decode)); close(report['decode_p95_ms'], quantile(decode, .95))
        close(report['decode_tokens_per_second'], report['batch_size'] * 1000 / statistics.mean(decode))
        if report['mode'] == 'independent':
            assert report['independent_registry_required'] is False and report['independent_serving_files'] == ['delta.safetensors', 'report.json']


def verify_spare_cohort(root, controller, records, inventory, private):
    plan = load(root / 'gpu-spare-lane-plan.json')
    original = load(root / 'llm/original-gpu-controller.json')
    spare = load(root / 'llm/spare-gpu-controller.json')
    assert plan['schema_version'] == 1
    assert controller['gpu_spare_plan_sha256'] == spare['plan_sha256'] == private['gpu-spare-lane-plan.json']
    assert controller['original_controller_sha256'] == private['llm/original-gpu-controller.json']
    assert controller['spare_controller_sha256'] == private['llm/spare-gpu-controller.json']
    assert original['status'] == spare['status'] == controller['status'] == 'complete'
    assert controller['requests'] == original['requests'] == plan['requests']
    assert controller['fixed_device_assignment'] == original['fixed_device_assignment'] == plan['original_devices']
    assert controller['cpu_affinity'] == original['cpu_affinity'] == plan['cpu_affinity']
    assert controller['spare_cpu_affinity'] == plan['spare_cpu_affinity']
    assert controller['spare_device_assignment'] == plan['spares']
    assert len(set(plan['spares'].values())) == len(plan['spares'])
    assert not set(plan['spares'].values()).intersection(plan['original_devices'].values())
    assert controller['workers'] == len(plan['original_devices']) + len(plan['spares']) == 8
    assert spare['workers'] == len(plan['spares']) and spare['workers_per_gpu'] == 1
    expected = [r for r in plan['requests'] if r['model'] in plan['spares'] and r['batch'] == 8]
    assert spare['requests'] == expected
    key = lambda r: json.dumps(r, sort_keys=True)
    def index(items):
        result = {key(v['request']): v for v in items}
        assert len(result) == len(items)
        return result
    prior, replacement, effective, published = map(index, (original['complete'], spare['complete'], controller['complete'], records))
    assert set(prior) == set(effective) == set(published) == {key(r) for r in plan['requests']}
    assert set(replacement) == {key(r) for r in expected}
    cores = set()
    for cpus in [*plan['cpu_affinity'].values(), *plan['spare_cpu_affinity'].values()]:
        assert cpus and cpus == sorted(set(cpus)) and not cores.intersection(cpus); cores.update(cpus)
    for request in plan['requests']:
        k = key(request); chosen = replacement.get(k, prior[k])
        assert effective[k]['sha256'] == published[k]['private_sha256'] == chosen['sha256']
        model = request['model']; routed = k in replacement
        gpu = plan['spares'][model] if routed else plan['original_devices'][model]
        cpus = plan['spare_cpu_affinity'][model] if routed else plan['cpu_affinity'][model]
        report = published[k]['report']
        if report.get('measurement_purpose') in ('prompt-shard-routing-receipt-only','extra-shard-routing-receipt-only'):
            assert not routed and report['cpu_affinity']==cpus and report['request']==request
            continue
        assert effective[k]['gpu'] == gpu and report['selected_gpu_selector'] == str(gpu)
        assert report['cpu_affinity'] == cpus
        source(report['environment'], inventory)
        files = inventory[report['environment']['commit']]['files']
        for field, filename in (('phase_source_sha256', 'benchmark_llm_codec_serving.py'),
                                ('cached_decode_source_sha256', 'cached_memory_decode.py'),
                                ('codec_adapter_source_sha256', 'codec_memory_adapter.py')):
            assert report[field] == files['tools/' + filename]['original_sha256']
    preflights = load(root / 'llm/spare-gpu-preflights.json.gz')
    assert len(preflights) == len(plan['spares']) and {p['model'] for p in preflights} == set(plan['spares'])
    for entry in preflights:
        assert entry['status'] == 'passed' and entry['gpu'] == plan['spares'][entry['model']]
        a, b = entry['reports']
        assert a['mode'] == 'source' and b['mode'] == 'codec'
        for report in (a, b):
            assert report['status'] == 'passed' and len(report['requests']) == 2 and report['batch_size'] == 8
            assert report['selected_gpu_selector'] == str(entry['gpu'])
            assert report['cpu_affinity'] == plan['spare_cpu_affinity'][entry['model']]
            assert not report['gpu_before']['compute_pids'] and report['gpu_before']['utilization_percent'] <= 5
            assert report['gpu_samples'] and all(not v['external_compute_pids'] for v in report['gpu_samples'])
        for x, y in zip(a['requests'], b['requests']):
            for field in ('input_ids_sha256', 'cached_logit_hashes', 'generated_ids_sha256'): assert x[field] == y[field]
    diagnostics = index(load(root / 'llm/original-batch8-diagnostics.json.gz'))
    assert set(diagnostics) == set(replacement)
    for k, item in diagnostics.items(): assert item['private_sha256'] == prior[k]['sha256']


def verify_prompt_shards(root,controller,records,inventory,private):
    plan=load(root/'gpu-prompt-shard-plan.json');sharded=load(root/'llm/prompt-shard-controller.json')
    parent=load(root/'llm/pre-sharded-gpu-controller.json')
    assert plan['schema_version']==1 and sharded['status']=='complete' and sharded['workers']==8
    assert controller['prompt_shard_plan_sha256']==sharded['plan_sha256']==private['gpu-prompt-shard-plan.json']
    assert controller['requests']==parent['requests']==plan['requests']
    assert controller['prompt_shard_groups']==plan['groups'] and controller['prompt_shard_host']==plan['measurement_host']
    key=lambda r:json.dumps(r,sort_keys=True)
    expected=[r for r in plan['requests'] if r['model'] in plan['groups']]
    assert sharded['requests']==expected
    index=lambda rows:{key(v['request']):v for v in rows}
    replacements=index(sharded['complete']);current=index(controller['complete']);published=index(records)
    assert len(replacements)==len(expected)==len(sharded['complete'])
    assert set(replacements)=={key(r) for r in expected}
    assert len(current)==len(published)==len(plan['requests'])
    devices={};inputs={};cores=set();gpus=set()
    for model,group in plan['groups'].items():
        assert len(group)==4
        for i,lane in enumerate(group):
            assert lane['slice']==[8*i,8] and lane['gpu'] not in gpus and not cores.intersection(lane['cpus'])
            gpus.add(lane['gpu']);cores.update(lane['cpus'])
    for request in expected:
        k=key(request);record=published[k];report=record['report'];group=plan['groups'][request['model']]
        assert record['private_sha256']==current[k]['sha256']==replacements[k]['sha256']
        assert report['measurement_mode']=='matched-prompt-shards' and report['prompt_shard_plan_sha256']==sharded['plan_sha256']
        files=inventory[report['environment']['commit']]['files']
        assert report['aggregation_source_sha256']==files['tools/gpu_prompt_shards.py']['original_sha256']
        assert len(report['shards'])==4
        for shard,lane in zip(report['shards'],group):
            child=shard['report'];source(child['environment'],inventory)
            assert child['measurement_host']==plan['measurement_host'] and child['request_slice']==lane['slice']
            assert child['selected_gpu_selector']==str(lane['gpu']) and child['cpu_affinity']==lane['cpus']
            assert child['job_sha256']==report['job_sha256'] and child['batch_size']==request['batch']
            assert child['mode']==request['mode'] and child['cache_bytes']==request['cache']
            uuid=child['gpu_before']['gpu_uuid'];device=(request['model'],lane['gpu'])
            assert device not in devices or devices[device]==uuid;devices[device]=uuid
            cohort=(request['model'],request['batch'],lane['gpu']);hashes=[v['input_ids_sha256'] for v in child['requests']]
            assert cohort not in inputs or inputs[cohort]==hashes;inputs[cohort]=hashes
            for field,filename in (('phase_source_sha256','benchmark_llm_codec_serving.py'),
                ('cached_decode_source_sha256','cached_memory_decode.py'),('codec_adapter_source_sha256','codec_memory_adapter.py')):
                assert child[field]==inventory[child['environment']['commit']]['files']['tools/'+filename]['original_sha256']
        assert report['unique_prompt_count']==len({p for s in report['shards'] for r in s['report']['requests'] for p in r['prompt_hashes']})
    assert len(set(devices.values()))==8
    assert len(sharded['preflights'])==8
    for preflight in sharded['preflights']:
        lane=next(v for v in plan['groups'][preflight['model']] if v['gpu']==preflight['gpu'])
        a,b=[v['report'] for v in preflight['records']]
        assert a['mode']=='source' and b['mode']=='codec'
        for report in (a,b):
            llm_timing(report,'gpu',request_count=2,repeat=1)
            assert report['selected_gpu_selector']==str(lane['gpu']) and report['cpu_affinity']==lane['cpus']
            assert report['measurement_host']==plan['measurement_host']
            assert report['gpu_before']['gpu_uuid']==devices[(preflight['model'],lane['gpu'])]
        for x,y in zip(a['requests'],b['requests']):
            for field in ('input_ids_sha256','cached_logit_hashes','generated_ids_sha256'):assert x[field]==y[field]
    prior_records=load(root/'llm/pre-sharded-gpu-records.json.gz');prior=index(parent['complete'])
    assert {key(v['request']) for v in prior_records}==set(replacements)
    for record in prior_records:
        k=key(record['request']);assert record['private_sha256']==prior[k]['sha256']
        value=record['report']
        if value.get('measurement_purpose')=='prompt-shard-routing-receipt-only':
            assert value['shard_measurement_sha256']==replacements[k]['sha256']
            assert value['prompt_shard_plan_sha256']==sharded['plan_sha256'] and value['request']==record['request']
    return parent,[r for r in records if r['request']['model'] not in plan['groups']]+prior_records


def verify_extra_shards(root,controller,records,inventory,private):
    plan=load(root/'gpu-extra-shard-plan.json');sharded=load(root/'llm/extra-shard-controller.json')
    parent=load(root/'llm/pre-extra-sharded-gpu-controller.json')
    assert plan['schema_version']==1 and sharded['status']=='complete' and sharded['workers']==4
    assert controller['extra_shard_plan_sha256']==sharded['plan_sha256']==private['gpu-extra-shard-plan.json']
    assert controller['requests']==parent['requests']==plan['requests']
    assert controller['extra_shard_groups']==plan['groups'] and controller['extra_shard_host']==plan['measurement_host']
    key=lambda r:json.dumps(r,sort_keys=True)
    expected=[r for r in plan['requests'] if r['model'] in plan['groups'] and r['batch']==1]
    assert sharded['requests']==expected
    index=lambda rows:{key(v['request']):v for v in rows}
    replacements=index(sharded['complete']);current=index(controller['complete']);published=index(records)
    assert len(replacements)==len(expected)==len(sharded['complete'])
    assert set(replacements)=={key(r) for r in expected}
    assert len(current)==len(published)==len(plan['requests'])
    devices={};inputs={};cores=set();gpus=set()
    for model,group in plan['groups'].items():
        assert len(group)==2
        for i,lane in enumerate(group):
            assert lane['slice']==[16*i,16] and lane['gpu'] not in gpus and not cores.intersection(lane['cpus'])
            gpus.add(lane['gpu']);cores.update(lane['cpus'])
    for request in expected:
        k=key(request);record=published[k];report=record['report'];group=plan['groups'][request['model']]
        assert record['private_sha256']==current[k]['sha256']==replacements[k]['sha256']
        assert report['measurement_mode']=='matched-prompt-shards' and report['extra_shard_plan_sha256']==sharded['plan_sha256']
        files=inventory[report['environment']['commit']]['files']
        assert report['aggregation_source_sha256']==files['tools/gpu_extra_shards.py']['original_sha256']
        assert len(report['shards'])==2
        for shard,lane in zip(report['shards'],group):
            child=shard['report'];source(child['environment'],inventory)
            assert child['measurement_host']==plan['measurement_host'] and child['request_slice']==lane['slice']
            assert child['selected_gpu_selector']==str(lane['gpu']) and child['cpu_affinity']==lane['cpus']
            assert child['job_sha256']==report['job_sha256'] and child['batch_size']==request['batch']
            assert child['mode']==request['mode'] and child['cache_bytes']==request['cache']
            uuid=child['gpu_before']['gpu_uuid'];device=(request['model'],lane['gpu'])
            assert device not in devices or devices[device]==uuid;devices[device]=uuid
            cohort=(request['model'],request['batch'],lane['gpu']);hashes=[v['input_ids_sha256'] for v in child['requests']]
            assert cohort not in inputs or inputs[cohort]==hashes;inputs[cohort]=hashes
            for field,filename in (('phase_source_sha256','benchmark_llm_codec_serving.py'),
                ('cached_decode_source_sha256','cached_memory_decode.py'),('codec_adapter_source_sha256','codec_memory_adapter.py')):
                assert child[field]==inventory[child['environment']['commit']]['files']['tools/'+filename]['original_sha256']
        assert report['unique_prompt_count']==len({p for s in report['shards'] for r in s['report']['requests'] for p in r['prompt_hashes']})
    assert len(set(devices.values()))==4
    assert len(sharded['preflights'])==4
    for preflight in sharded['preflights']:
        lane=next(v for v in plan['groups'][preflight['model']] if v['gpu']==preflight['gpu'])
        a,b=[v['report'] for v in preflight['records']]
        assert a['mode']=='source' and b['mode']=='codec'
        for report in (a,b):
            llm_timing(report,'gpu',request_count=2,repeat=1)
            assert report['selected_gpu_selector']==str(lane['gpu']) and report['cpu_affinity']==lane['cpus']
            assert report['measurement_host']==plan['measurement_host']
            assert report['gpu_before']['gpu_uuid']==devices[(preflight['model'],lane['gpu'])]
        for x,y in zip(a['requests'],b['requests']):
            for field in ('input_ids_sha256','cached_logit_hashes','generated_ids_sha256'):assert x[field]==y[field]
    prior_records=load(root/'llm/pre-extra-sharded-gpu-records.json.gz');prior=index(parent['complete'])
    assert {key(v['request']) for v in prior_records}==set(replacements)
    for record in prior_records:
        k=key(record['request']);assert record['private_sha256']==prior[k]['sha256']
        value=record['report']
        if value.get('measurement_purpose')=='extra-shard-routing-receipt-only':
            assert value['shard_measurement_sha256']==replacements[k]['sha256']
            assert value['extra_shard_plan_sha256']==sharded['plan_sha256'] and value['request']==record['request']
    return parent,[r for r in records if not (r['request']['model'] in plan['groups'] and r['request']['batch']==1)]+prior_records


def verify_long_model(root,prepared,selected,inventory,private_bytes):
    long_coverage=set();long_retimings=set();long_cpu_records=[]
    name = prepared['name']; folder = root / 'long/models' / name
    assert name == selected['name'] and len(prepared['jobs']) == 261
    metadata = load(folder / 'metadata.json'); grid = load(folder / 'grid.json.gz')
    refs = {b['name']: b for b in load(folder / 'paired-blocks.json.gz')}
    arrays=load(folder/'paired-arrays.json.gz')
    for checksum,values in arrays.items():
        assert hashlib.sha256(json.dumps(values,separators=(',',':'),allow_nan=False).encode()).hexdigest()==checksum
    def candidate_blocks(name):
        reference=refs[name];result={}
        for cached,splits in reference['blocks'].items():
            result[cached]={}
            for split,strata in splits.items():
                block={s:{field:arrays[key] for field,key in fields.items()} for s,fields in strata.items()}
                encoded=(json.dumps(block,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
                assert hashlib.sha256(encoded).hexdigest()==reference['private_block_json_sha256'][cached][split]
                result[cached][split]=block
        return result
    assert len(grid) == len(refs) == 261 and len({g['build']['name'] for g in grid}) == 261
    source(metadata['environment'], inventory) if 'environment' in metadata else None
    costs = {}; builds = {}
    for entry in grid:
        built = entry['build']; candidate = built['name']; long_coverage.add((name, candidate))
        blocks=candidate_blocks(candidate)
        source(built['environment'], inventory)
        assert built['status'] == 'complete' and built['job_sha256'] == entry['private_job_sha256']
        assert sum(v['bytes'] for v in built['serving_files'].values()) == built['row_bundle_bytes']
        assert built['model_bytes'] == built['row_bundle_bytes'] + built['common_model_bytes']
        assert entry['complete_serving_bytes'] == built['model_bytes'] + private_bytes['long/models/' + name + '/metadata.json']
        assert selected['costs'][candidate]['complete_serving_bytes'] == entry['complete_serving_bytes']
        assert selected['costs'][candidate]['raw_build_report_sha256'] == entry['private_build_sha256']
        row = entry['row_audit']; assert row['status'] == 'complete' and row['rows'] == metadata['shape'][0]
        assert row['decoded_row_sha256'] == built['decoded_row_sha256']
        assert row['private_build_report_sha256'] == entry['private_build_sha256']
        for cached in ('0', '1'):
            paired(blocks[cached]['validation'], built['validation'][cached], metadata['evaluations']['validation'])
            audit = entry['audits'][cached]['report']; long_timing(audit, metadata)
            assert audit['job_sha256'] == entry['private_job_sha256'] and audit['decoded_row_sha256'] == built['decoded_row_sha256']
            for split in ('test', 'confirmation'):
                paired(blocks[cached][split], audit['quality'][split], metadata['evaluations'][split])
        costs[candidate] = entry['complete_serving_bytes']; builds[candidate] = built
    assert len(selected['choices']) == 2 * len(BUDGETS) * len({v['family'] for v in builds.values()})
    for choice in selected['choices']:
        eligible = [b for b in builds.values() if b['family'] == choice['family'] and gate(b['validation'][str(choice['cached'])], choice['budget'])]
        best = min(eligible, key=lambda b: (costs[b['name']], b['name'])) if eligible else None
        assert choice['name'] == (best['name'] if best else None)
    for interval in load(folder / 'selected-intervals.json.gz'):
        assert (interval['name'], interval['cached']) in {(c['name'], c['cached']) for c in selected['choices']}
        for value in interval['strata'].values():
            intervals(value)
    for timing in load(folder / 'retimings.json.gz'):
        request = timing['request']; key = tuple(request[k] for k in ('model', 'name', 'cached', 'index', 'cache'))
        assert key not in long_retimings; long_retimings.add(key)
        long_timing(timing['report'], metadata)
        long_cpu_records.append({**timing, 'report': cpu_projection(timing['report'])})
    return long_coverage,long_retimings,long_cpu_records


def verify(root,jobs=1):
    manifest = load(root / 'manifest.json')
    names = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    assert names == set(manifest['files'])
    for name, entry in manifest['files'].items():
        path = root / name
        assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'], name
    publication = load(root / 'publication.json'); assert publication['status'] == 'complete'
    assert (publication['long_models'], publication['long_candidates'], publication['long_heldout_workers'],
            publication['llm_models'], publication['llm_candidates'], publication['independent_controls']) == (108, 28188, 56376, 6, 1566, 12)
    private = publication['private_input_sha256']; private_bytes = publication['private_input_bytes']
    inventory = load(root / 'runtime/inventory.json')
    assert set(inventory) == set(publication['source_commits'])
    for commit, entry in inventory.items():
        path = root / 'runtime' / (commit + '.tar.gz'); assert digest(path) == entry['archive_sha256']
        with tarfile.open(path, 'r:gz') as archive:
            assert {m.name for m in archive.getmembers()} == set(entry['files'])
            for member in archive.getmembers():
                value = archive.extractfile(member).read(); expected = entry['files'][member.name]
                assert len(value) == expected['bytes'] and hashlib.sha256(value).hexdigest() == expected['published_sha256']
    for model in ('qwen06','qwen35'):
        check=load(root/'diagnostics'/('real-batch-cache-'+model+'.json'))
        assert check['status']=='passed' and len(check['checks'])==4
        assert check['matched_complete_cached_logits']==68 and check['matched_generated_sequences']==18
        files=inventory[check['source_commit']]['files']
        assert files['tools/check_llm_memory_batches.py']['original_sha256']==check['checker_source_sha256']
        for value in check['checks']:
            assert value['batch'] in (1,8) and value['byte_exact_source_native'] is True
            for field in ('input_ids_sha256','cached_logits_sha256','generated_ids_sha256'):
                assert value['source'][field]==value['native'][field]
    prepare = load(root / 'long/prepare.json'); frozen = load(root / 'long/selection.json')
    complete = load(root / 'long/quality-complete.json')
    assert prepare['status'] == complete['status'] == 'complete' and len(prepare['models']) == len(frozen['models']) == 108
    assert complete['frozen_selection_sha256'] == private['long/selection.json']
    long_coverage = set(); long_retimings = set(); long_cpu_records = []
    from .evidence_parallel import ordered_map
    def verify_pair(pair):return verify_long_model(root,*pair,inventory,private_bytes)
    for coverage,retimings,cpu_records in ordered_map(verify_pair,list(zip(prepare['models'],frozen['models'])),jobs):
        assert not long_coverage.intersection(coverage) and not long_retimings.intersection(retimings)
        long_coverage.update(coverage);long_retimings.update(retimings);long_cpu_records.extend(cpu_records)
    assert len(long_coverage) == 28188
    training = load(root / 'long/training-selection.json'); assert len(training['groups']) == 18
    for group in training['groups']:
        assert len(group['points']) == 6
        best = min(group['points'], key=lambda p: (p['absolute_validation_loss'], -p['absolute_validation_discrimination'], p['name']))
        assert best['name'] == group['chosen'] and best['name'] in {m['name'] for m in prepare['models']}
    controller = load(root / 'long/retiming-controller.json')
    verify_cpu_schedule(root, controller, frozen, long_cpu_records, 'retime-long', 'long/models', inventory, private)
    assert len(controller['requests']) == len(controller['complete']) == len(long_retimings)
    prepare = load(root / 'llm/prepare.json'); frozen = load(root / 'llm/selection.json')
    assert len(prepare['models']) == len(frozen['models']) == 6
    llm_coverage = set(); control_count = 0; case_cache = {}; llm_cpu_records = []
    def case_summary(path):
        if path not in case_cache:
            cases=load(root/path);splits={c['split'] for c in cases}
            case_cache[path]={'splits':splits,'metrics':{s:metrics([c for c in cases if c['split']==s]) for s in splits}}
        return case_cache[path]
    timer_keys = {'cpu': set(), 'gpu': set()};model_devices={};llm_gpu_records=[]
    spare_plan = load(root / 'gpu-spare-lane-plan.json') if (root / 'gpu-spare-lane-plan.json').exists() else None
    shard_plan=load(root/'gpu-prompt-shard-plan.json') if (root/'gpu-prompt-shard-plan.json').exists() else None
    extra_plan=load(root/'gpu-extra-shard-plan.json') if (root/'gpu-extra-shard-plan.json').exists() else None
    for prepared, selected in zip(prepare['models'], frozen['models']):
        name = prepared['name']; folder = root / 'llm/models' / name
        assert name == selected['name']; metadata = load(folder / 'metadata.json')
        grid = load(folder / 'grid.json.gz'); assert len(grid) == 261
        calibration = load(folder / 'calibration-ledger.json'); test = load(folder / 'test-ledger.json')
        assert calibration['status'] == test['status'] == 'complete' and len(calibration['candidates']) == 261
        assert test['frozen_selection_sha256'] == private['llm/selection.json']
        values = {}; identities = {}
        for entry in grid:
            built = entry['build']; candidate = built['name']; llm_coverage.add((name, candidate))
            assert built['status'] == 'complete' and built['job_sha256'] == entry['private_job_sha256']
            assert built['model_bytes'] == built['row_bundle_bytes'] + built['common_model_bytes']
            assert built['row_bundle_bytes'] == sum(v['bytes'] for v in built['serving_files'].values())
            assert entry['complete_serving_bytes'] == built['model_bytes'] + private_bytes['llm/models/' + name + '/metadata.json']
            assert entry['row_audit']['decoded_row_sha256'] == built['decoded_row_sha256']
            for split, quality in entry['quality'].items():
                result = quality['report']; cases = case_summary(quality['public_case_scores'])
                qwen_source(result['environment'], inventory)
                files = inventory[result['environment']['code_sha']]['files']
                assert files['tools/llm_codec_frontier.py']['original_sha256'] == result['environment']['codec_frontier_source_sha256']
                assert files['tools/codec_memory_adapter.py']['original_sha256'] == result['environment']['codec_adapter_source_sha256']
                assert result['status'] == 'complete' and result['job_sha256'] == entry['private_job_sha256']
                assert result['decoded_row_sha256'] == built['decoded_row_sha256'] and len(result['complete_request_checks']) == 16
                assert cases['splits']=={split}
                check_metrics(cases['metrics'][split], result['metrics'])
                assert result['oov_contract_sha256'] == private['oov-contract.json']
                key = (split, result['decoded_row_sha256'])
                if key in identities:
                    leader = identities[key]
                    assert leader['complete_request_checks'] == result['complete_request_checks']
                    check_metrics(leader['metrics'], result['metrics'])
                else:
                    identities[key] = result
            assert 'calibration' in entry['quality']
            values[candidate] = entry
        for entry in grid:
            for split, quality in entry['quality'].items():
                result = quality['report']; reused = result['quality_reused_from']
                if reused:
                    leader = values[reused]['quality'][split]['report']
                    assert leader['quality_reused_from'] is None
                    for field in ('decoded_row_sha256', 'complete_request_checks', 'batch_schedule_sha256',
                                  'nonrow_delta_sha256', 'frozen_weight_sha256', 'oov_contract_sha256'):
                        assert result[field] == leader[field]
                    check_metrics(leader['metrics'], result['metrics'])
        baseline = metric_values(values['raw-fp32']['quality']['calibration']['report']['metrics'])
        for choice in selected['choices']:
            eligible = [v for v in values.values() if v['build']['family'] == choice['family'] and all(
                a - b <= choice['budget_pp'] / 100 for a, b in zip(baseline, metric_values(v['quality']['calibration']['report']['metrics'])))]
            best = min(eligible, key=lambda v: (v['complete_serving_bytes'], v['build']['name'])) if eligible else None
            assert choice['name'] == (best['build']['name'] if best else None)
            if best:
                assert 'test' in best['quality']
        controls = load(folder / 'controls.json.gz'); assert {c['name'] for c in controls} == {'head-constant', 'adapter', 'exact', 'hash'}
        reference = next(c for c in controls if c['name'] == 'exact')['report']
        for control in controls:
            report = control['report']; cases = case_summary(control['public_case_scores'])
            qwen_source(report['environment'], inventory)
            assert report['status'] == 'complete' and report['seed'] == metadata['seed']
            for split in ('calibration', 'test'):
                check_metrics(cases['metrics'][split], report['metrics'][split])
            if control['name'] in ('head-constant', 'adapter'):
                control_count += 1
                assert report['optimizer_steps'] == reference['optimizer_steps'] and len(report['epochs']) == 20
                assert report['prepared_sha256'] == reference['prepared_sha256']
                assert report['frozen_weight_sha256'] == reference['frozen_weight_sha256']
                assert report['frozen_decoder_suffix_sha256'] == reference['frozen_decoder_suffix_sha256']
                assert [e['actual_sequences'] for e in report['epochs']] == [e['actual_sequences'] for e in reference['epochs']]
            if control['name'] == 'adapter':
                assert abs(report['trainable_parameters'] - reference['trainable_parameters']) <= 12000
        for phase in ('cpu', 'gpu'):
            for timing in load(folder / ('retimings-' + phase + '.json.gz')):
                request = timing['request']; key = tuple(request.get(k) for k in ('model', 'name', 'mode', 'cache', 'batch'))
                assert key not in timer_keys[phase]; timer_keys[phase].add(key)
                llm_timing(timing['report'], phase)
                if phase == 'cpu':
                    llm_cpu_records.append({**timing, 'report': cpu_projection(timing['report'])})
                if phase=='gpu':
                    llm_gpu_records.append(timing)
                    report=timing['report'];assignment=load(root/'serving-addendum.json')['device_assignment']
                    if (shard_plan and name in shard_plan['groups']) or (extra_plan and name in extra_plan['groups'] and request['batch']==1):
                        assert report['measurement_mode']=='matched-prompt-shards'
                        continue
                    routed = spare_plan and name in spare_plan['spares'] and request['batch'] == 8
                    assert report['selected_gpu_selector']==str(spare_plan['spares'][name] if routed else assignment[name])
                    assert report['cpu_affinity']==(spare_plan['spare_cpu_affinity'][name] if routed else load(root/'gpu-worker-affinity.json')[name])
                    uuid=report['gpu_before']['gpu_uuid']
                    device_key = (name, request['batch']) if spare_plan else name
                    assert device_key not in model_devices or model_devices[device_key]==uuid
                    model_devices[device_key]=uuid
    assert len(llm_coverage) == 1566 and control_count == 12
    for phase in ('cpu', 'gpu'):
        controller = load(root / ('llm/retiming-' + phase + '-controller.json'))
        assert controller['status'] == 'complete'
        if phase=='cpu':
            verify_cpu_schedule(root, controller, frozen, llm_cpu_records, 'retime-llm-cpu', 'llm/models', inventory, private)
        else:
            assert controller['workers']==(16 if shard_plan else 8 if spare_plan else 6) and controller['workers_per_gpu']==1
            assert controller['gpu_addendum_sha256']==private['serving-addendum.json']
            assert controller['fixed_device_assignment']==load(root/'serving-addendum.json')['device_assignment']
            assert controller['cpu_affinity']==load(root/'gpu-worker-affinity.json')
            cores=set()
            for cpus in controller['cpu_affinity'].values():
                assert cpus and not cores.intersection(cpus);cores.update(cpus)
            parent,records=verify_extra_shards(root,controller,llm_gpu_records,inventory,private) if extra_plan else (controller,llm_gpu_records)
            parent,records=verify_prompt_shards(root,parent,records,inventory,private) if shard_plan else (parent,records)
            if spare_plan: verify_spare_cohort(root, parent, records, inventory, private)
        assert len(controller['requests']) == len(controller['complete']) == len(timer_keys[phase])
    analysis = root / 'analysis/report.json'
    if analysis.exists():
        summary = load(analysis)
        assert summary['status'] == 'complete' and summary['llm_models'] == 6 and summary['long_models'] == 108
        for result in load(root / 'analysis/llm-paired-comparisons.json'):
            if result['status'] != 'complete':
                assert result['status'] == 'some_seeds_have_no_calibration_qualified_candidate'
                continue
            samples = load(root / 'analysis' / result['samples_file'])['samples']
            assert len(samples) == result['replicates'] == 2000 and result['seed_clusters'] == 3
            assert all(len(row) == 4 and all(math.isfinite(v) for v in row) for row in samples)
            for metric in range(4):
                for endpoint, probability in enumerate((.025, .975)):
                    close(result['paired_95_percent_intervals'][metric][endpoint], quantile([row[metric] for row in samples], probability))
    print(json.dumps({'status': 'passed', 'long_candidates': len(long_coverage), 'long_heldout_workers': 56376,
                      'llm_candidates': len(llm_coverage), 'independent_controls': control_count,
                      'long_final_timings': len(long_retimings), 'llm_final_timings': {k: len(v) for k, v in timer_keys.items()}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--jobs',type=int,default=1)
    args = parser.parse_args()
    if not 1<=args.jobs<=64:parser.error('Verification workers must be within 1–64')
    verify(args.root,args.jobs)


if __name__ == '__main__':
    main()
