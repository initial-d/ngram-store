"""Spare GPUs must accelerate whole matched cohorts and retain prior evidence."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.gpu_spare_lanes import command, cohort_requests, effective_cohort, route_pending, digest, check_result
from tools.verify_memory_application import verify_spare_cohort


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    requests = []
    for model in ('a', 'b'):
        for batch in (1, 8):
            for name, mode in (('source-gpu', 'source'), ('native-none-fp32', 'codec')):
                job = root / model / (name + '.json'); save(job, {'name': name})
                requests.append(dict(model=model, batch=batch, job=str(job), name=name, mode=mode, cache=0))
    return dict(schema_version=1, requests=requests, spares={'b': 7}, original_devices={'a': 0, 'b': 1},
                cpu_affinity={'a': [0, 128], 'b': [1, 129]}, spare_cpu_affinity={'b': [2, 130]},
                canonical_directory=str(root / 'retime-llm-gpu'), upstream=str(root / 'upstream'))


def result(request, gpu, cpus):
    return dict(status='passed', job_sha256=digest(request['job']), selected_gpu_selector=str(gpu),
                cpu_affinity=cpus, batch_size=request['batch'], decode_steps=16, mode=request['mode'],
                cache_bytes=request['cache'], gpu_before={'compute_pids': []},
                gpu_samples=[{'external_compute_pids': []}])


class SpareLaneTests(unittest.TestCase):
    def test_commands_preserve_all_budgets_and_independent_control(self):
        request = dict(job='job', mode='independent', cache=42, batch=8, control_run='control')
        cmd = command(request, 'output', 'upstream')
        for key, value in (('requests', '32'), ('repeat', '3'), ('warmup', '1'), ('decode-steps', '16'),
                           ('control-run', 'control'), ('cache-bytes', '42'), ('batch-size', '8')):
            self.assertEqual(cmd[cmd.index('--' + key) + 1], value)
        self.assertIn('--skip-spare-routing', cmd)

    def test_effective_cohort_replaces_every_registered_batch8_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = fixture(Path(temp)); selected = [r for _, r in cohort_requests(plan)]
            original = dict(status='complete', requests=plan['requests'], complete=[
                {'request': r, 'sha256': 'old', 'gpu': plan['original_devices'][r['model']]} for r in plan['requests']])
            spare = dict(status='complete', requests=selected, complete=[
                {'request': r, 'sha256': 'new', 'gpu': 7} for r in selected])
            value = effective_cohort(plan, original, spare)
            for item in value['complete']:
                self.assertEqual(item['sha256'], 'new' if item['request'] in selected else 'old')
            self.assertTrue(all(i['sha256'] == 'old' for i in original['complete']))
            spare['complete'].pop()
            with self.assertRaisesRegex(ValueError, 'Incomplete'): effective_cohort(plan, original, spare)

    def test_duplicate_samples_and_overlapping_hardware_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = fixture(Path(temp)); plan['spares']['b'] = 0
            with self.assertRaisesRegex(ValueError, 'disjoint'): cohort_requests(plan)
            plan['spares']['b'] = 7; plan['spare_cpu_affinity']['b'] = [1, 129]
            with self.assertRaisesRegex(ValueError, 'CPU affinity'): cohort_requests(plan)

    def test_check_rejects_changed_cache_mode_device_or_external_process(self):
        with tempfile.TemporaryDirectory() as temp:
            request = fixture(Path(temp))['requests'][0]; value = result(request, 0, [0])
            check_result(value, request, 0, [0])
            for field, changed in (('mode', 'codec'), ('cache_bytes', 1), ('selected_gpu_selector', '7')):
                invalid = {**value, field: changed}
                with self.assertRaises(ValueError): check_result(invalid, request, 0, [0])
            value['gpu_samples'][0]['external_compute_pids'] = [123]
            with self.assertRaisesRegex(ValueError, 'isolated'): check_result(value, request, 0, [0])

    def test_public_verifier_rejects_mixed_cohorts_and_missing_preflights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); plan = fixture(root)
            plan['original_devices'] = {m: i for i, m in enumerate('abcdef')}
            plan['cpu_affinity'] = {m: [i, i + 128] for i, m in enumerate('abcdef')}
            plan['spares'] = {'a': 6, 'b': 7}; plan['spare_cpu_affinity'] = {'a': [6, 134], 'b': [7, 135]}
            original = dict(status='complete', requests=plan['requests'], complete=[
                {'request': r, 'sha256': 'original', 'gpu': plan['original_devices'][r['model']]} for r in plan['requests']],
                fixed_device_assignment=plan['original_devices'], cpu_affinity=plan['cpu_affinity'])
            expected = [r for _, r in cohort_requests(plan)]
            spare = dict(status='complete', requests=expected, complete=[
                {'request': r, 'sha256': 'spare', 'gpu': plan['spares'][r['model']]} for r in expected],
                workers=2, workers_per_gpu=1, plan_sha256='plan')
            controller = effective_cohort(plan, original, spare)
            controller.update(gpu_spare_plan_sha256='plan', original_controller_sha256='original-controller',
                              spare_controller_sha256='spare-controller')
            save(root / 'gpu-spare-lane-plan.json', plan)
            save(root / 'llm/original-gpu-controller.json', original)
            save(root / 'llm/spare-gpu-controller.json', spare)
            records = []
            for item in controller['complete']:
                r = item['request']; routed = r in expected
                report = result(r, item['gpu'], (plan['spare_cpu_affinity'] if routed else plan['cpu_affinity'])[r['model']])
                report.update(environment={'commit': 'fixture'}, phase_source_sha256='hash',
                              cached_decode_source_sha256='hash', codec_adapter_source_sha256='hash')
                records.append(dict(request=r, report=report, private_sha256=item['sha256']))
            preflights = []
            for model, gpu in plan['spares'].items():
                reports = [result(r, gpu, plan['spare_cpu_affinity'][model]) for r in expected if r['model'] == model]
                for report in reports:
                    report['requests'] = [dict(input_ids_sha256='input', cached_logit_hashes=['logits'], generated_ids_sha256='ids')] * 2
                    report['gpu_before']['utilization_percent'] = 0
                preflights.append(dict(model=model, gpu=gpu, status='passed', reports=reports))
            def compressed(name, values):
                path = root / 'llm' / name
                with gzip.open(path, 'wt') as stream: json.dump(values, stream)
            compressed('spare-gpu-preflights.json.gz', preflights)
            compressed('original-batch8-diagnostics.json.gz', [dict(request=r, private_sha256='original') for r in expected])
            inventory = {'fixture': {'files': {'tools/' + n: {'original_sha256': 'hash'} for n in
                ('benchmark_llm_codec_serving.py', 'cached_memory_decode.py', 'codec_memory_adapter.py')}}}
            private = {'gpu-spare-lane-plan.json': 'plan', 'llm/original-gpu-controller.json': 'original-controller',
                       'llm/spare-gpu-controller.json': 'spare-controller'}
            with patch('tools.verify_memory_application.source'):
                verify_spare_cohort(root, controller, records, inventory, private)
                invalid = copy.deepcopy(records); invalid[2]['private_sha256'] = 'original'
                with self.assertRaises(AssertionError): verify_spare_cohort(root, controller, invalid, inventory, private)
                compressed('spare-gpu-preflights.json.gz', preflights[:1])
                with self.assertRaises(AssertionError): verify_spare_cohort(root, controller, records, inventory, private)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'Optional NumPy for runtime routing')
    def test_routing_drains_batch1_serially_retains_existing_samples_and_copies_spare(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); plan = fixture(root); save(root / 'gpu-spare-lane-plan.json', plan)
            first, second = [(i, r) for i, r in enumerate(plan['requests']) if r['model'] == 'b' and r['batch'] == 1]
            retained = root / 'retime-llm-gpu' / f'{first[0]:05d}.json'
            save(retained, result(first[1], 1, plan['cpu_affinity']['b'])); retained_bytes = retained.read_bytes()
            i, request = next((i, r) for i, r in enumerate(plan['requests']) if r['model'] == 'b' and r['batch'] == 8)
            target = root / 'gpu-spare-lanes/outputs' / f'{i:05d}.json'
            save(target, result(request, 7, plan['spare_cpu_affinity']['b']))
            args = SimpleNamespace(output=root / 'retime-llm-gpu' / f'{i:05d}.json', batch_size=8,
                job=Path(request['job']), mode=request['mode'], cache_bytes=0,
                requests=32, repeat=3, warmup=1, decode_steps=16)
            seen = []
            def fake(cmd, env, stdout, stderr):
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1'); self.assertIn('--skip-spare-routing', cmd)
                output = Path(cmd[cmd.index('--output') + 1]); seen.append(output.name)
                save(output, result(second[1], 1, plan['cpu_affinity']['b']))
                return SimpleNamespace(returncode=0)
            with patch('os.sched_getaffinity', return_value=set(plan['cpu_affinity']['b']), create=True), \
                 patch('tools.gpu_spare_lanes.subprocess.run', side_effect=fake):
                self.assertTrue(route_pending(args))
                self.assertEqual(seen, [f'{second[0]:05d}.json'])
                self.assertEqual(retained.read_bytes(), retained_bytes)
                receipt = json.loads(args.output.read_text())
                self.assertEqual(receipt['measurement_purpose'], 'routing-receipt-only')
                self.assertEqual(receipt['spare_measurement_sha256'], digest(target))
                self.assertEqual(receipt['cpu_affinity'], plan['cpu_affinity']['b'])
                self.assertEqual(receipt['measurement_cpu_affinity'], plan['spare_cpu_affinity']['b'])
                self.assertTrue(route_pending(args))
                args.output.write_text('{"existing":"evidence"}')
                with self.assertRaisesRegex(ValueError, 'never be overwritten'): route_pending(args)
                self.assertEqual(args.output.read_text(), '{"existing":"evidence"}')
                args.batch_size = 1
                self.assertFalse(route_pending(args))


if __name__ == '__main__': unittest.main()
