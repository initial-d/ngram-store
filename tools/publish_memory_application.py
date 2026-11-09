"""Publish the complete finite memory study as numeric, hash-linked evidence.

The exporter rejects missing candidates, quality audits, independent controls,
or final timers. Private vectors, token sequences and key dictionaries are not
part of the allowlist. The archived arithmetic is independently verifiable;
rerunning model training requires the separately identified public inputs.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import io

from .tree_node_quality import BUDGETS, SPLITS, bootstrap, digest, load, select, write
from .finalize_memory_application import long_requests, llm_requests
from .counterfact_data import summarize


class Archive:
    def __init__(self, output, replacements):
        self.output = output
        self.replacements = sorted(replacements.items(), key=lambda p: -len(p[0]))
        self.inputs = {}
        self.input_bytes = {}
        self.output.mkdir(parents=True, exist_ok=False)

    def public(self, value):
        if isinstance(value, dict):
            return {self.public(k): self.public(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.public(v) for v in value]
        if isinstance(value, str):
            for private, placeholder in self.replacements:
                value = value.replace(private, placeholder)
            # This sentinel is split so runtime source itself is portable.
            if '/' + 'llm-align/' in value:
                raise ValueError('Unredacted compute path outside the declared map')
        return value

    def put(self, name, value):
        write(self.output / name, self.public(value))

    def take(self, source, name):
        self.inputs[name] = digest(source)
        self.input_bytes[name] = Path(source).stat().st_size
        self.put(name, load(source))
        return name

    def cases(self, source):
        key = digest(source)
        name = 'cases/' + key + '.json.gz'
        if name not in self.inputs:
            self.take(source, name)
        return name


def check_controller(path, requests):
    report = load(path / 'report.json')
    if report['status'] != 'complete' or report['requests'] != requests:
        raise ValueError('Complete unchanged final timing schedule required: ' + str(path))
    if len(report['complete']) != len(requests):
        raise ValueError('Final timing coverage differs')
    outputs = {}
    for item in report['complete']:
        if digest(item['output']) != item['sha256']:
            raise ValueError('Final timing output changed')
        result = load(item['output'])
        if result['status'] not in ('complete', 'passed') or result['job_sha256'] != digest(item['request']['job']):
            raise ValueError('Final timing provenance changed')
        key = json.dumps(item['request'], sort_keys=True)
        if key in outputs:
            raise ValueError('Duplicate final timing')
        outputs[key] = result
    if set(outputs) != {json.dumps(r, sort_keys=True) for r in requests}:
        raise ValueError('Missing final timing requests')
    return report, outputs


def long_evidence(args, archive):
    root = args.root / 'long-frontier'
    prepare = load(root / 'prepare.json'); frozen = load(root / 'selection.json')
    complete = load(root / 'quality-complete.json')
    rows = load(args.root / 'long-row-audit' / 'report.json')
    if (prepare['status'] != 'complete' or complete['status'] != 'complete' or rows['status'] != 'complete'
            or len(prepare['models']) != 108 or complete['requested_candidates'] != 28188
            or len(rows['complete']) != 28188 or complete['fresh_audit_workers'] != 56376
            or complete['frozen_selection_sha256'] != digest(root / 'selection.json')):
        raise ValueError('Complete 108-point row and held-out audits required')
    for file in ('prepare.json', 'selection.json', 'training-selection.json', 'quality-complete.json'):
        archive.take(root / file, 'long/' + file)
    for source in root.glob('stage-*/*progress.json'):
        archive.take(source, 'long/stages/' + source.parent.name + '-' + source.name)
    archive.take(args.root / 'long-row-audit' / 'report.json', 'long/row-audit-controller.json')
    timer, timings = check_controller(args.root / 'retime-long', long_requests(root))
    archive.take(args.root / 'retime-long' / 'report.json', 'long/retiming-controller.json')
    for prepared, selected in zip(prepare['models'], frozen['models']):
        if prepared['name'] != selected['name'] or len(prepared['jobs']) != 261:
            raise ValueError('Long grid changed')
        folder = 'long/models/' + prepared['name']
        common = Path(load(prepared['jobs'][0])['common'])
        archive.take(common / 'metadata.json', folder + '/metadata.json')
        grid = []; builds = []; all_blocks = []
        for path in prepared['jobs']:
            job = load(path); built = load(job['report'])
            if (built['status'] != 'complete' or built['job_sha256'] != digest(path)
                    or selected['costs'][job['name']]['raw_build_report_sha256'] != digest(job['report'])):
                raise ValueError('Frozen long build changed')
            size = built['model_bytes'] + (common / 'metadata.json').stat().st_size
            if size != selected['costs'][job['name']]['complete_serving_bytes']:
                raise ValueError('Incomplete long serving cost')
            builds.append({**built, 'model_bytes': size})
            audits = {}; blocks = {}
            for cached in (0, 1):
                audit_path = Path(job['report']).with_name(job['name'] + f'-cache{cached}-datrie-rowcache0-audit.json')
                audit = load(audit_path)
                if audit['status'] != 'complete' or audit['job_sha256'] != digest(path):
                    raise ValueError('Incomplete long held-out audit')
                audits[str(cached)] = {'report': audit, 'private_sha256': digest(audit_path)}
                blocks[str(cached)] = {split: load(Path(job['report']).with_name(job['name'] + f'-{split}-cache{cached}-blocks.json.gz')) for split in SPLITS}
            row_path = args.root / 'long-row-audit' / prepared['name'] / (job['name'] + '.json')
            row = load(row_path)
            if row['job_sha256'] != digest(path) or row['private_build_report_sha256'] != digest(job['report']):
                raise ValueError('Long row audit changed')
            grid.append({'job': job, 'private_job_sha256': digest(path), 'build': built,
                         'private_build_sha256': digest(job['report']), 'complete_serving_bytes': size,
                         'audits': audits, 'row_audit': row, 'private_row_audit_sha256': digest(row_path)})
            all_blocks.append({'name': job['name'], 'blocks': blocks})
        for choice in selected['choices']:
            best = select([b for b in builds if b['family'] == choice['family']], choice['budget'], choice['cached'])
            if choice['name'] != (best['name'] if best else None):
                raise ValueError('Validation-only long selection changed')
        archive.put(folder + '/grid.json.gz', grid)
        archive.put(folder + '/paired-blocks.json.gz', all_blocks)
        chosen = {(c['name'], c['cached']) for c in selected['choices'] if c['name']}
        intervals = []
        by_name = {b['name']: b['blocks'] for b in all_blocks}
        for name, cached in sorted(chosen):
            for split in ('test', 'confirmation'):
                intervals.append({'name': name, 'cached': cached, 'split': split,
                    'strata': {s: bootstrap(by_name[name][str(cached)][split][s]) for s in ('all', 'oov_nonempty')}})
        archive.put(folder + '/selected-intervals.json.gz', intervals)
        archive.put(folder + '/retimings.json.gz', [{'request': item['request'], 'report': timings[json.dumps(item['request'], sort_keys=True)],
                                                   'private_sha256': item['sha256']} for item in timer['complete'] if item['request']['model'] == prepared['name']])
        print({'published_long_models': prepared['name']}, flush=True)


def llm_evidence(args, archive):
    root = args.root / 'llm-frontier'; controls = args.root / 'independent-controls'
    prepare = load(root / 'prepare.json'); frozen = load(root / 'selection.json')
    rows = load(args.root / 'llm-row-audit' / 'report.json')
    queue = load(controls / 'queue.json')
    if (prepare['status'] != 'complete' or len(prepare['models']) != 6 or rows['status'] != 'complete'
            or len(rows['complete']) != 1566 or queue['status'] != 'complete' or len(queue['runs']) != 12):
        raise ValueError('Complete six-model rows and twelve independent controls required')
    requests = llm_requests(root, controls)
    gpu_timer, gpu = check_controller(args.root / 'retime-llm-gpu', requests)
    cpu_requests = [r for r in requests if r['mode'] == 'codec' and r['batch'] == 1]
    cpu_timer, cpu = check_controller(args.root / 'retime-llm-cpu', cpu_requests)
    for file in ('prepare.json', 'selection.json'):
        archive.take(root / file, 'llm/' + file)
    for source in root.glob('stage-*/*progress.json'):
        archive.take(source, 'llm/stages/' + source.parent.name + '-' + source.name)
    archive.take(args.root / 'llm-row-audit' / 'report.json', 'llm/row-audit-controller.json')
    archive.take(controls / 'queue.json', 'llm/independent-controls-controller.json')
    for phase in ('cpu', 'gpu'):
        archive.take(args.root / ('retime-llm-' + phase) / 'report.json', 'llm/retiming-' + phase + '-controller.json')
    for prepared, selected in zip(prepare['models'], frozen['models']):
        if prepared['name'] != selected['name'] or len(prepared['jobs']) != 261:
            raise ValueError('Six-model candidate grid changed')
        model = root / prepared['name']; common = model / 'common'; folder = 'llm/models/' + prepared['name']
        metadata = load(common / 'metadata.json'); archive.take(common / 'metadata.json', folder + '/metadata.json')
        ledgers = {s: load(model / (s + '-ledger.json')) for s in ('calibration', 'test')}
        for split, ledger in ledgers.items():
            if ledger['status'] != 'complete' or len(ledger['candidates']) != ledger['requested_candidates']:
                raise ValueError('Incomplete LLM quality')
            archive.take(model / (split + '-ledger.json'), folder + '/' + split + '-ledger.json')
        if ledgers['test']['frozen_selection_sha256'] != digest(root / 'selection.json'):
            raise ValueError('LLM test changed frozen choice')
        grid = []
        for path in prepared['jobs']:
            job = load(path); built = load(job['report']); quality = {}
            if built['status'] != 'complete' or built['job_sha256'] != digest(path):
                raise ValueError('LLM construction changed')
            for split in ledgers:
                entries = [e for e in ledgers[split]['candidates'] if e['name'] == job['name']]
                if not entries:
                    continue
                entry = entries[0]; result = load(entry['quality_report'])
                if digest(entry['quality_report']) != entry['quality_report_sha256'] or result['job_sha256'] != digest(path):
                    raise ValueError('LLM quality report changed')
                if digest(result['case_scores']) != result['case_scores_sha256']:
                    raise ValueError('LLM case scores changed')
                if len(result['complete_request_checks']) != 16 or result['decoded_row_sha256'] != built['decoded_row_sha256']:
                    raise ValueError('Full original-row/model request audit absent')
                cases_file = archive.cases(Path(result['case_scores']))
                if summarize(load(result['case_scores'])) != result['metrics']:
                    raise ValueError('LLM reported metrics differ from all official cases')
                quality[split] = {'report': result, 'public_case_scores': cases_file,
                                  'private_sha256': entry['quality_report_sha256']}
            row_path = args.root / 'llm-row-audit' / prepared['name'] / (job['name'] + '.json')
            row = load(row_path)
            if row['job_sha256'] != digest(path) or row['decoded_row_sha256'] != built['decoded_row_sha256']:
                raise ValueError('LLM all-row audit changed')
            grid.append({'job': job, 'private_job_sha256': digest(path), 'build': built,
                         'private_build_sha256': digest(job['report']), 'quality': quality,
                         'complete_serving_bytes': built['model_bytes'] + (common / 'metadata.json').stat().st_size,
                         'row_audit': row, 'private_row_audit_sha256': digest(row_path)})
        archive.put(folder + '/grid.json.gz', grid)
        base, seed = prepared['name'].split('-exact-s')
        control_records = []
        for kind in ('head-constant', 'adapter', 'exact', 'hash'):
            source = (controls / f'{base}-{kind}-s{seed}' if kind in ('head-constant', 'adapter')
                      else args.sources / 'runs' / f'{base}-20877-{kind}-s{seed}')
            report = load(source / 'report.json'); cases = source / 'case-scores.json'
            if report['status'] != 'complete' or digest(cases) != report['score_artifacts']['case-scores.json']['sha256']:
                raise ValueError('Complete independently trained/source control required')
            if report['prepared_sha256'] != load(common / 'model-recipe.json')['prepared_sha256']:
                raise ValueError('Control dataset differs')
            record = {'name': kind, 'report': report, 'private_report_sha256': digest(source / 'report.json'),
                      'private_report_bytes': (source / 'report.json').stat().st_size,
                      'public_case_scores': archive.cases(cases)}
            if kind == 'hash':
                native = source / 'case-scores-native-oov.json'
                if digest(native) != report['native_oov_control']['case_scores_sha256']:
                    raise ValueError('Open-vocabulary hash control changed')
                record['public_native_oov_case_scores'] = archive.cases(native)
            control_records.append(record)
        archive.put(folder + '/controls.json.gz', control_records)
        for phase, timer, values in (('cpu', cpu_timer, cpu), ('gpu', gpu_timer, gpu)):
            archive.put(folder + '/retimings-' + phase + '.json.gz', [{'request': item['request'],
                'report': values[json.dumps(item['request'], sort_keys=True)], 'private_sha256': item['sha256']}
                for item in timer['complete'] if item['request']['model'] == prepared['name']])
        print({'published_llm_model': prepared['name']}, flush=True)


def runtimes(args, archive):
    inventories = {}
    for commit in args.commits:
        command = ['git', 'ls-tree', '-r', '--name-only', commit, 'tools', 'tests', 'native', 'ngram_store', 'meft_memory', 'src', 'mylib', 'Makefile', 'pyproject.toml', 'setup.py', 'benchmarks']
        names = subprocess.check_output(command, text=True).splitlines()
        inventory = {}
        target = archive.output / 'runtime' / (commit + '.tar.gz'); target.parent.mkdir(exist_ok=True)
        with target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed, tarfile.open(fileobj=compressed, mode='w') as tar:
            for name in sorted(names):
                original = subprocess.check_output(['git', 'show', commit + ':' + name])
                try:
                    public = archive.public(original.decode()).encode()
                except UnicodeDecodeError:
                    public = original
                info = tarfile.TarInfo(name); info.size = len(public); info.mode = 0o644; info.mtime = 0
                tar.addfile(info, io.BytesIO(public))
                inventory[name] = {'original_sha256': hashlib.sha256(original).hexdigest(),
                                   'published_sha256': hashlib.sha256(public).hexdigest(), 'bytes': len(public)}
        inventories[commit] = {'files': inventory, 'archive_sha256': digest(target)}
    archive.put('runtime/inventory.json', inventories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'sources', 'output', 'path-map'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--commits', nargs='+', required=True)
    args = parser.parse_args()
    ready = load(args.root / 'quality-ready.json')
    if ready['status'] != 'quality-complete':
        raise ValueError('All-quality readiness required')
    archive = Archive(args.output, load(args.path_map))
    archive.take(args.root / 'quality-ready.json', 'quality-ready.json')
    archive.take(Path('benchmarks/memory_application_frontier_v1.json'), 'protocol.json')
    archive.take(Path('benchmarks/memory_application_oov_contract_v1.json'), 'oov-contract.json')
    archive.take(Path('benchmarks/memory_application_serving_addendum_v1.json'), 'serving-addendum.json')
    archive.take(args.root / 'gpu-worker-affinity.json', 'gpu-worker-affinity.json')
    long_evidence(args, archive)
    llm_evidence(args, archive)
    runtimes(args, archive)
    for source in args.root.glob('serving-smoke-*.json'):
        archive.take(source, 'diagnostics/' + source.name)
    for name in ('quality-dispatch.json', 'cpu-quality-dispatch.json', 'final-serving-dispatch.json'):
        archive.take(args.root / name, 'diagnostics/' + name)
    for source in (args.root / 'cache-preflight-qwen06/report.json', args.root / 'cache-preflight-qwen35/report.json'):
        archive.take(source, 'diagnostics/' + source.parent.name + '.json')
    archive.put('publication.json', {'status': 'complete', 'long_models': 108, 'long_candidates': 28188,
        'long_heldout_workers': 56376, 'llm_models': 6, 'llm_candidates': 1566, 'independent_controls': 12,
        'source_commits': args.commits, 'private_input_sha256': archive.inputs, 'private_input_bytes': archive.input_bytes,
        'disclosure': 'Numeric results, complete official case scores and paired token-block sufficient statistics; lexical keys, token sequences and weights excluded. Runtime path placeholders retain original and published source hashes.'})
    write(archive.output / 'manifest.json', {'files': {str(p.relative_to(archive.output)):
        {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(archive.output.rglob('*')) if p.is_file()},
        'status': 'complete', 'schema_version': 1})


if __name__ == '__main__':
    main()
