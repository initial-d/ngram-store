"""Stdlib verification of complete archived arithmetic and frozen choices.

Does not recreate private weights, lexical queries, training or native assertions.
Bootstrap sample quantiles are checked; NumPy random draws are not regenerated.
"""
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import tarfile
from functools import lru_cache

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT.parents[1] / 'artifacts/tree-node-quality-20261109/raw'
BUDGETS = (.0001, .001, .005, .01, .02)
FAMILIES = ('raw', 'dense', 'constant', 'native-identity', 'tree-datrie',
            'tree-reference', 'faiss', 'faiss-dedup')
STRATA = ('all', 'known', 'oov_nonempty', 'oov_empty')
SPLITS = ('validation', 'test', 'confirmation')


def load(path):
    path = resolve_evidence(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())


def digest(path):
    path = resolve_evidence(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def resolve_evidence(path):
    if path.exists():
        return path
    try:
        return RAW_ROOT / path.relative_to(ROOT)
    except ValueError:
        return path


@lru_cache(maxsize=None)
def snapshot_digest(root, commit, source):
    with tarfile.open(root / 'runtime' / (commit + '.tar.gz'), 'r:gz') as archive:
        return hashlib.sha256(archive.extractfile(source).read()).hexdigest()


def close(a, b, tol=1e-10):
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def quantile(values, p):
    values = sorted(values)
    at = (len(values) - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] + (at - lo) * (values[hi] - values[lo])


def gate(q, budget):
    return all(q[s]['pairs'] > 0 and q[s]['loss_relative_change'] <= budget
               and q[s]['discrimination_change'] >= -.005 for s in ('all', 'oov_nonempty'))


def env(e, root):
    assert e['device'] == 'cpu' and e['gpu'] == 'unused' and e['threads'] == 1
    assert e['numpy'] and e['torch'] and len(e['loadavg']) == 3
    assert len(e['commit']) == 40
    source = ('tools/train_tree_node_mechanism.py' if 'train_tree_node_mechanism' in e['command']
              else 'tools/tree_node_quality.py')
    if source == 'tools/tree_node_quality.py':
        assert e['faiss']
    assert e['source_sha256'] == snapshot_digest(root, e['commit'], source), (e['command'], source)
    assert '/llm-align/' not in e['command']


def intervals(value):
    if value['observed_blocks'] == 0:
        assert value['interval'] is None
        return
    assert value['repeats'] == len(value['samples']) == 2000
    assert value['seed'] == 42 and value['block_tokens'] == 128
    for i, field in enumerate(('loss_relative_change_95_percent_interval', 'discrimination_change_95_percent_interval')):
        assert all(len(s) == 2 and all(math.isfinite(x) for x in s) for s in value['samples'])
        for j, p in enumerate((.025, .975)):
            close(value[field][j], quantile([s[i] for s in value['samples']], p))


def paired(blocks, q, total):
    assert set(blocks) == set(STRATA)
    for s, b in blocks.items():
        n = len(b['block_ids'])
        assert b['block_ids'] == sorted(set(b['block_ids']))
        assert all(0 <= bid <= (total['tokens'] - 1) // 128 for bid in b['block_ids'])
        assert all(len(v) == n for v in b.values())
        assert all(isinstance(c, int) and c > 0 for c in b['count'])
        assert all(math.isfinite(v) for k, vs in b.items() for v in vs)
        size = sum(b['count']); assert q[s]['pairs'] == size
        if size:
            a, ref = sum(b['candidate_loss']), sum(b['baseline_loss'])
            win, refwin = sum(b['candidate_discrimination']), sum(b['baseline_discrimination'])
            close(q[s]['sgns_loss_per_pair'], a / size)
            close(q[s]['baseline_sgns_loss_per_pair'], ref / size)
            close(q[s]['context_discrimination'], win / size)
            close(q[s]['baseline_context_discrimination'], refwin / size)
            close(q[s]['loss_relative_change'], a / ref - 1)
            close(q[s]['discrimination_change'], (win - refwin) / size)
            assert all(0 <= w <= c for w, c in zip(b['candidate_discrimination'], b['count']))
        else:
            assert q[s]['loss_relative_change'] is q[s]['discrimination_change'] is None
    for k in ('count', 'candidate_loss', 'baseline_loss', 'candidate_discrimination', 'baseline_discrimination'):
        parts = {}
        for s in STRATA[1:]:
            for bid, value in zip(blocks[s]['block_ids'], blocks[s][k]):
                parts[bid] = parts.get(bid, 0) + value
        assert set(parts) == set(blocks['all']['block_ids'])
        for bid, value in zip(blocks['all']['block_ids'], blocks['all'][k]):
            close(value, parts[bid], tol=1e-9)
    assert q['all']['pairs'] == total['pairs']


def timing(a, metadata):
    assert a['status'] == 'complete'
    for field in ('memory_before_open', 'memory_after_query'):
        m = a[field]
        assert m['peak_bytes'] > 0 and m['rss_bytes'] >= m['anonymous_bytes'] >= 0
    assert a['load_seconds'] >= 0
    for name, q in a['query'].items():
        assert q['inputs'] == metadata['query_types'][name] and 0 < q['inputs'] <= 512
        assert len(q['complete_samples_ms']) == len(q['replay_samples_ms']) == 15
        assert len(q['batch_complete_samples_ms']) == 15 * math.ceil(q['inputs'] / 64)
        assert all(v > 0 for k in ('complete_samples_ms', 'replay_samples_ms', 'batch_complete_samples_ms') for v in q[k])
        close(q['complete_p50_ms'], statistics.median(q['complete_samples_ms']))
        close(q['complete_p95_ms'], quantile(q['complete_samples_ms'], .95))
        close(q['replay_p50_ms'], statistics.median(q['replay_samples_ms']))
        close(q['batch_p95_ms'], quantile(q['batch_complete_samples_ms'], .95))
        close(q['inputs_per_second'], q['inputs'] * 1000 / q['complete_p50_ms'])
        if a['cached_known_words'] and name == 'known':
            assert q['known_cache_hits'] == q['inputs'] and q['reducer_occurrences'] == 0
        assert len(q['output_sha256']) == 64


def verify(root=ROOT):
    manifest = load(root / 'manifest.json')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    assert actual == set(manifest['files'])
    for name, value in manifest['files'].items():
        p = root / name
        assert p.stat().st_size == value['bytes'] and digest(p) == value['sha256'], name
    raw_manifest = load(root / 'raw-manifest.json')
    members = [n for a in raw_manifest['assets'] for n in a['members']]
    assert len(members) == len(set(members)) and set(members) == set(raw_manifest['files'])
    for name, value in raw_manifest['files'].items():
        p = resolve_evidence(root / name)
        assert p.is_file(), 'Run fetch_raw.py first: ' + name
        assert p.stat().st_size == value['bytes'] and digest(p) == value['sha256'], name
    for commit, inventory in load(root / 'runtime/inventory.json').items():
        archive_path = root / 'runtime' / (commit + '.tar.gz')
        assert digest(archive_path) == inventory['archive_sha256']
        with tarfile.open(archive_path, 'r:gz') as archive:
            files = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                     for m in archive.getmembers() if m.isfile()}
        assert files == inventory['files']
    for item in load(root / 'auxiliary/source-path-redactions.json')['archives']:
        assert len(item['original_archive_sha256']) == 64
        assert digest(root / item['archive']) == item['published_archive_sha256']
        with tarfile.open(root / item['archive'], 'r:gz') as archive:
            for record in item['files']:
                assert len(record['original_sha256']) == 64
                data = archive.extractfile(record['file']).read()
                assert hashlib.sha256(data).hexdigest() == record['published_sha256']
                assert b'/llm-align/' not in data.replace(b"'/llm-align/'", b'')
    recovery = load(root / 'retiming-recovery.json')
    assert recovery['native_source_sha256'] == snapshot_digest(root, recovery['source_commit'], 'native/meft_structural.cc')
    assert recovery['regression_tests']['run'] == 10 and recovery['regression_tests']['failures'] == recovery['regression_tests']['errors'] == recovery['regression_tests']['skipped'] == 0
    codec_metadata = {(v['model'], v['name']): v for v in load(root / 'faiss-codec-metadata.json')}
    assert len(codec_metadata) == 18 * 38
    retiming_records = {tuple(v['request'][k] for k in ('model', 'name', 'cached', 'index', 'cache')): v
                       for v in load(root / 'retiming-record-integrity.json')}
    prepare, selection, complete, publication = [load(root / x) for x in
        ('prepare.json', 'selection.json', 'complete.json', 'publication.json')]
    raw_hashes = publication['private_input_sha256']
    assert raw_hashes['selection.json'] == complete['frozen_selection_sha256']
    assert complete['status'] == prepare['status'] == publication['status'] == 'complete'
    assert complete['models'] == len(prepare['models']) == len(selection['models']) == 18
    assert complete['requested_candidates'] == publication['candidates'] == 3672
    assert complete['fresh_audit_workers'] == publication['audit_workers'] == 7344
    original = load(root.parent / 'cross_corpus_evidence_20261102/training-report.json')
    assert prepare['source_training_report_sha256'] == digest(root.parent / 'cross_corpus_evidence_20261102/training-report.json')
    assert prepare['corpus_zip_sha256'] == original['corpus_zip_sha256']
    assert prepare['confirmation_token_range'] == [200000, 300000]
    assert selection['selection_uses'] == 'validation only'
    integrity = load(root / 'input-integrity.json')
    assert integrity['status'] == 'complete' and len(integrity['models']) == 18
    assert integrity['source_prepare_report_sha256'] == raw_hashes['prepare.json']
    assert integrity['source_training_report_sha256'] == prepare['source_training_report_sha256']
    env(integrity['environment'], root)
    assert integrity['phase_source_sha256'] == snapshot_digest(root, integrity['environment']['commit'], 'tools/audit_tree_node_inputs.py')
    input_models = {m['name']: m for m in integrity['models']}
    for e in (prepare['environment'], selection['environment'], publication['environment']):
        env(e, root)
    coverage, final_requests, row_reports, summary = set(), set(), {}, []
    for prepared, chosen in zip(prepare['models'], selection['models']):
        name = prepared['name']; folder = root / 'models' / name
        assert name == chosen['name']
        meta = load(folder / 'metadata.json')
        audited = input_models[name]
        assert audited['metadata_sha256'] == raw_hashes['models/' + name + '/metadata.json']
        assert audited['checkpoint_sha256'] == meta['source_checkpoint_sha256']
        assert audited['canonical_cached_words_byte_exact'] is audited['training_occurrence_weights_verified'] is True
        assert audited['known_words_verified'] >= meta['query_types']['known']
        assert audited['structural_library_sha256'] == recovery['compiled_library_sha256']
        assert audited['both_indexes_all_ids_verified'] == meta['shape'][0]
        assert audited['cached_word_serving_file']['bytes'] == meta['known_cache_file_bytes']
        assert len(audited['cached_word_serving_file']['sha256']) == 64
        assert audited['fit_original_ids_sha256'] == meta['fit_original_ids_sha256']
        assert audited['query_type_samples_verified'] == meta['query_types']
        for split, value in audited['evaluations'].items():
            assert value['arrays_and_source_scores_byte_exact'] is True
            assert value['pairs'] == meta['evaluations'][split]['pairs']
            assert value['evaluation_sha256'] == meta['evaluations'][split]['sha256']
        cost = load(folder / 'complete-cost.json')
        assert cost['reader_metadata_file']['sha256'] == raw_hashes['models/' + name + '/metadata.json']
        assert cost['reader_metadata_file']['bytes'] > 0
        assert cost['adjusted_common_model_bytes'] == meta['common_model_bytes'] + cost['reader_metadata_file']['bytes']
        assert prepared['metadata_sha256'] == raw_hashes['models/' + name + '/metadata.json']
        old = next(r for r in original['runs'] if all(r[k] == meta[k] for k in ('convention', 'order', 'seed')))
        assert old['checkpoint_sha256'] == meta['source_checkpoint_sha256']
        identity = (meta['convention'], meta['order'], meta['seed'])
        assert identity not in coverage; coverage.add(identity)
        assert meta['known_cache_file_bytes'] == meta['common_serving_files']['word-rows.npy']['bytes']
        assert meta['common_model_bytes'] == sum(v['bytes'] for v in meta['common_serving_files'].values())
        assert meta['index_bytes']['datrie'] == meta['common_serving_files']['keys.datrie']['bytes']
        assert [meta['evaluations'][s]['tokens'] for s in SPLITS] == [20000, 20000, 100000]
        grid = load(folder / 'grid.json.gz'); assert len(grid) == 204
        by_name = {g['build']['name']: g for g in grid}; assert len(by_name) == 204
        rows = load(folder / 'row-fidelity.json.gz')
        assert len(rows) == 204 and {r['report']['name'] for r in rows} == set(by_name)
        for item in rows:
            r = item['report']; build = by_name[r['name']]['build']
            env(r['environment'], root)
            assert r['status'] == 'complete' and r['rows'] == meta['shape'][0]
            assert r['private_build_report_sha256'] == by_name[r['name']]['private_build_sha256']
            assert r.get('construction_decoded_row_sha256', r['decoded_row_sha256']) == build['decoded_row_sha256']
            assert r['job_sha256'] == build['job_sha256']
            close(r.get('construction_relative_frobenius', r['relative_l2']['global']), build['stored_row_relative_frobenius'])
            if r.get('thread_rounding'):
                drift = r['thread_rounding']
                assert build['encoding'].startswith('opq-')
                assert drift['construction_omp_threads'] == 4 and drift['query_omp_threads'] == 1
                assert drift['construction_reconstruction_byte_exact_to_build'] is drift['serialized_serving_files_sha256_verified'] is True
                assert drift['maximum_absolute_difference'] > 0 and drift['relative_frobenius_difference'] > 0
                assert r['decoded_row_sha256'] != build['decoded_row_sha256']
                assert r['phase_source_sha256'] == snapshot_digest(root, r['environment']['commit'], 'tools/audit_tree_node_rows.py')
                changes = []
                for c in (0, 1):
                    q = r['validation_at_query_threads'][str(c)]
                    paired(r['validation_at_query_threads_paired_blocks'][str(c)], q, meta['evaluations']['validation'])
                    for budget in BUDGETS:
                        before, after = gate(build['validation'][str(c)], budget), gate(q, budget)
                        if before != after:
                            changes.append({'cached': c, 'budget': budget, 'construction_pass': before, 'query_pass': after})
                assert changes == drift['validation_gate_changes']
            else:
                assert r['decoded_row_sha256'] == build['decoded_row_sha256']
            assert 0 <= r['undefined_zero_norm_rows'] <= r['rows']
            assert 0 <= r['relative_l2']['p50'] <= r['relative_l2']['p95'] <= r['relative_l2']['maximum']
            if r['cosine']['minimum'] is not None:
                assert -1.00000001 <= r['cosine']['minimum'] <= r['cosine']['p05'] <= r['cosine']['p50'] <= 1.00000001
                assert 0 <= r['norm_ratio']['minimum'] <= r['norm_ratio']['p50'] <= 1.00000001
            if build['family'].startswith('tree-'):
                assert 0 <= r['post_encoding_rows_outside_original_joint_gate'] <= r['rows']
                if build['encoding'] == 'fp32':
                    assert r['post_encoding_rows_outside_original_joint_gate'] == 0
            if build['encoding'] == 'fp32' and build['family'] in ('raw', 'dense', 'native-identity'):
                assert r['source_row_bytes_preserved'] is True and r['relative_l2']['maximum'] == 0
            row_reports[name, r['name']] = item['private_report_sha256']
        assert {f: sum(g['build']['family'] == f for g in grid) for f in FAMILIES} == {
            'raw': 3, 'dense': 1, 'constant': 3, 'native-identity': 9,
            'tree-datrie': 75, 'tree-reference': 75, 'faiss': 19, 'faiss-dedup': 19}
        selected_pairs = {(c['name'], c['cached']) for c in chosen['choices'] if c['name']}
        ci = load(folder / 'selected-intervals.json.gz')
        assert {(x['name'], x['cached'], x['split']) for x in ci} == {
            (n, c, s) for n, c in selected_pairs for s in ('test', 'confirmation')}
        assert len(ci) == len(selected_pairs) * 2
        for record in ci:
            for value in record['strata'].values():
                intervals(value)
        assert len(chosen['choices']) == 80
        assert {(c['cached'], c['budget'], c['family']) for c in chosen['choices']} == {
            (c, b, f) for c in (0, 1) for b in BUDGETS for f in FAMILIES}
        for c in chosen['choices']:
            candidates = [g['build'] for g in grid if g['build']['family'] == c['family']
                          and gate(g['build']['validation'][str(c['cached'])], c['budget'])]
            best = min(candidates, key=lambda b: (b['model_bytes'], b['name'])) if candidates else None
            assert c['name'] == (best['name'] if best else None)
            assert c['model_bytes'] == (best['model_bytes'] if best else None)
            if c['budget'] == .001 and c['name']:
                g = by_name[c['name']]
                summary.append({'model': name, **c, 'test_pass': gate(g['audits'][str(c['cached'])]['quality']['test'], c['budget']),
                    'confirmation_pass': gate(g['audits'][str(c['cached'])]['quality']['confirmation'], c['budget'])})
        for g in grid:
            b = g['build']; n = b['name']
            assert chosen['candidate_report_sha256'][n] == g['private_build_sha256']
            assert b['source_checkpoint_sha256'] == meta['source_checkpoint_sha256']
            assert b['status'] == 'complete' and b['decoded_source_ids_verified'] == meta['shape'][0]
            assert b['row_bundle_bytes'] == sum(v['bytes'] for v in b['serving_files'].values())
            assert b['model_bytes'] == b['row_bundle_bytes'] + b['common_model_bytes']
            assert g['complete_serving_bytes'] == b['row_bundle_bytes'] + cost['adjusted_common_model_bytes']
            assert b['common_model_bytes'] == meta['common_model_bytes']
            env(b['environment'], root)
            if b['family'] in ('faiss', 'faiss-dedup'):
                codec = codec_metadata[name, n]
                assert codec['private_metadata_sha256'] == b['serving_files']['codec.json']['sha256']
                assert codec['metadata']['training_threads'] == 4 and codec['metadata']['seed'] == 42
            if b['family'].startswith('tree-'):
                assert b['cosine'] in (.99, .98, .97, .95, .9) and b['norm_ratio'] in (.99, .98, .97, .95, .9)
                f = b['planning']['fidelity']
                assert f['minimum_cosine'] >= b['cosine'] - 1e-7
                assert f['max_relative_norm_change'] <= 1 / b['norm_ratio'] - 1 + 1e-7
            for cached, a in g['audits'].items():
                assert a['job_sha256'] == b['job_sha256']
                assert a['cached_known_words'] == int(cached) and a['index_kind'] == 'datrie' and a['native_cache_bytes'] == 0
                env(a['environment'], root); timing(a, meta)
        seen = set()
        with gzip.open(resolve_evidence(folder / 'paired-blocks.jsonl.gz'), 'rt') as stream:
            for line in stream:
                record = json.loads(line); n = record['name']
                assert n not in seen; seen.add(n); g = by_name[n]
                for cached, splits in record['blocks'].items():
                    assert set(splits) == set(SPLITS)
                    for split, blocks in splits.items():
                        q = g['build']['validation'][cached] if split == 'validation' else g['audits'][cached]['quality'][split]
                        paired(blocks, q, meta['evaluations'][split])
                    for x in ci:
                        if x['name'] == n and str(x['cached']) == cached:
                            assert x['private_paired_blocks_sha256'] == record['private_block_sha256'][cached][x['split']]
                            for s, value in x['strata'].items():
                                assert value['observed_blocks'] == len(splits[x['split']][s]['count'])
        assert seen == set(by_name)
        retimed = load(folder / 'retimings.json.gz'); seen = set()
        expected_pairs = selected_pairs | {(n, c) for n in ('raw-fp32', 'raw-fp16', 'raw-int8', 'dense-fp32',
            'native-none-fp32', 'native-none-fp16', 'native-none-int8', 'constant-zero', 'constant-mean', 'constant-weighted-mean') for c in (0, 1)}
        expected = {(n, c, index, cache) for n, c in expected_pairs for index in ('datrie', 'hash')
                    for cache in ((0, 1048576) if n.startswith(('native-', 'tree-')) else (0,))}
        for r in retimed:
            identity = (r['name'], r['cached_known_words'], r['index_kind'], r['native_cache_bytes'])
            assert identity not in seen; seen.add(identity); final_requests.add((name, *identity))
            assert r['job_sha256'] == by_name[r['name']]['build']['job_sha256']
            env(r['environment'], root); timing(r, meta)
            if r['index_kind'] == 'hash' or r.get('structural_library_sha256'):
                assert r['structural_library_sha256'] == recovery['compiled_library_sha256']
            original_record = retiming_records[(name, *identity)]
            assert original_record['public_record_sha256'] == hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        reference_outputs = {(r['name'], r['cached_known_words'], s): q['output_sha256']
                             for r in retimed if r['index_kind'] == 'datrie' and r['native_cache_bytes'] == 0
                             for s, q in r['query'].items()}
        for r in retimed:
            assert all(q['output_sha256'] == reference_outputs[r['name'], r['cached_known_words'], s]
                       for s, q in r['query'].items())
        assert seen == expected
        print(name + ': 204 candidates, 408 audits, paired blocks, frozen choices and ' + str(len(seen)) + ' final timings verified')
    assert coverage == {(c, n, s) for c in ('legacy', 'average') for n in (4, 8, 12) for s in (3, 7, 11)}
    row_controller = load(root / 'row-audit-controller.json')
    assert row_controller['status'] == 'complete' and not row_controller['failures']
    assert row_controller['requested'] == len(row_controller['complete']) == len(row_reports) == 3672
    assert row_controller['frozen_selection_sha256'] == raw_hashes['selection.json']
    for r in row_controller['complete']:
        assert row_reports[r['model'], r['name']] == r['report_sha256']
    failure = load(root / 'row-audit-initial-failure.json')
    assert failure['status'] == 'failed' and len(failure['complete']) == 3564 and len(failure['failures']) == 108
    assert failure['private_report_sha256'] == row_controller['initial_failure_report_sha256']
    assert len(row_controller['repaired']) == 108 and not row_controller['repair_failures']
    initial_row_logs = load(root / 'auxiliary/row-initial-failure-logs.json.gz')
    assert {(v['model'], v['name']) for v in initial_row_logs} == {(v['model'], v['name']) for v in row_controller['repaired']}
    rounding = load(root / 'opq-rounding-summary.json')
    assert rounding['repaired'] == len(rounding['candidates']) == 108
    assert rounding['changed_candidates'] == sum(bool(v['thread_rounding']['validation_gate_changes']) for v in rounding['candidates']) == 0
    controller = load(root / 'retiming-controller.json')
    assert controller['status'] == 'complete' and controller['workers'] == 1 and not controller['failures']
    requests = {(r['model'], r['name'], r['cached'], r['index'], r['cache']) for r in controller['requests']}
    assert requests == final_requests and len(requests) == len(controller['complete']) == publication['retiming_requests']
    assert controller['frozen_selection_sha256'] == raw_hashes['selection.json']
    assert set(retiming_records) == final_requests
    for item in controller['complete']:
        key = tuple(item['request'][k] for k in ('model', 'name', 'cached', 'index', 'cache'))
        assert retiming_records[key]['private_report_sha256'] == item['report_sha256']
    before = load(root / 'retiming-before-resume.json')
    initial = load(root / 'retiming-initial-interrupted.json')
    assert before['private_report_sha256'] == recovery['recovery_source_report_sha256'] == controller['resume']['source_report_sha256']
    assert before['requests'] == initial['requests'] == controller['requests']
    assert before['execution_schedule_sha256'] == controller['execution_schedule_sha256']
    assert before['complete'] == initial['complete']
    assert all(v in controller['complete'] for v in before['complete'])
    assert recovery['retained_before_resume'] == len(before['complete'])
    assert initial['failures'] == controller['resume']['prior_failures'] and all(v['request']['index'] == 'hash' for v in initial['failures'])
    assert recovery['initial_failures'] == len(initial['failures'])
    env(controller['environment'], root); env(controller['resume']['environment'], root)
    initial_retime_logs = load(root / 'auxiliary/retiming-initial-failure-logs.json.gz')
    assert {tuple(v['request'][k] for k in ('model','name','cached','index','cache')) for v in initial_retime_logs} == {tuple(v['request'][k] for k in ('model','name','cached','index','cache')) for v in initial['failures']}
    training, analysis = load(root / 'mechanism-training.json'), load(root / 'mechanism-analysis.json')
    assert training['status'] == analysis['status'] == 'complete' and not training['failures'] and not analysis['failures']
    assert len(training['trajectories']) == 36 and len(analysis['complete']) == len(analysis['jobs']) == 108
    assert training['training_root_report_sha256'] == prepare['source_training_report_sha256']
    assert analysis['source_factorial_report_sha256'] == raw_hashes['mechanism-training.json']
    spec = importlib.util.spec_from_file_location('conditional_verifier', root.parent / 'conditional_similarity_evidence_20261109/verify.py')
    previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
    points, replay, trajectories = set(), 0, set()
    for t in training['trajectories']:
        path = 'trajectories/' + t['name'] + '/report.json'; r = load(root / path)
        assert t['report_sha256'] == raw_hashes[path]
        assert r['status'] == 'complete' and [p['steps'] for p in r['checkpoints']] == [2000, 8000, 32000]
        env(r['environment'], root)
        key = (r['convention'], r['order'], r['seed'], r['ngram_initialization_scale'])
        old = next(v for v in original['runs'] if (v['convention'], v['order'], v['seed']) == key[:3])
        assert r['source_checkpoint_sha256'] == old['checkpoint_sha256']
        assert key not in trajectories; trajectories.add(key)
        schedule_path = 'schedules/order-' + str(r['order']) + '-seed-' + str(r['seed']) + '.json'
        assert r['schedule_sha256'] == raw_hashes[schedule_path]
        schedule = load(root / schedule_path)
        assert schedule['maximum_steps'] == 32000
        assert schedule['source_training_report_sha256'] == prepare['source_training_report_sha256']
        assert schedule['corpus_sha256'] == original['tokenized_corpus_sha256']
        for p in r['checkpoints']:
            if p['steps'] == 2000:
                assert p['sampled_center_events_sha256'] == old['sampled_center_events_sha256']
            if p['steps'] == 2000 and r['ngram_initialization_scale'] == 1:
                assert p['original_input_bytes_replayed'] is p['original_output_bytes_replayed'] is True
                replay += 1
            assert p['sampled_pairs'] == p['steps'] * r['settings']['batch_size']
            pointname = t['name'].replace('/', '-') + '-step-' + str(p['steps'])
            folder = root / 'points' / pointname
            d = load(folder / 'report.json')
            assert d['status'] == 'complete' and d['point'] == p
            assert d['source_checkpoint_sha256'] == p['checkpoint_sha256']
            assert (d['convention'], d['order'], d['seed'], d['ngram_initialization_scale']) == key
            points.add(pointname)
            env(d['environment'], root)
            relation_path = 'points/' + pointname + '/conditional.json.gz'
            assert d['conditional_report_sha256'] == raw_hashes[relation_path]
            for relation in load(folder / 'conditional.json.gz')['relations']:
                previous.relation(relation, d['shape'][0], True)
            for variant, v in d['variants'].items():
                assert v['retained_rows'] + v['merged_rows'] == d['shape'][0]
                assert v['native_reconstruction_byte_exact_to_approximation'] is True
                assert v['fidelity']['minimum_cosine'] >= .99 - 1e-7
                assert v['fidelity']['max_relative_norm_change'] <= 1 / .99 - 1 + 1e-7
                assert v['model_bytes'] - v['row_bundle_bytes'] == d['native_identity_model_bytes'] - d['native_identity_row_bundle_bytes']
                for cached in (0, 1):
                    for split in SPLITS:
                        b = load(folder / f'{variant}-{split}-cache{cached}-blocks.json.gz')
                        meta = load(root / f'models/{d["convention"]}-order-{d["order"]}-seed-{d["seed"]}/metadata.json')
                        paired(b, v['quality'][str(cached)][split], meta['evaluations'][split])
                        for s in STRATA:
                            q, source_q = v['quality'][str(cached)][split][s], d['source_quality'][split][s]
                            assert q['pairs'] == source_q['pairs']
                            if q['pairs']:
                                close(q['baseline_sgns_loss_per_pair'], source_q['sgns_loss_per_pair'])
                                close(q['baseline_context_discrimination'], source_q['context_discrimination'])
                        if not cached and split != 'validation':
                            for s, ci in load(folder / f'{variant}-{split}-intervals.json.gz').items():
                                intervals(ci); assert ci['observed_blocks'] == len(b[s]['count'])
    assert replay == 18
    assert trajectories == {(c, n, s, scale) for c in ('legacy', 'average') for n in (4, 8, 12)
                            for s in (3, 7, 11) for scale in (1., .1)}
    assert points == {p['name'] for p in analysis['complete']}
    for p in analysis['complete']:
        assert p['report_sha256'] == raw_hashes['points/' + p['name'] + '/report.json']
    assert set(p.name for p in resolve_evidence(root / 'points').iterdir()) == points
    print('36 trajectories / 108 points; all 18 source input/output replays; conditional counts and intervals verified')
    print('Primary frozen choices: ' + str(len(summary)) + '; test failures ' + str(sum(not s['test_pass'] for s in summary))
          + '; confirmation failures ' + str(sum(not s['confirmation_pass'] for s in summary)))
    import summarize as derived
    assert load(root / 'summary.json') == derived.summarize(root)
    print('Paper-facing summary independently reconstructed from the archived reports')
    return summary


if __name__ == '__main__':
    verify()
