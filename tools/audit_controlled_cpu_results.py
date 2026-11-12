"""Audit frozen row-packet selections and canonical-query cache experiments.

Only hashes and numeric summaries are exported. All losing layouts, direct
fallbacks and slower held-out choices remain visible. Cache timings are
canonical-row queries, not end-to-end Qwen inference or exclusive-CPU claims.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics

from .controlled_native_deployment import select_layouts
from .tree_node_quality import digest, environment, load, write


def distribution(values):
    return {'minimum': min(values), 'median': statistics.median(values), 'maximum': max(values)}


def audit(root, output, compact_output=None):
    reports = {}; selections = {}; records = []; groups = defaultdict(list)
    models = sorted((root / 'row-packets/results').glob('*/capacity-selection.json'))
    if len(models) != 6: raise ValueError('all six frozen row-packet sources required')
    for selection_path in models:
        selection = load(selection_path)
        if selection['status'] != 'frozen' or selection['test_used']:
            raise ValueError('capacity selection must be frozen without test selection')
        progress = load(selection_path.parent / 'cpu-progress.json')
        if progress['status'] != 'complete' or len(progress['completed']) != 174:
            raise ValueError('complete 174-cell CPU grid required per source')
        completed = progress['completed']; direct = {}; values = {}
        for entry in completed:
            p = Path(entry['report']); value = load(p); sha = digest(p)
            if selection['cpu_reports_sha256'][str(p)] != sha:
                raise ValueError('frozen CPU report changed')
            if value['status'] != 'passed' or value['job_sha256'] != digest(entry['job']):
                raise ValueError('CPU result or original job changed')
            if value['cache_bytes'] != 0: raise ValueError('registered zero-cache result required')
            values[str(p)] = value; reports[str(p)] = sha
            if entry['variant'] == 'direct': direct[entry['codec']] = value
        for entry in completed:
            reference = direct[entry['codec']]; candidate = values[entry['report']]
            if candidate['decoded_row_sha256'] != reference['decoded_row_sha256']:
                raise ValueError('decoded rows changed')
            for split in ('calibration', 'test'):
                for key in ('input_hashes', 'output_hashes'):
                    if candidate['query'][split][key] != reference['query'][split][key]:
                        raise ValueError('actual token inputs or query vectors changed')
        selections[str(selection_path)] = digest(selection_path)
        for family, chosen in selection['choices'].items():
            variant, gate = family.rsplit('-', 1); gate = float(gate)
            if chosen != select_layouts(completed, variant, gate):
                raise ValueError('published choice differs from registered calibration rule')
            for entry in chosen:
                reference = direct[entry['codec']]; candidate = values[entry['report']]
                saving = 1 - candidate['complete_branch_bytes']/reference['complete_branch_bytes']
                fallback = entry['variant'] == 'direct'
                if not fallback and saving < gate: raise ValueError('capacity gate violation')
                record = {'source': selection_path.parent.name, 'family': variant, 'gate': gate,
                    'codec': entry['codec'], 'fallback': fallback, 'selected_job_sha256': digest(entry['job']),
                    'report_sha256': reports[entry['report']], 'complete_branch_saving_fraction': saving,
                    'direct_branch_bytes': reference['complete_branch_bytes'],
                    'selected_branch_bytes': candidate['complete_branch_bytes'],
                    'test_query_p50_speed_ratio_direct_over_selected': reference['query']['test']['p50_ms']/candidate['query']['test']['p50_ms'],
                    'test_row_only_p50_speed_ratio_direct_over_selected': reference['row_only']['test']['p50_ms']/candidate['row_only']['test']['p50_ms'],
                    'cpu_affinity': candidate['cpu_affinity']}
                records.append(record); groups[(variant, gate, entry['codec'])].append(record)
    if len(reports) != 1044 or len(records) != 288: raise ValueError('complete registered grid required')
    summary = []
    for (family, gate, codec), cells in sorted(groups.items()):
        ratios = [c['test_query_p50_speed_ratio_direct_over_selected'] for c in cells]
        summary.append({'family': family, 'gate': gate, 'codec': codec, 'sources': len(cells),
            'fallback_sources': sum(c['fallback'] for c in cells),
            'held_out_faster_and_capacity_qualifying_sources': sum(not c['fallback'] and ratio > 1 for c, ratio in zip(cells, ratios)),
            'complete_branch_saving_fraction': distribution([c['complete_branch_saving_fraction'] for c in cells]),
            'test_query_p50_speed_ratio_direct_over_selected': distribution(ratios)})
    cache_reports = {}; cache_groups = defaultdict(list)
    cache_models = sorted((root / 'application-frontier/native-cache').glob('*/complete.json'))
    if len(cache_models) != 6: raise ValueError('six completed canonical cache grids required')
    for complete_path in cache_models:
        complete = load(complete_path); source = complete_path.parent.name
        values = []; direct = {}
        for name, expected in complete['reports'].items():
            path = Path(name)
            if digest(path) != expected: raise ValueError('completed cache result changed')
            value = load(path)
            if 'entry' not in value: continue  # hashed trace manifest
            if value['status'] != 'passed' or value['job_sha256'] != digest(value['entry']['job']):
                raise ValueError('cache result or original job changed')
            policy = path.stem.rsplit('-', 2)[1]
            if policy not in ('blocks', 'rows'): raise ValueError('registered cache policy missing')
            value['_requested_policy'] = policy
            if value['trace_sha256'] != digest(complete_path.parent / 'traces.npz'):
                raise ValueError('canonical trace changed')
            workers = value['workers']; hashes = workers[0]['output_hashes']
            if any(w['status'] != 'passed' or w['output_hashes'] != hashes for w in workers):
                raise ValueError('concurrent worker vectors changed')
            if len(workers) != value['concurrency'] or len({w['core'] for w in workers}) != len(workers):
                raise ValueError('worker count or logical CPU isolation mismatch')
            if sum(w['actual_native_cache_bytes'] for w in workers) != value['total_native_cache_bytes']:
                raise ValueError('aggregate cache accounting mismatch')
            for worker in workers:
                if worker['actual_native_cache_bytes'] > worker['requested_cache_bytes']:
                    raise ValueError('worker cache budget exceeded')
            elapsed = (max(w['end_ns'] for w in workers)-min(w['start_ns'] for w in workers))/1e9
            expected_rps = sum(w['completed_requests'] for w in workers)/elapsed
            if abs(expected_rps/value['requests_per_second']-1) > 1e-10:
                raise ValueError('aggregate throughput is not completed work over wall time')
            cache_reports[str(path)] = expected; values.append(value)
            entry = value['entry']; worker = workers[0]
            key = (entry['codec'], worker['requested_cache_bytes'], policy, len(workers))
            if entry['variant'] == 'direct': direct[key] = value
        if len(values) != 840: raise ValueError('complete 840-cell cache grid required per source')
        selection = load(root / 'native-serving' / source / 'selection.json')
        selected = {e['codec']: e for e in selection['selected']}
        for value in values:
            entry = value['entry']; worker = value['workers'][0]
            key = (entry['codec'], worker['requested_cache_bytes'], value['_requested_policy'], len(value['workers']))
            reference = direct[key]
            if value['workers'][0]['output_hashes'] != reference['workers'][0]['output_hashes']:
                raise ValueError('cache layout changed original canonical query vectors')
            chosen = selected[entry['codec']]
            selected_layout = (entry.get('block_rows'),entry.get('predictor')) == (chosen['block_rows'],chosen['predictor'])
            role = 'direct' if entry['variant']=='direct' else 'selected' if selected_layout else 'fixed-256-xor'
            group = (entry['codec'], entry['variant'], role, key[1], key[2], key[3])
            cache_groups[group].append(value['requests_per_second']/reference['requests_per_second'])
    cache_summary = [{'codec': k[0], 'variant': k[1], 'layout_role': k[2], 'cache_bytes_per_worker': k[3],
        'reported_policy': k[4], 'workers': k[5], 'source_observations': len(v),
        'query_throughput_ratio_to_direct': distribution(v)} for k,v in sorted(cache_groups.items())]
    if len(cache_reports) != 5040: raise ValueError('complete six-source cache grid required')
    result = {'status': 'audited-complete-cpu-grids', 'source_sha256': digest(__file__), 'audit_environment': environment(),
        'row_packet_reports': len(reports), 'row_packet_choices': len(records),
        'row_packet_summary': summary, 'row_packet_choices_by_source': records,
        'row_packet_report_sha256': reports, 'frozen_selection_sha256': selections,
        'canonical_cache_reports': len(cache_reports), 'canonical_cache_summary': cache_summary,
        'canonical_cache_report_sha256': cache_reports,
        'interpretation': 'Ratios above one mean faster. Distribution spans six trained sources, not a confidence interval. Direct fallback has zero additional savings and ratio one; it is not a compressed success. No held-out speed ratio changes calibration selection.',
        'scope': 'Row-packet complete-token CPU queries and canonical-row-only cache queries are separate. Earlier shared-host/SMT-overlap timing remains shared-host; no exclusive-CPU or GPU inference claim. Generic byte-packet layouts are separate from DATrie/mark-compact structural sharing.'}
    write(output, result)
    if compact_output:
        families = []
        for family in ('row-packets', 'lz4', 'zlib', 'zstd'):
            cells = [c for c in records if c['family'] == family]
            nonfallback = [c for c in cells if not c['fallback']]
            families.append({'family': family, 'frozen_choices': len(cells), 'nonfallback_choices': len(nonfallback),
                'held_out_faster_and_capacity_qualifying_choices': sum(c['test_query_p50_speed_ratio_direct_over_selected']>1 for c in nonfallback)})
        write(compact_output, {'status': result['status'], 'source_sha256': result['source_sha256'],
            'audit_environment': result['audit_environment'], 'full_audit_sha256': digest(output),
            'full_audit_artifact': str(output), 'row_packet_reports': len(reports),
            'canonical_cache_reports': len(cache_reports), 'frozen_choice_counts_by_family': families,
            'row_packet_summary': summary,
            'canonical_cache_fixed_scenarios': [c for c in cache_summary if c['layout_role']=='selected'
                and c['workers']==1 and c['cache_bytes_per_worker'] in (0,8<<20) and c['reported_policy']=='blocks'],
            'cache_snapshot_rule': 'Fixed readable slice: selected calibration layouts, one worker, block policy, cache zero and 8 MiB. Full audit retains every policy, concurrency, fixed layout and losing result.',
            'interpretation': result['interpretation'], 'scope': result['scope']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compact-output', type=Path)
    args = p.parse_args(); audit(args.root, args.output, args.compact_output)


if __name__ == '__main__': main()
