"""Verify the complete frozen native Qwen grid and summarize paired timing.

Direct and native variants share decoded codec bytes; byte hashes, prompts,
cached logits and generated tokens must match within each physical-device pair.
Timing summaries are descriptive. Request repeats and three source seeds are
not presented as independent statistical samples or simultaneous confidence.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics

from .run_native_gpu_blocks import paired_cells, reusable_block
from .tree_node_quality import digest, environment, load, write


def geometric_mean(values):
    if not values or any(v <= 0 or not math.isfinite(v) for v in values):
        raise ValueError('positive finite paired latency ratios required')
    return math.exp(statistics.mean(math.log(v) for v in values))


def audit(root, output, compact_output):
    state = load(root / 'native-parallel/dispatch.json')
    if state['status'] != 'complete' or state['failed'] or state['running']:
        raise ValueError('complete terminal block dispatcher required')
    report_hashes = {}; contrasts = []; groups = defaultdict(list); blocks_checked = 0
    request_keys = ('batch_case_ids', 'batch_groups', 'prompt_hashes', 'input_ids_sha256',
        'input_tokens', 'cached_logit_hashes', 'generated_ids_sha256')
    for selection_path in sorted((root / 'native-serving').glob('*/selection.json')):
        prepared = load(root / 'native-layouts' / selection_path.parent.name / 'prepared.json')
        selection = load(selection_path); complete = load(selection_path.parent / 'complete.json')
        if complete['status'] != 'complete' or complete['selection_sha256'] != digest(selection_path):
            raise ValueError('source completion or frozen selection changed')
        for block in paired_cells(prepared, selection, selection_path.parent):
            if not reusable_block(block): raise ValueError('incomplete, mixed-device or contaminated pair')
            blocks_checked += 1; reports = {}
            for cell in block['cells']:
                path = Path(cell['output']); sha = digest(path); value = load(path)
                if complete['gpu_reports'][str(path)] != sha: raise ValueError('completed GPU report changed')
                if value['decode_steps'] != 16 or value['cache_bytes'] != 0 or value['mode'] != 'codec':
                    raise ValueError('registered cached-decoding schedule changed')
                if len(value['requests']) != 8 or len(value['samples']) != 24:
                    raise ValueError('registered eight requests and three repeats required')
                if any(len(s['decode_ms']) != 16 for s in value['samples']):
                    raise ValueError('cached continuation length changed')
                variant = load(cell['job'])
                # Direct standalone job uses the direct-<codec> file stem.
                name = Path(cell['job']).stem
                kind = 'direct' if name == 'direct-' + block['codec'] else name.rsplit('-', 1)[1]
                reports[(kind, cell['batch_size'])] = value; report_hashes[str(path)] = sha
                if value['decoded_row_sha256'] != load(variant['report'])['decoded_row_sha256']:
                    raise ValueError('original-ID decoded code bytes changed')
            for batch in (1, 4):
                direct = reports[('direct', batch)]
                for kind in ('prior', 'zlib', 'libdeflate'):
                    candidate = reports[(kind, batch)]
                    for key in ('decoded_row_sha256', 'phase_source_sha256', 'cached_decode_source_sha256',
                                'codec_adapter_source_sha256', 'request_slice', 'unique_prompt_count'):
                        if candidate[key] != direct[key]: raise ValueError('paired byte/schedule identity changed: ' + key)
                    for expected, actual in zip(direct['requests'], candidate['requests']):
                        for key in request_keys:
                            if expected[key] != actual[key]: raise ValueError('cached Qwen outputs or prompts changed: ' + key)
                    ratios = []
                    for expected, actual in zip(direct['samples'], candidate['samples']):
                        if expected['memory_history_tokens'] != actual['memory_history_tokens']:
                            raise ValueError('causal memory history changed')
                        ratios.append((expected['prefill_ms'] + sum(expected['decode_ms'])) /
                                      (actual['prefill_ms'] + sum(actual['decode_ms'])))
                    ratio = geometric_mean(ratios)
                    record = {'source': block['source'], 'codec': block['codec'], 'cohort': block['cohort'],
                        'batch_size': batch, 'variant': kind, 'gpu_uuid': direct['gpu_before']['gpu_uuid'],
                        'cpu_affinity': direct['cpu_affinity'], 'complete_request_speed_ratio_direct_over_native': ratio,
                        'paired_repeat_request_ratios': ratios,
                        'complete_branch_saving_fraction': 1 - candidate['complete_serving_bytes']/direct['complete_serving_bytes']}
                    contrasts.append(record)
                    groups[(block['source'].split('-')[0], block['codec'], batch, kind)].append(record)
    if blocks_checked != 108 or len(report_hashes) != 864 or len(contrasts) != 648:
        raise ValueError('complete six-source/six-codec/three-cohort paired grid required')
    summaries = []
    for (model, codec, batch, kind), cells in sorted(groups.items()):
        if len(cells) != 9 or len({c['source'] for c in cells}) != 3:
            raise ValueError('three source seeds and three cohorts required')
        ratios = [c['complete_request_speed_ratio_direct_over_native'] for c in cells]
        savings = [c['complete_branch_saving_fraction'] for c in cells]
        summaries.append({'model': model, 'codec': codec, 'batch_size': batch, 'variant': kind,
            'seed_cohort_pairs': len(cells), 'complete_request_speed_ratio_direct_over_native': {
                'geometric_mean': geometric_mean(ratios), 'minimum': min(ratios), 'maximum': max(ratios)},
            'complete_branch_saving_fraction': {'median': statistics.median(savings), 'minimum': min(savings), 'maximum': max(savings)}})
    result = {'status': 'audited-complete-qwen-native-grid', 'audit_environment': environment(),
        'source_sha256': digest(__file__), 'paired_blocks': blocks_checked, 'gpu_reports': len(report_hashes),
        'paired_contrasts': len(contrasts), 'prompt_cached_logit_and_generation_hashes_match': True,
        'summary': summaries, 'contrasts': contrasts, 'gpu_report_sha256': report_hashes,
        'timing_definition': 'Paired direct/native ratio of complete prefill plus all 16 cached continuation steps. Geometric mean per eight-request/three-repeat cell, then balanced geometric mean of three seed times three cohort cells. Each contrast uses the same physical GPU and CPU core.',
        'interpretation': 'Ratio above one means faster than direct FAISS under the same codec bytes. Bounds are observed pair ranges, not confidence intervals. Source seeds and repeat requests are clustered; no independent-sample, simultaneous or universal dominance claim.',
        'capacity_scope': 'Complete serving branch including row-store/common model metadata, separate from base Qwen weights and whole-process RAM. Compression is relative to the same decoded codec; quantization error relative to FP32 is a separate result.',
        'scientific_scope': 'Frozen trained full-key Qwen3-0.6B and Qwen3.5-4B branches. Generic native byte access, separate from DATrie/mark-compact and the independent AG News/DBpedia application grid.'}
    write(output, result)
    compact = {k:v for k,v in result.items() if k not in ('contrasts', 'gpu_report_sha256')}
    compact.update(full_audit_sha256=digest(output), full_audit_artifact=str(output))
    write(compact_output, compact)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'output', 'compact-output'): p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args(); audit(args.root, args.output, args.compact_output)


if __name__ == '__main__': main()
