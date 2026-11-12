"""Keep measured query costs next to completed public-application savings."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(value):
    result = {k: v for k, v in value.items() if k not in ('timings', 'report_sha256')}
    result['models'] = [{k: v for k, v in model.items() if k != 'quality'} for model in value['models']]
    models = {m['model']: m for m in value['models']}
    groups = defaultdict(list)
    for block in value['timings']:
        model = models[block['model']]
        selected = {c['heldout']['name'] for c in model['choices'] if c['heldout']}
        hosts = {r['job']: r for r in block['reports'] if r['mode'] == 'host' and r['status'] == 'passed'}
        for name in sorted(selected):
            if name not in hosts:
                raise ValueError('selected host timing missing: ' + name)
            candidate = hosts[name]
            baselines = ['direct-fp32'] + (['direct-sq-fp16'] if name.endswith('sq-fp16') else [])
            for baseline in baselines:
                reference = hosts[baseline]
                if (candidate['gpu_uuid'], candidate['cpu_affinity'], candidate['resource_policy']) != (
                        reference['gpu_uuid'], reference['cpu_affinity'], reference['resource_policy']):
                    raise ValueError('paired device/core/policy differs')
                groups[(*model['family'][:2], name, baseline)].append({
                    'p50_slowdown': candidate['p50_ms'] / reference['p50_ms'],
                    'throughput_ratio': candidate['documents_per_second'] / reference['documents_per_second'],
                    'saving_fraction': 1 - candidate['complete_model_bytes'] / reference['complete_model_bytes'],
                    'resource_policy': candidate['resource_policy'],
                })

    def distribution(values):
        return {'minimum': min(values), 'geometric_mean': math.exp(statistics.mean(map(math.log, values))),
                'maximum': max(values)}

    result['selected_host_query_costs'] = [{
        'dataset': key[0], 'initialization': key[1], 'selected': key[2], 'baseline': key[3],
        'paired_cells': len(cells),
        'p50_slowdown_selected_over_baseline': distribution([c['p50_slowdown'] for c in cells]),
        'documents_per_second_selected_over_baseline': distribution([c['throughput_ratio'] for c in cells]),
        'complete_model_saving_fraction': {'minimum': min(c['saving_fraction'] for c in cells),
                                         'maximum': max(c['saving_fraction'] for c in cells)},
        'resource_policies': sorted({c['resource_policy'] for c in cells}),
    } for key, cells in sorted(groups.items())]
    result['timing_interpretation'] = ('Descriptive observed ranges and balanced geometric means over three seeds, '
        'three cohorts, batches 1/32 and five allocator budgets. Repeats are not independent confidence bounds. '
        'A p50 slowdown above one means slower. Lossless capacity saving does not imply faster queries. '
        'FP16 comparisons to direct FP32 combine quantization and storage; the separate direct-FP16 comparison '
        'isolates storage over the same codes. CUDA allocator caps exclude host RAM and staging.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.audit.read_text()))
    result.update(full_audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
                  full_audit_artifact='cpu-audit/application-results.json',
                  evidence_release='https://github.com/initial-d/ngram-store/releases/tag/controlled-deployment-evidence-20261112')
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
