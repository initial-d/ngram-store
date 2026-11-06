"""Dependency-free integrity and paired-control metric audit."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,item in manifest.items():
        path=ROOT/name
        if not path.is_file() or path.stat().st_size!=item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise AssertionError('artifact changed: '+name)
    spec=importlib.util.spec_from_file_location('joint_analysis',ROOT/'runtime/analyze_tinyengram_joint.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    actual=module.analyze(ROOT/'training',protocol)
    expected=json.loads((ROOT/'summary.json').read_text())
    if actual!=expected:raise AssertionError('paired-control metrics/bootstrap/gates changed')
    inventory=json.loads((ROOT/'inventory.json').read_text())
    cleanup=json.loads((ROOT/'cleanup.json').read_text())
    if cleanup['status']!='passed' or len(cleanup['training_exit_codes'])!=9 or any(v!='0' for v in cleanup['training_exit_codes'].values()) or any(v['still_running_this_task'] for v in cleanup['known_workers'].values()):
        raise AssertionError('task workers did not finish cleanly')
    tokenizers=json.loads((ROOT/'tokenizer-check.json').read_text())
    environment=json.loads((ROOT/'environment.json').read_text())
    if hashlib.sha256((ROOT/'verify-saved-tokenizers.py').read_bytes()).hexdigest()!=tokenizers['script_sha256']:
        raise AssertionError('tokenizer verification source changed')
    for seed in protocol['seeds']:
        for control in module.CONTROLS:
            key=control+'-seed'+str(seed)
            report=module.read_json(ROOT/'training'/key/'report.json')
            if report['code_sha']!=environment['source_sha'] or report['protocol_sha256']!=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest():
                raise AssertionError('runtime/protocol provenance changed')
            if tokenizers['checks'][key]['splits']!=report['splits'] or not tokenizers['checks'][key]['all_split_token_ids_exact']:
                raise AssertionError('saved tokenizer changed data')
            for name,digest in report['source_file_hashes'].items():
                if hashlib.sha256((ROOT/'runtime'/name).read_bytes()).hexdigest()!=digest:
                    raise AssertionError('training runtime differs from recorded source')
            quality=module.read_json(module.quality_path(ROOT/'training'/key))
            if report['quality']['summary']!=quality['summary']:raise AssertionError('reported quality changed')
            for split in ('calibration','test'):
                reference=module.source_windows(quality,control,split)
                for name,groups in quality['windows'].items():
                    windows=groups[split];summary=quality['summary'][split][name]
                    if len(windows)!=protocol[split+'_windows']:raise AssertionError('window budget changed')
                    targets=sum(w['targets'] for w in windows)
                    for i,w in enumerate(windows):
                        if w['source_nll']!=reference[i]['source_nll']:raise AssertionError('ablation source changed')
                    for field in ('source_nll','candidate_nll'):
                        mean=math.fsum(x for w in windows for x in w[field])/targets
                        if abs(mean-summary[field])>1e-12:raise AssertionError('NLL summary changed')
                        ppl=field.replace('nll','ppl')
                        if abs(math.exp(mean)-summary[ppl])>1e-10:raise AssertionError('PPL summary changed')
                    if summary['targets']!=targets:raise AssertionError('target budget changed')
                    if name=='native-fp32' and not all(w['logits_byte_exact'] and w['source_nll']==w['candidate_nll'] and not any(w['source_to_candidate_kl']) and w['top1_changes']==0 for w in windows):
                        raise AssertionError('native byte-preservation changed')
            files=inventory[key]
            for kind in ('checkpoint','bundle'):
                if kind not in files:continue
                total=sum(f['bytes'] for f in files[kind])
                if total!=report['complete_'+kind+'_bytes']:raise AssertionError('complete file accounting changed')
            if control=='head-constant':
                expected_bytes=len(report['head_offsets'])*protocol['n_embed_per_ngram']//protocol['n_head_per_ngram']*4
                if report['constant_vector_payload_bytes']!=expected_bytes:raise AssertionError('constant payload changed')
            if control=='off':
                weights=next(f for f in files['checkpoint'] if f['path']=='model.safetensors')
                if weights['sha256']!=report['plain_baseline_sha256']:raise AssertionError('ordinary saved baseline changed')
            else:
                bundle=next(f for f in files['bundle'] if f['path']=='bundle.json')
                if bundle['sha256']!=report['bundle_manifest_sha256']:raise AssertionError('bundle manifest changed')
    original=module.read_json(ROOT/'original-qwen.json.gz')
    if hashlib.sha256((ROOT/'evaluate-original.py').read_bytes()).hexdigest()!=original['script_sha256']:
        raise AssertionError('unadapted reference source changed')
    full=module.read_json(ROOT/'training/full-seed3/report.json')
    if any(original[k]!=full[k] for k in ('splits','frozen_weight_sha256','initial_decoder_sha256')):
        raise AssertionError('unadapted reference changed model/data')
    for split,windows in original['windows'].items():
        ppl=math.exp(math.fsum(x for w in windows for x in w['source_nll'])/sum(w['targets'] for w in windows))
        if abs(ppl-original['summary'][split]['source_ppl'])>1e-10:raise AssertionError('reference PPL changed')
    print('Verified nine final joint/control models, exact pairing, paired-window bootstrap, fixed gates and complete file accounting.')


if __name__=='__main__':main()
