"""Frozen original-key Qwen memory codecs, official quality and real requests.

Model state, token IDs and key dictionaries remain private run artifacts.
Only numeric outcomes and hashes belong in public evidence.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .complete_memory_frontier import add_native_code_jobs, execute_resumable
from .tree_node_quality import (Reader, array_digest, candidate_jobs, digest,
    environment, load, memory, write)

ROW_NAME = 'ngram_memory.multi_head_embedding.embedding.weight'


def original_row_digest(reader, rows):
    result = hashlib.sha256()
    for begin in range(0, rows, 4096):
        ids = np.arange(begin, min(begin + 4096, rows), dtype=np.int64)
        selected = reader.mapping[ids] if reader.mapping is not None else ids
        values = reader.rows(selected)
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('invalid actual decoded FP32 rows')
        result.update(values.astype('<f4').tobytes())
    return result.hexdigest()


def prepare(args):
    import torch
    from safetensors.torch import load_file, save_file
    from .ngram_identity import Vocabulary
    root = args.output; root.mkdir(parents=True, exist_ok=True)
    protocol = load(args.protocol); all_models = []
    for model_name in ('qwen06', 'qwen35'):
        for seed in (3, 7, 11):
            run = args.sources / 'runs' / f'{model_name}-20877-exact-s{seed}'
            report = load(run / 'report.json'); recipe = load(run / 'model-recipe.json')
            if report['status'] != 'complete' or report['variant'] != 'exact' or report['seed'] != seed:
                raise ValueError('complete exact-key trained source required')
            if digest(run / 'delta.safetensors') != report['delta']['sha256']:
                raise ValueError('trained source delta changed')
            if recipe['identity'] != report['identity'] or recipe['protocol'] != report['protocol']:
                raise ValueError('trained source recipe changed')
            output = root / f'{model_name}-exact-s{seed}'
            existing = output / 'prepared.json'
            if existing.exists():
                value = load(existing)
                if value['metadata_sha256'] != digest(output / 'common' / 'metadata.json'):
                    raise ValueError('existing LLM preparation changed')
                all_models.append(value); continue
            if output.exists(): raise ValueError('partial LLM preparation exists; preserve and diagnose')
            common = output / 'common'; common.mkdir(parents=True)
            state = load_file(str(run / 'delta.safetensors'), device='cpu')
            rows = state.pop(ROW_NAME).numpy()
            heads = report['identity']['head_sizes']
            if rows.dtype != np.float32 or rows.shape != (sum(heads), 32):
                raise ValueError('trained original-head row shape changed')
            np.save(common / 'source-rows.npy', rows)
            save_file(state, str(common / 'nonrow-delta.safetensors')); del state
            shutil.copyfile(run / 'vocabulary.npz', common / 'vocabulary.npz')
            if digest(common / 'vocabulary.npz') != report['identity']['vocabulary_sha256']:
                raise ValueError('canonical training vocabulary changed')
            vocabulary = Vocabulary.load(common / 'vocabulary.npz')
            groups = []; occurrence = np.zeros(len(rows), dtype=np.int64); offset = 0
            for head, size in enumerate(heads):
                order = 2 + head // 2
                # Fixed-width signed integer fields with boundary delimiters
                # encode token containment faithfully. Each head has one order.
                # No head labels or sentinel strings generate false relations.
                keys = {str(i + 1): ''.join(f'[{int(token):+021d}]' for token in key)
                        for i, key in enumerate(vocabulary.keys[order])}
                if len(keys) + 1 > size: raise ValueError('canonical keys exceed trained head')
                groups.append({'start': offset, 'size': size, 'order': order, 'head': head, 'keys': keys})
                occurrence[offset + 1:offset + 1 + len(keys)] = vocabulary.counts[order]
                offset += size
            write(common / 'gram-key-groups.json', groups)
            np.save(common / 'source-occurrence-counts.npy', occurrence)
            fit_ids = np.random.default_rng(42).choice(len(rows), min(16384, len(rows)), replace=False)
            np.save(common / 'fit-original-ids.npy', fit_ids)
            recipe.update(nonrow_delta='nonrow-delta.safetensors',
                nonrow_delta_sha256=digest(common / 'nonrow-delta.safetensors'),
                prepared_sha256=report['prepared_sha256'], global_step=report['optimizer_steps'],
                frozen_weight_sha256=report['frozen_weight_sha256'],
                frozen_decoder_suffix_sha256=report['frozen_decoder_suffix_sha256'])
            write(common / 'model-recipe.json', recipe)
            filenames = ('nonrow-delta.safetensors', 'vocabulary.npz', 'model-recipe.json')
            files = {name: {'bytes': (common / name).stat().st_size, 'sha256': digest(common / name)} for name in filenames}
            base = Path(recipe['base_model'])
            base_files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
                          for p in sorted(base.iterdir()) if p.is_file() and p.suffix in ('.safetensors', '.json')}
            metadata = {'shape': list(rows.shape), 'head_sizes': heads, 'seed': seed, 'model': model_name,
                'source_row_sha256': array_digest(rows), 'source_checkpoint_sha256': report['delta']['sha256'],
                'source_report_sha256': digest(run / 'report.json'), 'source_recipe_sha256': digest(run / 'model-recipe.json'),
                'source_run': str(run), 'prepared': str(args.sources / f'prepared-{model_name}-20877'),
                'prefix_cache': str(args.sources / f'cache-{model_name}-20877'),
                'study_protocol_sha256': digest(args.protocol), 'common_serving_files': files,
                'common_model_bytes': sum(x['bytes'] for x in files.values()), 'base_files': base_files,
                'base_bytes': sum(x['bytes'] for x in base_files.values()),
                'unregistered_rows': len(rows) - sum(len(g['keys']) for g in groups),
                'structural_key_contract': 'Original per-head canonical training keys, zero/padding omitted; heads remain independent',
                'actual_same_head_cross_order_containment_pairs': 0,
                'category': 'Official frozen-backbone learned-memory editing storage comparison'}
            write(common / 'metadata.json', metadata)
            jobs = candidate_jobs(output, args.native_library, args.structural_library,
                codecs=protocol['llm']['faiss_32dim_codecs'], quality_contract='deferred-llm')
            value = add_native_code_jobs({'name': output.name, 'jobs': jobs})
            value['metadata_sha256'] = digest(common / 'metadata.json')
            write(existing, value); all_models.append(value)
            print({'prepared': output.name, 'candidates': len(value['jobs'])}, flush=True)
    if len(all_models) != 6: raise ValueError('six LLM sources required')
    write(root / 'prepare.json', {'status': 'complete', 'environment': environment(),
        'protocol_sha256': digest(args.protocol), 'models': all_models, 'candidates': 1566})


def build(args):
    prepared = load(args.output / 'prepare.json')
    if prepared['status'] != 'complete': raise ValueError('complete LLM preparation required')
    jobs = [p for model in prepared['models'] for p in model['jobs']]
    stages = (
        ('direct', [p for p in jobs if not load(p).get('native_codes_source') and load(p)['backend'] != 'dedup']),
        ('dedup', [p for p in jobs if not load(p).get('native_codes_source') and load(p)['backend'] == 'dedup']),
        ('native-codes', [p for p in jobs if load(p).get('native_codes_source')]),
    )
    for name, paths in stages:
        execute_resumable(paths, 'build', args.jobs, args.output / ('stage-' + name))
    write(args.output / 'build.json', {'status': 'complete', 'candidates': len(jobs),
        'prepare_sha256': digest(args.output / 'prepare.json')})


def reopen_codec(job, upstream, *, device='cuda', cache_bytes=0, profile=False):
    """Load the shared base and unchanged nonrow weights without dense rows."""
    import torch
    from safetensors.torch import load_file
    from .codec_memory_adapter import CodecMultiHeadEmbedding
    from .counterfact_models import setup
    from .counterfact_native import configure_identity, NativeHashMemory
    common = Path(job['common']); recipe = load(common / 'model-recipe.json')
    metadata = load(common / 'metadata.json'); reader = Reader(job, cache_bytes=cache_bytes)
    staged = None; replay = None
    def factory(list_of_N, D):
        nonlocal staged
        if len(list_of_N) != len(metadata['head_sizes']) or D != metadata['shape'][1]:
            raise ValueError('source head constructor differs')
        staged = CodecMultiHeadEmbedding(reader, metadata['head_sizes'], device, profile=profile, zero_sentinels=True)
        return staged
    try:
        p = recipe['protocol']; v = p['variants'][recipe['variant']]
        model, tokenizer, replay, module, config, loading = setup(recipe['base_model'], upstream,
            'full', v['vocab_sizes'], recipe['seed'], device=device,
            suffix_layers=p['suffix_layers'], train_decoder_suffix=p.get('train_decoder_suffix', True),
            embedding_factory=factory)
        configure_identity(replay, module, common, recipe, initialize_rows=False)
        heads = [n for order in replay.memory.hash_mapping.vocab_size_across_layers[replay.memory.layer_id] for n in order]
        if heads != list(staged.head_sizes): raise ValueError('configured canonical head IDs differ')
        path = common / recipe['nonrow_delta']
        if digest(path) != recipe['nonrow_delta_sha256']: raise ValueError('nonrow checkpoint changed')
        state = load_file(str(path), device='cpu')
        expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if state.keys() != expected.keys(): raise ValueError('missing or extra nonrow parameters')
        with torch.no_grad():
            for name, parameter in expected.items():
                value = state[name]
                if parameter.shape != value.shape or parameter.dtype != value.dtype:
                    raise ValueError('nonrow parameter shape/dtype changed')
                parameter.copy_(value)
        if replay.frozen_hash() != recipe['frozen_weight_sha256']:
            raise ValueError('immutable backbone changed')
        if replay.frozen_suffix_hash() != recipe['frozen_decoder_suffix_sha256']:
            raise ValueError('immutable decoder suffix changed')
        model.requires_grad_(False).eval(); module.set_global_step(recipe['global_step'])
        wrapper = NativeHashMemory(replay.memory).eval()
        model.ngram_memory = wrapper; replay.memory = wrapper
        return model, replay, staged
    except BaseException:
        if staged is not None: staged.close()
        else: reader.close()
        if replay is not None: replay.close()
        raise


def verify_candidate(job):
    import faiss
    faiss.omp_set_num_threads(1)
    built = load(job['report']); metadata = load(Path(job['common']) / 'metadata.json')
    if built['status'] != 'complete' or built['job_sha256'] != digest(Path(job['report']).parent.parent / 'jobs' / (job['name'] + '.json')):
        raise ValueError('unverified build job')
    for name, entry in built['serving_files'].items():
        if digest(Path(job['directory']) / name) != entry['sha256']:
            raise ValueError('immutable codec serving file changed')
    reader = Reader(job)
    try: actual = original_row_digest(reader, metadata['shape'][0])
    finally: reader.close()
    if actual != built['decoded_row_sha256']: raise ValueError('inference decoded rows differ from construction')
    return actual


def quality_worker(args):
    """All calibration candidates, then frozen selected test candidates.

    Each reader is reopened and all original rows verified. Equal complete row
    bytes share scores only after identical actual complete/replayed requests.
    """
    import faiss
    import torch
    from .codec_memory_adapter import CodecMultiHeadEmbedding
    from .counterfact_capacity import Prefixes, filled, ids_for, choice_scores, environment as qwen_environment
    from .counterfact_data import batch_indices, case_metrics, summarize
    from .counterfact_native import digest as tensor_digest
    env = qwen_environment(); faiss.omp_set_num_threads(1)
    env['codec_frontier_source_sha256'] = digest(__file__)
    env['codec_adapter_source_sha256'] = digest(Path(__file__).with_name('codec_memory_adapter.py'))
    out = args.model; prepared = load(out / 'prepared.json'); common = out / 'common'
    metadata = load(common / 'metadata.json'); recipe = load(common / 'model-recipe.json')
    source_prepared = Path(metadata['prepared']); pre = load(source_prepared / 'prepare.json')
    if digest(source_prepared / 'prepare.json') != recipe['prepared_sha256']:
        raise ValueError('official prepared manifest changed')
    if digest(source_prepared / 'sequences.json') != pre['payload_sha256']:
        raise ValueError('official sequences changed')
    payload = load(source_prepared / 'sequences.json'); sequences = payload['sequences']
    cases = [c for c in payload['cases'] if c['split'] == args.split]
    indices = sorted({pair[k] for c in cases for pairs in c['groups'].values() for pair in pairs for k in ('new', 'true')})
    batch_size = recipe['protocol']['batch_size']
    batches = batch_indices(indices, sequences, batch_size)
    schedule_hash = array_digest(np.array([i for valid in batches for i in filled(valid, batch_size)], dtype='<i8'))
    jobs = prepared['jobs']
    frozen_hash = None
    if args.split == 'test':
        frozen_hash = digest(args.output / 'selection.json')
        selection = next(m for m in load(args.output / 'selection.json')['models'] if m['name'] == out.name)
        names = {c['name'] for c in selection['choices'] if c['name'] is not None} | {'raw-fp32'}
        jobs = [p for p in jobs if load(p)['name'] in names]
    # Source comes first; other decoded identities choose a stable leader.
    jobs = sorted(jobs, key=lambda p: (load(p)['name'] != 'raw-fp32', load(p)['name']))
    ledger = out / (args.split + '-ledger.json')
    progress = {'status': 'running', 'environment': env, 'split': args.split,
        'protocol_sha256': digest(args.protocol), 'frozen_selection_sha256': frozen_hash,
        'oov_contract_sha256': digest(Path(args.protocol).with_name('memory_application_oov_contract_v1.json')),
        'model': out.name, 'cases': len(cases), 'choices': len(indices),
        'batch_schedule_sha256': schedule_hash, 'candidates': [], 'failures': []}
    prior = load(ledger) if ledger.exists() else None
    if prior:
        for field in ('protocol_sha256', 'oov_contract_sha256', 'split', 'batch_schedule_sha256', 'frozen_selection_sha256'):
            if prior[field] != progress[field]: raise ValueError('existing quality schedule changed')
        progress['failures'] = prior.get('failures', [])
    write(ledger, progress)
    model = replay = staged = prefixes = None
    completed = {}; validated_requests = {}; source_scores = load(Path(metadata['source_run']) / 'choice-scores.json')
    try:
        for job_path in jobs:
            job = load(job_path)
            wait_started = time.monotonic()
            while not Path(job['report']).exists():
                if not args.wait_build_seconds or time.monotonic() - wait_started > args.wait_build_seconds:
                    raise ValueError('candidate construction has not completed: ' + job['name'])
                time.sleep(10)
            decoded_hash = verify_candidate(job)
            result_path = out / 'quality' / (job['name'] + '-' + args.split + '.json')
            if model is None:
                model, replay, staged = reopen_codec(job, args.upstream, device=args.device)
                prefixes = Prefixes(Path(metadata['prefix_cache']), source_prepared, replay, batch_size)
                if len(prefixes.index) != len(sequences): raise ValueError('incomplete immutable prefix cache')
            else:
                staged.close()
                staged = CodecMultiHeadEmbedding(Reader(job), metadata['head_sizes'], args.device, zero_sentinels=True)
                replay.memory.source.multi_head_embedding = staged
            checks = []
            with torch.no_grad():
                for valid in batches[:16]:
                    batch = filled(valid, batch_size); ids = ids_for(batch, sequences)
                    keep = max(len(sequences[i]['target_ids']) for i in batch) + 1
                    hidden = prefixes.get(batch, replay.device)
                    if not torch.equal(hidden, replay.capture(ids)):
                        raise AssertionError('actual complete prefix differs from immutable cached prefix')
                    direct = replay.complete(ids, True, keep)
                    cached = replay.forward(ids, hidden, True, keep)
                    if tensor_digest(direct) != tensor_digest(cached):
                        raise AssertionError('actual full model and replay differ')
                    checks.append({'input_ids_sha256': tensor_digest(ids), 'logits_sha256': tensor_digest(direct),
                        'batch': len(batch), 'input_tokens': ids.shape[1], 'logits_to_keep': keep})
            if decoded_hash in validated_requests and checks != validated_requests[decoded_hash]:
                raise AssertionError('same decoded bytes changed actual full requests')
            validated_requests[decoded_hash] = checks
            reused = completed.get(decoded_hash)
            if result_path.exists():
                result = load(result_path)
                if (result['job_sha256'] != digest(job_path) or result['decoded_row_sha256'] != decoded_hash
                        or result['batch_schedule_sha256'] != schedule_hash or result['complete_request_checks'] != checks):
                    raise ValueError('existing quality evidence differs')
                if digest(result['case_scores']) != result['case_scores_sha256']:
                    raise ValueError('completed official case scores changed')
            elif reused:
                result = {**reused, 'name': job['name'], 'family': job['family'], 'job_sha256': digest(job_path),
                    'quality_reused_from': reused['name'], 'complete_request_checks': checks}
                write(result_path, result)
            else:
                scores = {}; started = time.perf_counter()
                with torch.no_grad():
                    for n, valid in enumerate(batches):
                        batch = filled(valid, batch_size); ids = ids_for(batch, sequences)
                        keep = max(len(sequences[i]['target_ids']) for i in batch) + 1
                        logits = replay.forward(ids, prefixes.get(batch, replay.device), True, keep)
                        scores.update(choice_scores(logits, batch, sequences, len(valid)))
                        if n % 200 == 0: print({'model': out.name, 'codec': job['name'], 'split': args.split, 'batches': n, 'scores': len(scores)}, flush=True)
                case_values = case_metrics(cases, scores)
                scores_path = out / 'quality' / (job['name'] + '-' + args.split + '-cases.json.gz')
                write(scores_path, case_values)
                result = {'status': 'complete', 'name': job['name'], 'family': job['family'],
                    'job_sha256': digest(job_path), 'decoded_row_sha256': decoded_hash,
                    'nonrow_delta_sha256': recipe['nonrow_delta_sha256'], 'frozen_weight_sha256': recipe['frozen_weight_sha256'],
                    'batch_schedule_sha256': schedule_hash, 'complete_request_checks': checks,
                    'cases': len(cases), 'choices': len(scores), 'metrics': summarize(case_values),
                    'case_scores': str(scores_path), 'case_scores_sha256': digest(scores_path),
                    'evaluation_seconds': time.perf_counter() - started, 'environment': env,
                    'quality_reused_from': None}
                result['oov_contract_sha256'] = progress['oov_contract_sha256']
                result['zero_sentinel_contract'] = 'Original local-ID zero outputs are exactly zero for all codecs; approximation of stored physical rows is reported separately'
                if job['name'] == 'raw-fp32':
                    mismatch = sum(scores[k] != source_scores[k] for k in scores)
                    result['source_original_choice_score_mismatches'] = mismatch
                    result['source_original_metrics'] = load(Path(metadata['source_run']) / 'report.json')['metrics'][args.split]
                    # Rebatching may change a few FP32 NLL bits. Qualification
                    # uses the freshly evaluated source under this exact schedule.
                    result['source_matches_original_metrics'] = result['metrics'] == result['source_original_metrics']
                write(result_path, result)
            completed.setdefault(decoded_hash, result)
            built = load(job['report'])
            entry = {'name': job['name'], 'family': job['family'], 'quality_report': str(result_path),
                'quality_report_sha256': digest(result_path), 'decoded_row_sha256': decoded_hash,
                'quality_reused_from': result['quality_reused_from'], 'metrics': result['metrics'],
                'complete_serving_bytes': built['model_bytes'] + (common / 'metadata.json').stat().st_size,
                'row_bundle_bytes': built['row_bundle_bytes'], 'base_bytes': metadata['base_bytes']}
            progress['candidates'].append(entry); write(ledger, progress)
            print({'model': out.name, 'split': args.split, 'candidates_complete': len(progress['candidates']), 'distinct_rows_evaluated': len(completed)}, flush=True)
        if frozen_hash and digest(args.output / 'selection.json') != frozen_hash:
            raise AssertionError('test altered frozen selection')
        progress.update(status='complete', requested_candidates=len(jobs), distinct_decoded_identities=len(completed))
        write(ledger, progress)
    except BaseException as error:
        progress['status'] = 'failed'; progress['failures'].append({'error': repr(error), 'time': time.time()})
        write(ledger, progress); raise
    finally:
        if staged is not None: staged.close()
        if replay is not None: replay.close()


def metric_values(metrics):
    return [metrics['harmonic_score']] + [metrics[k]['success'] for k in ('rewrite', 'paraphrase', 'neighborhood')]


def select(args):
    protocol = load(args.protocol); models = load(args.output / 'prepare.json')['models']; selected = []
    for model in models:
        ledger = load(args.output / model['name'] / 'calibration-ledger.json')
        if ledger['status'] != 'complete' or len(ledger['candidates']) != 261:
            raise ValueError('all 261 calibration candidates required before selection')
        rows = ledger['candidates']; source = next(r for r in rows if r['name'] == 'raw-fp32')
        baseline = metric_values(source['metrics']); choices = []
        for budget_pp in protocol['llm']['quality_budgets_percentage_points']:
            for family in sorted({r['family'] for r in rows}):
                eligible = [r for r in rows if r['family'] == family and all(
                    a - b <= budget_pp / 100 for a, b in zip(baseline, metric_values(r['metrics'])))]
                chosen = min(eligible, key=lambda r: (r['complete_serving_bytes'], r['name'])) if eligible else None
                choices.append({'budget_pp': budget_pp, 'family': family,
                    'name': chosen['name'] if chosen else None,
                    'complete_serving_bytes': chosen['complete_serving_bytes'] if chosen else None})
        selected.append({'name': model['name'], 'calibration_ledger_sha256': digest(args.output / model['name'] / 'calibration-ledger.json'),
            'source': source['metrics'], 'choices': choices})
    target = args.output / 'selection.json'
    result = {'protocol_sha256': digest(args.protocol), 'selection_uses': 'official calibration cases only', 'models': selected}
    if target.exists() and load(target) != result: raise ValueError('existing frozen choices changed')
    write(target, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'build', 'quality', 'select'))
    for name in ('output', 'sources', 'protocol', 'native-library', 'structural-library', 'model', 'upstream'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--jobs', type=int, default=16)
    parser.add_argument('--split', choices=('calibration', 'test'), default='calibration')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--wait-build-seconds', type=int, default=0,
        help='Bounded wait for an immutable candidate report when quality overlaps construction')
    args = parser.parse_args()
    if args.phase != 'quality' and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU preparation/build/selection requires empty CUDA_VISIBLE_DEVICES')
    if args.phase == 'prepare': prepare(args)
    elif args.phase == 'build': build(args)
    elif args.phase == 'quality': quality_worker(args)
    else: select(args)


if __name__ == '__main__': main()
