"""Real GPU allocator caps and matched host staging for public classifiers.

Allocator budgets cover PyTorch allocations, not CUDA contexts, host RSS or
other processes. OOM and external competition are retained as failed cells.
Full official test quality is separate from the deterministic timing prefix.
"""
import argparse
import gc
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

from .large_application_frontier import ApplicationRows
from .tree_node_quality import array_digest, digest, environment, load, memory, write


class SourceRows:
    def __init__(self, source):
        self.array = np.load(Path(source) / 'rows.npy', mmap_mode='r')
    def rows(self, ids): return np.ascontiguousarray(self.array[ids])
    def close(self): self.array._mmap.close()


def document_batches(ids, offsets, labels, batch_size, mapping=None):
    for begin in range(0, len(labels), batch_size):
        indices = range(begin, min(begin + batch_size, len(labels)))
        documents = [np.asarray(ids[offsets[i]:offsets[i + 1]]) for i in indices]
        if mapping is not None: documents = [mapping[x] for x in documents]
        yield begin, documents


def memory_details():
    result = memory()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(minor_faults=usage.ru_minflt, major_faults=usage.ru_majflt)
    path = Path('/proc/self/smaps_rollup')
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(('Pss:', 'Private_Dirty:', 'Shared_Clean:')):
                key, value = line.split(':', 1)
                result[key.lower() + '_bytes'] = int(value.split()[0]) * 1024
    return result


class StagedClassifier:
    """Same FP64 reductions/staging chunk schedule for every host backend.

    The resident control performs the same reduction using resident FP32 rows.
    Chunk sums can round differently from the CPU audit. Actual logit error and
    prediction disagreement against that audit are reported separately.
    """
    def __init__(self, reader, classifier, device, *, resident=None, chunk_rows=4096):
        import torch
        self.reader, self.resident = reader, resident
        self.device, self.chunk_rows = torch.device(device), chunk_rows
        self.weight = torch.as_tensor(classifier['weight'].astype(np.float64), device=device)
        self.bias = torch.as_tensor(classifier['bias'].astype(np.float64), device=device)
        self.staged_bytes = 0
        self.maximum_chunk_rows = 0

    def __call__(self, documents):
        import torch
        hidden = torch.zeros((len(documents), self.weight.shape[1]), dtype=torch.float64, device=self.device)
        for i, ids in enumerate(documents):
            ids = ids[ids != 0]  # Identical matched-domain OOV semantics.
            for start in range(0, len(ids), self.chunk_rows):
                chunk = ids[start:start + self.chunk_rows]
                if self.resident is None:
                    values = self.reader.rows(chunk)
                    if values.dtype != np.float32 or values.shape != (len(chunk), self.weight.shape[1]):
                        raise ValueError('reader returned invalid original row shape')
                    tensor = torch.from_numpy(np.ascontiguousarray(values)).to(self.device)
                    self.staged_bytes += values.nbytes
                else:
                    tensor = self.resident[torch.from_numpy(chunk.copy()).to(self.device)]
                hidden[i] += tensor.to(torch.float64).sum(dim=0)
                self.maximum_chunk_rows = max(self.maximum_chunk_rows, len(chunk))
            hidden[i] /= max(len(ids), 1)
        return (hidden @ self.weight.T + self.bias).to(torch.float32)


def resolve_worker_client(before, after, process_pid):
    """Identify our client without hiding existing host/container processes."""
    clients = set(after['compute_pids'])
    if process_pid in clients:
        return process_pid
    added = clients - set(before['compute_pids'])
    if len(added) != 1:
        raise ValueError('cannot uniquely identify worker CUDA client')
    return added.pop()


def worker(args):
    import torch
    import faiss
    from .benchmark_device_gather import gpu_snapshot
    torch.set_num_threads(1); torch.set_num_interop_threads(1); faiss.omp_set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    selector = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not selector or ',' in selector or torch.cuda.device_count() != 1:
        raise ValueError('exactly one selected GPU required')
    uuid = subprocess.check_output(['nvidia-smi', '--id=' + selector, '--query-gpu=uuid',
        '--format=csv,noheader,nounits'], text=True).strip()
    job = load(args.job); source = Path(job['source']); data = Path(job['data'])
    report = {'status': 'running', 'environment': {**environment(), 'device': 'cuda', 'gpu': uuid},
        'source_sha256': digest(__file__), 'job_sha256': digest(args.job), 'mode': args.mode,
        'allocator_budget_bytes': int(args.budget_gib * (1 << 30)), 'batch_size': args.batch_size,
        'resource_policy': 'shared' if args.allow_shared_gpu else 'exclusive',
        'cuda_selector': selector, 'gpu_samples': [], 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'host_before': memory_details(), 'samples_ms': [], 'complete_model_bytes': job['complete_model_bytes'],
        'timing_scope': 'First 256 official test documents in original order; repeated complete classification requests, fixed request order, no document/token truncation.',
        'budget_scope': 'PyTorch CUDA allocator only; context, driver, external allocations and all host memory remain outside this cap.',
        'host_cache_scope': 'Warm OS pages; zero application row/block cache; no global cache eviction.',
        'full_quality_scope': ('Every official test document, using the same reader and GPU reduction as timing.'
            if not args.timing_only else 'Timing prefix only. Full official test quality is in the separate CPU and full-GPU audits; this cell makes no full-test accuracy claim.')}
    reader = resident = engine = classifier = None
    try:
        before = gpu_snapshot(uuid, own_pid=0); report['gpu_before'] = before
        if not args.allow_shared_gpu and (before['external_compute_pids'] or before['utilization_percent'] > 5):
            raise ValueError('external GPU competition before opening')
        total = torch.cuda.get_device_properties(0).total_memory
        if report['allocator_budget_bytes'] > total: raise ValueError('allocator cap exceeds physical GPU memory')
        torch.cuda.set_per_process_memory_fraction(report['allocator_budget_bytes'] / total)
        probe = torch.empty(1, device='cuda'); torch.cuda.synchronize()
        monitor_pid = resolve_worker_client(before, gpu_snapshot(uuid, own_pid=0), os.getpid()); del probe
        report['monitor_client_pid'] = monitor_pid
        report['environment']['gpu'] = torch.cuda.get_device_name()
        def snapshot():
            value = gpu_snapshot(uuid, own_pid=monitor_pid); report['gpu_samples'].append(value)
            if monitor_pid not in value['compute_pids'] or (not args.allow_shared_gpu and value['external_compute_pids']):
                raise ValueError('external GPU competition during measurement')
        if digest(source / 'training.json') != job['source_report_sha256']:
            raise ValueError('frozen source changed')
        reader = SourceRows(source) if job['kind'] == 'source' else ApplicationRows(job)
        classifier = np.load(source / 'classifier.npz')
        if args.mode == 'resident':
            if job['kind'] != 'source': raise ValueError('resident controls require a source FP32 table')
            shape = reader.array.shape
            # Perform the real allocation: no analytical OOM shortcut.
            resident = torch.empty(shape, dtype=torch.float32, device='cuda')
            for begin in range(0, shape[0], args.chunk_rows):
                part = np.array(reader.array[begin:begin + args.chunk_rows], copy=True)
                resident[begin:begin + len(part)].copy_(torch.from_numpy(part))
        engine = StagedClassifier(reader, classifier, 'cuda', resident=resident, chunk_rows=args.chunk_rows)
        ids = np.load(data / 'test-ids.npy', mmap_mode='r')
        ptr = np.load(data / 'test-offsets.npy', mmap_mode='r')
        labels = np.load(data / 'test-labels.npy')
        mapping = np.load(source / 'canonical-to-bucket.npy', mmap_mode='r') if (source / 'canonical-to-bucket.npy').exists() else None
        if args.timing_only: labels = labels[:min(256, len(labels))]
        all_logits = []; start = time.perf_counter(); snapshot()
        with torch.no_grad():
            for index, documents in document_batches(ids, ptr, labels, args.batch_size, mapping):
                all_logits.append(engine(documents).cpu().numpy())
                if index % (args.batch_size * 128) == 0: snapshot()
            torch.cuda.synchronize()
            report['quality_seconds'] = time.perf_counter() - start
            logits = np.concatenate(all_logits)
            shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
            nll = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(logits)), labels]
            payload = args.output.with_suffix('.npz')
            np.savez(payload, logits=logits, labels=labels, nll=nll)
            report.update(documents=len(labels), accuracy=float(np.mean(logits.argmax(1) == labels)),
                mean_nll=float(nll.mean()), payload_sha256=digest(payload), logits_sha256=array_digest(logits))
            requests = list(document_batches(ids, ptr, labels[:min(256, len(labels))], args.batch_size, mapping))
            for _ in range(args.warmup):
                for _, documents in requests: engine(documents)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            timed_documents = 0; total_start = time.perf_counter()
            for _ in range(args.repeat):
                for _, documents in requests:
                    torch.cuda.synchronize(); start = time.perf_counter_ns()
                    engine(documents); torch.cuda.synchronize()
                    report['samples_ms'].append((time.perf_counter_ns() - start) / 1e6)
                    timed_documents += len(documents)
                snapshot()
            elapsed = time.perf_counter() - total_start
            report.update(status='passed', elapsed_wall_seconds=elapsed, timed_documents=timed_documents,
                documents_per_second=timed_documents / elapsed, p50_ms=float(np.median(report['samples_ms'])),
                p95_ms=float(np.quantile(report['samples_ms'], .95)), p99_ms=float(np.quantile(report['samples_ms'], .99)),
                staged_bytes=engine.staged_bytes, maximum_chunk_rows=engine.maximum_chunk_rows,
                host_after=memory_details(), cuda_allocated=torch.cuda.memory_allocated(),
                cuda_reserved=torch.cuda.memory_reserved(), cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved=torch.cuda.max_memory_reserved())
    except torch.cuda.OutOfMemoryError as error:
        report.update(status='allocator-oom', error=str(error), host_after=memory_details(),
            cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved())
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        if reader is not None: reader.close()
        if classifier is not None: classifier.close()
        resident = engine = None; gc.collect()
        write(args.output, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'output'): p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--mode', choices=('resident', 'host'), required=True)
    p.add_argument('--budget-gib', type=float, required=True)
    p.add_argument('--batch-size', type=int, choices=(1, 32), default=32)
    p.add_argument('--chunk-rows', type=int, default=4096)
    p.add_argument('--repeat', type=int, default=3); p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--timing-only', action='store_true')
    p.add_argument('--allow-shared-gpu', action='store_true', help='Record external GPU clients; shared timings are not exclusive performance evidence.')
    args = p.parse_args()
    if min(args.budget_gib, args.chunk_rows, args.repeat) <= 0 or args.warmup < 0:
        p.error('positive allocator/staging/repeat settings required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker(args)


if __name__ == '__main__': main()
