"""Independent ordered model work in CPU-only preparation processes."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import threading

_callback = None


def _invoke(item):
    return _callback(item)


def ordered_map(callback, items, jobs=1):
    if not 1 <= jobs <= 64:
        raise ValueError('Evidence workers must be within 1–64')
    if jobs == 1:
        yield from map(callback, items)
        return
    if 'fork' not in multiprocessing.get_all_start_methods():
        raise ValueError('Parallel evidence preparation requires fork; use one worker on this platform')
    if threading.active_count() != 1:
        raise ValueError('Evidence preparation parent must have one Python thread')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Parallel evidence preparation requires an explicitly CPU-only parent')
    global _callback
    if _callback is not None:
        raise ValueError('Nested parallel evidence work is not supported')
    _callback = callback
    try:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('fork')) as pool:
            yield from pool.map(_invoke, items)
    finally:
        _callback = None
