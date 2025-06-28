"""
pyir.core.cache: Disk cache, kernel cache, registration, and HybridCache utilities
"""
import os
import hashlib
import pickle
import numpy as np
import threading
import time
import collections

from ..typing import *
from .function import register_function

# Cache directory configuration
PYIR_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.pyir_cache')
os.makedirs(PYIR_CACHE_DIR, exist_ok=True)

_kernel_cache = {}

# Internal caches
_cpu_kernel_cache = {}
_gpu_kernel_cache = {}

PYIR_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.pyir_cache')
os.makedirs(PYIR_CACHE_DIR, exist_ok=True)

def cpu_kernel_cache():
    return _cpu_kernel_cache

def gpu_kernel_cache():
    return _gpu_kernel_cache

__all__ = ['_kernel_cache', '_cpu_kernel_cache', '_gpu_kernel_cache', 'cpu_kernel_cache', 'gpu_kernel_cache']

def should_use_disk_cache(ir, access_count=1):
    return len(ir) > 4096 or access_count > 3

def _disk_cache_path(key, ext):
    h = hashlib.sha256(str(key).encode()).hexdigest()
    return os.path.join(PYIR_CACHE_DIR, f'{h}.{ext}')

def save_kernel_to_disk(key, ir, binary=None, meta=None):
    with open(_disk_cache_path(key, 'ir'), 'w') as f:
        f.write(ir)
    if binary is not None:
        with open(_disk_cache_path(key, 'bin'), 'wb') as f:
            f.write(binary)
    if meta is not None:
        with open(_disk_cache_path(key, 'meta'), 'wb') as f:
            pickle.dump(meta, f)

def load_kernel_from_disk(key):
    ir_path = _disk_cache_path(key, 'ir')
    bin_path = _disk_cache_path(key, 'bin')
    meta_path = _disk_cache_path(key, 'meta')
    if not os.path.exists(ir_path):
        return None
    with open(ir_path, 'r') as f:
        ir = f.read()
    binary = None
    if os.path.exists(bin_path):
        binary = np.memmap(bin_path, dtype='uint8', mode='r')
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    return {'ir': ir, 'binary': binary, 'meta': meta}

class HybridCache:
    def __init__(self, max_size=128, name="cache"):
        self.max_size = int(os.environ.get(f"PYIR_{name.upper()}_CACHE_SIZE", max_size))
        self.lock = threading.RLock()
        self.data = collections.OrderedDict()
        self.freq = collections.Counter()
        self.last_access = {}
        self.compile_time = {}
        self.exec_time = {}
        self.ir_size = {}
        self.user_priority = collections.defaultdict(lambda: 0)
        self.name = name
        self.prefetch_queue = collections.deque()
        self.prefetch_event = threading.Event()
        self._stop = False
        self.worker = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.worker.start()
        self.access_window = collections.deque(maxlen=16)
        self.alpha = float(os.environ.get(f"PYIR_{name.upper()}_ALPHA", 1.0))
        self.beta = float(os.environ.get(f"PYIR_{name.upper()}_BETA", 0.5))
        self.gamma = float(os.environ.get(f"PYIR_{name.upper()}_GAMMA", 2.0))
        self.delta = float(os.environ.get(f"PYIR_{name.upper()}_DELTA", 1.0))
        self.epsilon = float(os.environ.get(f"PYIR_{name.upper()}_EPSILON", 0.01))
        self.zeta = float(os.environ.get(f"PYIR_{name.upper()}_ZETA", 5.0))
    def __getitem__(self, key):
        with self.lock:
            value = self.data[key]
            self.freq[key] += 1
            self.last_access[key] = time.time()
            self.data.move_to_end(key)
            self.access_window.append(key)
            self._maybe_prefetch()
            return value
    def __setitem__(self, key, value):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
            self.data[key] = value
            self.freq[key] += 1
            self.last_access[key] = time.time()
            meta = getattr(value, '_cache_meta', None)
            if meta:
                self.compile_time[key] = meta.get('compile_time', 0)
                self.exec_time[key] = meta.get('exec_time', 0)
                self.ir_size[key] = meta.get('ir_size', 0)
                self.user_priority[key] = meta.get('user_priority', 0)
            self._evict_if_needed()
    def __contains__(self, key):
        with self.lock:
            return key in self.data
    def __len__(self):
        with self.lock:
            return len(self.data)
    def clear(self):
        with self.lock:
            self.data.clear()
            self.freq.clear()
            self.last_access.clear()
            self.compile_time.clear()
            self.exec_time.clear()
            self.ir_size.clear()
            self.user_priority.clear()
            self.access_window.clear()
    def _score(self, key):
        freq = self.freq[key]
        recency = time.time() - self.last_access.get(key, 0)
        compile_time = self.compile_time.get(key, 0)
        exec_time = self.exec_time.get(key, 0)
        ir_size = self.ir_size.get(key, 0)
        user_priority = self.user_priority.get(key, 0)
        return (self.alpha * freq
                - self.beta * recency
                + self.gamma * compile_time
                + self.delta * exec_time
                - self.epsilon * ir_size
                + self.zeta * user_priority)
    def _evict_if_needed(self):
        while len(self.data) > self.max_size:
            scores = {k: self._score(k) for k in self.data}
            min_score = min(scores.values())
            for k in self.data:
                if scores[k] == min_score:
                    self.data.pop(k)
                    self.freq.pop(k, None)
                    self.last_access.pop(k, None)
                    self.compile_time.pop(k, None)
                    self.exec_time.pop(k, None)
                    self.ir_size.pop(k, None)
                    self.user_priority.pop(k, None)
                    break
    def prefetch(self, fn, *args, **kwargs):
        key = kwargs.get('key')
        if key is not None and key in self.data:
            return
        self.prefetch_queue.append((fn, args, kwargs))
        self.prefetch_event.set()
    def _prefetch_worker(self):
        while not self._stop:
            self.prefetch_event.wait()
            while self.prefetch_queue:
                fn, args, kwargs = self.prefetch_queue.popleft()
                try:
                    fn(*args, **kwargs)
                except Exception as e:
                    pass
            self.prefetch_event.clear()
    def stop(self):
        self._stop = True
        self.prefetch_event.set()
        self.worker.join()
    def _maybe_prefetch(self):
        if len(self.access_window) < 3:
            return
        keys = list(self.access_window)[-3:]
        try:
            suffixes = [int(str(k[0]).rsplit('_', 1)[-1]) for k in keys]
            if suffixes[2] - suffixes[1] == suffixes[1] - suffixes[0] == 1:
                next_suffix = suffixes[2] + 1
                next_key = tuple(list(keys[2][:-1]) + [str(next_suffix)])
                if next_key not in self.data:
                    self.prefetch_queue.append((self.data[keys[2]], (), {'key': next_key}))
                    self.prefetch_event.set()
        except Exception:
            pass
    def set_weights(self, alpha=None, beta=None, gamma=None, delta=None, epsilon=None, zeta=None):
        if alpha is not None: self.alpha = alpha
        if beta is not None: self.beta = beta
        if gamma is not None: self.gamma = gamma
        if delta is not None: self.delta = delta
        if epsilon is not None: self.epsilon = epsilon
        if zeta is not None: self.zeta = zeta

def cache_priority(priority):
    def decorator(fn):
        if not hasattr(fn, '_cache_meta'):
            fn._cache_meta = {}
        fn._cache_meta['user_priority'] = priority
        return fn
    return decorator

_cpu_kernel_cache = HybridCache(max_size=128, name="cpu")
_gpu_kernel_cache = HybridCache(max_size=64, name="gpu")

def _select_cache(target):
    if target in ("cuda", "gpu"):
        return _gpu_kernel_cache
    return _cpu_kernel_cache

def _kernel_cache_key(name, ir, dtype, target, fast_math, no_optims=False):
    ir_hash = hashlib.blake2b(ir.encode(), digest_size=16).hexdigest()
    return (name, ir_hash, str(dtype), str(target), str(fast_math), str(no_optims))

def get_or_register_kernel(name, ir, dtype, target, fast_math, access_count=1, arg_ctypes=None, no_optims=False):
    key = _kernel_cache_key(name, ir, dtype, target, fast_math, no_optims)
    cache = _select_cache(target)
    if should_use_disk_cache(ir, access_count):
        disk = load_kernel_from_disk(key)
        if disk is not None:
            def dummy_kernel(*args, **kwargs):
                raise NotImplementedError("AOT kernel loaded from disk. Implement binary loading.")
            return dummy_kernel
    if key in cache:
        return cache[key]
    from . import _compile_ir
    register_function(name, ir)
    _compile_ir(ir, fn_name=name)
    import ctypes
    from ..typing import float32
    from . import _engine
    addr = _engine.get_function_address(name)
    if arg_ctypes is not None:
        cfunc_ty = ctypes.CFUNCTYPE(None, *arg_ctypes)
    else:
        arg_types = [dtype]
        cfunc_ty = ctypes.CFUNCTYPE(None, *(t.ctype for t in arg_types))
    cfunc = cfunc_ty(addr)
    cache[key] = cfunc
    return cfunc

def prefetch_kernel(fn, dtype=None, target=None, fast_math=None, no_optims=False):
    from ..fusion import get_kernel_ir_objects
    ir_module, _, _ = get_kernel_ir_objects(fn)
    ir = str(ir_module)
    name = fn.__name__
    if dtype is None:
        dtype = getattr(fn, '_arg_types', [None])[0] or 'float32'
    if target is None:
        target = 'cpu'
    if fast_math is None:
        fast_math = True
    key = _kernel_cache_key(name, ir, dtype, target, fast_math, no_optims)
    cache = _select_cache(target)
    if key in cache:
        return cache[key]
    cache.prefetch(get_or_register_kernel, name, ir, dtype, target, fast_math, key=key, no_optims=no_optims)
    return None

def clear_kernel_cache():
    _cpu_kernel_cache.clear()
    _gpu_kernel_cache.clear()

def get_cache_stats():
    return {
        'cpu_cache_size': len(_cpu_kernel_cache),
        'gpu_cache_size': len(_gpu_kernel_cache),
        'cpu_max_size': _cpu_kernel_cache.max_size,
        'gpu_max_size': _gpu_kernel_cache.max_size
    }

def set_fast_math(enabled=True):
    global _fast_math_flags
    _fast_math_flags = enabled
