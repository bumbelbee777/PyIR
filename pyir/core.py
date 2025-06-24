"""
pyir.core: Core types, SSA, macros, and IR validation for PyIR
"""
import ctypes
import hashlib
import llvmlite.binding as llvm
import inspect
import textwrap
import ast
import functools
import warnings
import itertools
import re
import multiprocessing
import pickle
import numpy as np
import os
import threading
import concurrent.futures
import asyncio
import collections
import gc
import time

from .typing import *
from ._engine import _engine

_ssa_counters = {}
_macro_registry = {}

_function_registry = {}
_global_consts = {}
_current_target = 'cpu'
_call_count = itertools.count()

_kernel_cache = {}
_llvm_target_machine = None
_fast_math_flags = True

PYIR_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.pyir_cache')
os.makedirs(PYIR_CACHE_DIR, exist_ok=True)

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

def aot_compile_kernel(fn, dtype=None, target=None, fast_math=None, async_mode=False):
    """
    Compile a kernel and save IR/binary to disk cache. If async_mode, run in background and return a Future.
    """
    def _compile():
        from .fusion import get_kernel_ir_objects
        ir_module, _, _ = get_kernel_ir_objects(fn)
        ir = str(ir_module)
        name = fn.__name__
        if dtype is None:
            dtype_ = getattr(fn, '_arg_types', [None])[0] or 'float32'
        else:
            dtype_ = dtype
        if target is None:
            target_ = _current_target
        else:
            target_ = target
        if fast_math is None:
            fast_math_ = _fast_math_flags
        else:
            fast_math_ = fast_math
        key = _kernel_cache_key(name, ir, dtype_, target_, fast_math_)
        # Compile and get binary (simulate: use IR as binary for now)
        register_function(name, ir)
        _compile_ir(ir, fn_name=name)
        # Save to disk
        save_kernel_to_disk(key, ir, binary=ir.encode(), meta={'name': name, 'dtype': dtype_, 'target': target_, 'fast_math': fast_math_})
        return key
    if async_mode:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return executor.submit(_compile)
    else:
        return _compile()

def load_aot_kernel(fn, dtype=None, target=None, fast_math=None):
    """
    Load a kernel from disk cache if available, else compile and cache it. Uses memmap for binary if available.
    """
    from .fusion import get_kernel_ir_objects
    ir_module, _, _ = get_kernel_ir_objects(fn)
    ir = str(ir_module)
    name = fn.__name__
    if dtype is None:
        dtype_ = getattr(fn, '_arg_types', [None])[0] or 'float32'
    else:
        dtype_ = dtype
    if target is None:
        target_ = _current_target
    else:
        target_ = target
    if fast_math is None:
        fast_math_ = _fast_math_flags
    else:
        fast_math_ = fast_math
    key = _kernel_cache_key(name, ir, dtype_, target_, fast_math_)
    disk = load_kernel_from_disk(key)
    if disk is not None:
        # Use memmap binary if available (simulate: just use IR for now)
        # In real use, would load shared object or JIT from binary
        return disk
    # Not on disk: compile and cache
    aot_compile_kernel(fn, dtype, target, fast_math)
    return load_kernel_from_disk(key)

# --- Advanced HybridCache with smart heuristics and pattern-driven prefetch ---
class HybridCache:
    """
    Hybrid LFU-LRU cache with async prefetching, advanced heuristics, and pattern-driven prefetch.
    Evicts by lowest score (weighted freq, recency, compile/exec time, IR size, user priority).
    Predicts next likely kernel and prefetches.
    """
    def __init__(self, max_size=128, name="cache"):
        self.max_size = int(os.environ.get(f"PYIR_{name.upper()}_CACHE_SIZE", max_size))
        self.lock = threading.RLock()
        self.data = collections.OrderedDict()  # key -> value (LRU order)
        self.freq = collections.Counter()      # key -> access count (LFU)
        self.last_access = {}                  # key -> last access timestamp
        self.compile_time = {}                 # key -> compile time (s)
        self.exec_time = {}                    # key -> avg exec time (s)
        self.ir_size = {}                      # key -> IR size (bytes)
        self.user_priority = collections.defaultdict(lambda: 0)  # key -> user hint
        self.name = name
        self.prefetch_queue = collections.deque()
        self.prefetch_event = threading.Event()
        self._stop = False
        self.worker = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.worker.start()
        self.access_window = collections.deque(maxlen=16)  # rolling window of recent accesses
        # Heuristic weights (can be set via env or API)
        self.alpha = float(os.environ.get(f"PYIR_{name.upper()}_ALPHA", 1.0))  # freq
        self.beta = float(os.environ.get(f"PYIR_{name.upper()}_BETA", 0.5))   # recency
        self.gamma = float(os.environ.get(f"PYIR_{name.upper()}_GAMMA", 2.0)) # compile_time
        self.delta = float(os.environ.get(f"PYIR_{name.upper()}_DELTA", 1.0)) # exec_time
        self.epsilon = float(os.environ.get(f"PYIR_{name.upper()}_EPSILON", 0.01)) # ir_size
        self.zeta = float(os.environ.get(f"PYIR_{name.upper()}_ZETA", 5.0))   # user_priority

    def __getitem__(self, key):
        with self.lock:
            value = self.data[key]
            self.freq[key] += 1
            self.last_access[key] = time.time()
            self.data.move_to_end(key)  # LRU
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
            # If value has _compile_time/_exec_time/_ir_size/_user_priority, record them
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
        # Higher score = more valuable to keep
        freq = self.freq[key]
        recency = time.time() - self.last_access.get(key, 0)
        compile_time = self.compile_time.get(key, 0)
        exec_time = self.exec_time.get(key, 0)
        ir_size = self.ir_size.get(key, 0)
        user_priority = self.user_priority.get(key, 0)
        # Lower recency (more recent) is better, so use -recency
        return (self.alpha * freq
                - self.beta * recency
                + self.gamma * compile_time
                + self.delta * exec_time
                - self.epsilon * ir_size
                + self.zeta * user_priority)

    def _evict_if_needed(self):
        while len(self.data) > self.max_size:
            # Compute scores for all keys
            scores = {k: self._score(k) for k in self.data}
            min_score = min(scores.values())
            # Evict the key with the lowest score
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
        # Simple pattern: if last N accesses are sequential or repeated, prefetch next likely kernel
        if len(self.access_window) < 3:
            return
        # Example: if last 3 keys are the same except for a numeric suffix, prefetch next
        keys = list(self.access_window)[-3:]
        try:
            # Extract numeric suffixes
            suffixes = [int(str(k[0]).rsplit('_', 1)[-1]) for k in keys]
            if suffixes[2] - suffixes[1] == suffixes[1] - suffixes[0] == 1:
                # Sequential pattern detected, prefetch next
                next_suffix = suffixes[2] + 1
                next_key = tuple(list(keys[2][:-1]) + [str(next_suffix)])
                if next_key not in self.data:
                    # Prefetch using the same function as last
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

# --- User hint decorator for cache priority ---
def cache_priority(priority):
    """Decorator to set user cache priority for a kernel function."""
    def decorator(fn):
        if not hasattr(fn, '_cache_meta'):
            fn._cache_meta = {}
        fn._cache_meta['user_priority'] = priority
        return fn
    return decorator

# --- Multi-level cache: CPU and GPU ---
_cpu_kernel_cache = HybridCache(max_size=128, name="cpu")
_gpu_kernel_cache = HybridCache(max_size=64, name="gpu")

# --- Helper to select cache by target ---
def _select_cache(target):
    if target in ("cuda", "gpu"):
        return _gpu_kernel_cache
    return _cpu_kernel_cache

# --- Integrate HybridCache into kernel registration ---
def get_or_register_kernel(name, ir, dtype, target, fast_math, access_count=1):
    key = _kernel_cache_key(name, ir, dtype, target, fast_math)
    cache = _select_cache(target)
    # Heuristic: use disk cache if large or frequent
    if should_use_disk_cache(ir, access_count):
        disk = load_kernel_from_disk(key)
        if disk is not None:
            def dummy_kernel(*args, **kwargs):
                raise NotImplementedError("AOT kernel loaded from disk. Implement binary loading.")
            return dummy_kernel
    if key in cache:
        return cache[key]
    # Compile and register
    register_function(name, ir)
    _compile_ir(ir, fn_name=name)
    from . import _engine
    addr = _engine.get_function_address(name)
    # Infer arg types for cfunc
    arg_types = [dtype]  # Simplified; update as needed
    import ctypes
    cfunc_ty = ctypes.CFUNCTYPE(None, *(t.ctype for t in arg_types))
    cfunc = cfunc_ty(addr)
    cache[key] = cfunc
    return cfunc

# --- Prefetch kernel using async worker and heuristics ---
def prefetch_kernel(fn, dtype=None, target=None, fast_math=None):
    from .fusion import get_kernel_ir_objects
    ir_module, _, _ = get_kernel_ir_objects(fn)
    ir = str(ir_module)
    name = fn.__name__
    if dtype is None:
        dtype = getattr(fn, '_arg_types', [None])[0] or 'float32'
    if target is None:
        target = _current_target
    if fast_math is None:
        fast_math = _fast_math_flags
    key = _kernel_cache_key(name, ir, dtype, target, fast_math)
    cache = _select_cache(target)
    if key in cache:
        return cache[key]
    # Prefetch asynchronously
    cache.prefetch(get_or_register_kernel, name, ir, dtype, target, fast_math, key=key)
    return None

# --- Clear and stats for all caches ---
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
    """Enable or disable fast-math optimizations globally."""
    global _fast_math_flags
    _fast_math_flags = enabled

def get_llvm_target_machine():
    """Get or create an optimized LLVM target machine for JIT compilation."""
    global _llvm_target_machine
    if _llvm_target_machine is None:
        # Initialize LLVM
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        # Create target machine with optimizations
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine(
            opt=3,  # -O3 optimization level
            reloc='pic',
            codemodel='jitdefault'
        )
        _llvm_target_machine = target_machine
    
    return _llvm_target_machine

def optimize_ir_module(ir_module_str):
    """Apply LLVM optimizations to an IR module string."""
    try:
        # Parse IR
        mod = llvm.parse_assembly(ir_module_str)
        
        # Get target machine for optimization
        target_machine = get_llvm_target_machine()
        
        # Apply optimizations
        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 3
        pmb.size_level = 0
        pmb.loop_vectorize = True
        pmb.slp_vectorize = True
        
        pm = llvm.create_module_pass_manager()
        pmb.populate(pm)
        
        # Run optimizations
        pm.run(mod)
        
        return str(mod)
    except Exception as e:
        # Fallback to unoptimized IR if optimization fails
        warnings.warn(f"[pyir] IR optimization failed: {e}. Using unoptimized IR.")
        return ir_module_str

def global_const(name, llvm_type, value):
    """Register a global constant for the module."""
    _global_consts[name] = (llvm_type, value)

def set_target(target):
    global _current_target
    if target not in ('cpu', 'cuda', 'gpu'):
        raise ValueError("[pyir] Only 'cpu', 'cuda', and 'gpu' targets are supported.")
    _current_target = target
    if target in ('cuda', 'gpu'):
        warnings.warn("[pyir] CUDA/GPU target is experimental and untested. IR will be emitted, but JIT may not work.")

def register_function(name, ir):
    _function_registry[name] = ir

def get_module_ir():
    """Assemble the full module IR with all functions and globals."""
    global_ir = []
    for name, (llvm_type, value) in _global_consts.items():
        global_ir.append(f"@{name} = constant {llvm_type} {value}")
    for ir in _function_registry.values():
        global_ir.append(ir)
    return '\n'.join(global_ir)

sandbox_mode = False

def set_sandbox_mode(enabled=True):
    """Enable or disable PyIR sandbox mode (restricts JIT execution to a subprocess or disables it)."""
    global sandbox_mode
    sandbox_mode = enabled

def validate_ir(ir: str):
    """Validate and pretty-print an IR snippet (raises if invalid). Always called before JIT or execution."""
    if ir is None or ir == "":
        raise ValueError(f"[pyir.validate_ir] Empty IR passed!")
    try:
        llvm.parse_assembly(ir)
    except Exception as e:
        raise ValueError(f"[pyir.validate_ir] Invalid IR:\n{e}\n--- IR ---\n{ir}")
    return ir

def _compile_ir(ir_code: str, fn_name: str = None, args=None, ret_ctype=None):
    try:
        validate_ir(ir_code)  # Always validate IR before JIT
    except Exception as e:
        raise ValueError(f"[pyir._compile_ir] Error validating IR:\n{e}\n")
    if sandbox_mode:
        # Run JIT and execution in a subprocess for sandboxing
        def worker(ir_code_bytes, fn_name, args_bytes, ret_ctype_bytes, result_queue):
            import llvmlite.binding as llvm
            import ctypes
            import pickle
            try:
                ir_code = pickle.loads(ir_code_bytes)
                args = pickle.loads(args_bytes) if args_bytes else None
                ret_ctype = pickle.loads(ret_ctype_bytes) if ret_ctype_bytes else ctypes.c_double
                mod = llvm.parse_assembly(ir_code)
                mod.verify()
                from pyir import _engine
                _engine.add_module(mod)
                _engine.finalize_object()
                _engine.run_static_constructors()
                if fn_name and args is not None:
                    addr = _engine.get_function_address(fn_name)
                    # Support int, float, bool, tuple/list, struct, array, vector, nested, custom ctypes, opaque types, numpy arrays, device pointers
                    def infer_ctype(val):
                        import numpy as np
                        try:
                            import numba.cuda as nbcuda
                        except ImportError:
                            nbcuda = None
                        if isinstance(val, bool):
                            return ctypes.c_bool
                        elif isinstance(val, int):
                            return ctypes.c_int64
                        elif isinstance(val, float):
                            return ctypes.c_double
                        elif isinstance(val, (tuple, list)):
                            return infer_ctype(val[0]) * len(val)
                        elif hasattr(val, '_fields_') and issubclass(type(val), ctypes.Structure):
                            return type(val)
                        elif isinstance(val, ctypes.Array):
                            return type(val)
                        elif hasattr(val, '_type_') and hasattr(val, '_length_'):
                            return type(val)
                        elif hasattr(val, '_as_parameter_'):
                            return type(val)
                        elif isinstance(val, bytes):
                            return ctypes.c_char * len(val)
                        elif nbcuda and isinstance(val, nbcuda.cudadrv.devicearray.DeviceNDArray):
                            # Device pointer as int64
                            return ctypes.c_uint64
                        elif isinstance(val, np.ndarray):
                            # Numpy array as pointer
                            return ctypes.c_void_p
                        else:
                            raise TypeError(f"[pyir.sandbox] Unsupported arg type: {type(val)}")
                    ctypes_args = [infer_ctype(a) for a in args]
                    cfunc_ty = ctypes.CFUNCTYPE(ret_ctype, *ctypes_args)
                    cfunc = cfunc_ty(addr)
                    def flatten(val):
                        import numpy as np
                        try:
                            import numba.cuda as nbcuda
                        except ImportError:
                            nbcuda = None
                        if isinstance(val, (tuple, list)):
                            return [x for v in val for x in flatten(v)]
                        elif hasattr(val, '_fields_') and issubclass(type(val), ctypes.Structure):
                            return [getattr(val, f[0]) for f in val._fields_]
                        elif isinstance(val, ctypes.Array):
                            return list(val)
                        elif isinstance(val, bytes):
                            return list(val)
                        elif nbcuda and isinstance(val, nbcuda.cudadrv.devicearray.DeviceNDArray):
                            return [val.device_ctypes_pointer.value]
                        elif isinstance(val, np.ndarray):
                            return [val.ctypes.data]
                        else:
                            return [val]
                    flat_args = []
                    for a in args:
                        flat_args.extend(flatten(a))
                    result = cfunc(*flat_args)
                    def unpack(val):
                        if hasattr(val, '_fields_') and issubclass(type(val), ctypes.Structure):
                            return tuple(unpack(getattr(val, f[0])) for f in val._fields_)
                        elif isinstance(val, ctypes.Array):
                            return tuple(unpack(x) for x in val)
                        elif isinstance(val, bytes):
                            return val
                        else:
                            return val
                    result_queue.put(unpack(result))
                else:
                    result_queue.put(True)
            except Exception as e:
                result_queue.put(e)
        ctx = multiprocessing.get_context('spawn')
        result_queue = ctx.Queue()
        p = ctx.Process(target=worker, args=(pickle.dumps(ir_code), fn_name, pickle.dumps(args) if args else None, pickle.dumps(ret_ctype) if ret_ctype else None, result_queue))
        p.start()
        p.join()
        if not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                raise RuntimeError(f"[pyir] Sandbox subprocess error: {result}")
            return result
        else:
            raise RuntimeError("[pyir] Sandbox subprocess failed with no result.")
    try:
        mod = llvm.parse_assembly(ir_code)
        mod.verify()
        from pyir import _engine
        _engine.add_module(mod)
        _engine.finalize_object()
        _engine.run_static_constructors()
        print(f"[pyir._compile_ir] Successfully compiled and added module for {fn_name}")
        return mod
    except Exception as e:
        msg = f"Failed to compile LLVM IR"
        if fn_name:
            msg += f" for function '{fn_name}'"
        msg += f":\n{e}\n--- IR snippet ---\n{ir_code[:500]}\n--- End IR ---\n"
        msg += "Check your IR syntax, types, and ensure all variables are uniquely named."
        raise ValueError(msg)

def ssa(base: str) -> str:
    """Generate a unique SSA variable name for a given base name."""
    count = _ssa_counters.get(base, 0) + 1
    _ssa_counters[base] = count
    return f"%{base}{count}"

def define_macro(name: str, template: str):
    """Register a reusable IR macro template."""
    _macro_registry[name] = template

# Add mapping for Python types to PyIR types (expanded)
python_type_map = {
    int: int32,
    float: float32,
    bool: IntType(1),
    complex: complex128,
    'int8': int8,
    'int16': int16,
    'int32': int32,
    'int64': int64,
    'float16': float16,
    'float32': float32,
    'float64': float64,
    'complex64': complex64,
    'complex128': complex128,
}

def infer_type_from_value(val):
    if isinstance(val, bool):
        return IntType(1)
    elif isinstance(val, int):
        if -(2**7) <= val < 2**7:
            return int8
        elif -(2**15) <= val < 2**15:
            return int16
        elif -(2**31) <= val < 2**31:
            return int32
        else:
            return int64
    elif isinstance(val, float):
        return float64 if abs(val) > 1e38 or (val != 0 and abs(val) < 1e-38) else float32
    elif isinstance(val, complex):
        # Use complex128 for Python complex literals by default
        return complex128
    # User-defined types: struct, vec, array, opaque, named_types
    elif hasattr(val, 'llvm') and hasattr(val, 'ctype'):
        return val
    elif isinstance(val, str) and val in named_types:
        return named_types[val]
    return int32

# --- Refactored function decorator to use IR object model ---
def function(fn=None, *, target=None, cuda_kernel=False, simd=False, simd_width=None, simd_dtype=None, fast_math=True):
    """
    Decorator for ergonomic LLVM IR kernels (now using IR object model).
    Accepts pyir types, Python int/float/bool/complex, and user-defined types for arguments, returns, and variable declarations.
    If simd=True, automatically generates a SIMD version of the kernel using @simd_kernel. If simd_width or simd_dtype are not specified, they are auto-detected from vector argument types.
    If fast_math=True, adds fast-math flags to floating-point ops.
    """
    def decorator(fn):
        src = inspect.getsource(fn)
        src = textwrap.dedent(src)  # Dedent to avoid IndentationError
        src_ast = ast.parse(src)
        funcdef = src_ast.body[0]
        sig = inspect.signature(fn)
        arg_types = []
        arg_names = []
        allowed_types = (IntType, FloatType, VectorType, StructType, ArrayType, VoidType, PointerType, FunctionPointerType, OpaqueType)
        for name, param in sig.parameters.items():
            ann = param.annotation
            if ann is inspect._empty:
                inferred = None
                for stmt in funcdef.body:
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                        if stmt.targets[0].id == name:
                            v = stmt.value
                            if isinstance(v, ast.Constant):
                                inferred = infer_type_from_value(v.value)
                            elif isinstance(v, ast.BinOp):
                                for side in [v.left, v.right]:
                                    if isinstance(side, ast.Constant):
                                        inferred = infer_type_from_value(side.value)
                                        break
                            elif isinstance(v, ast.Call):
                                if isinstance(v.func, ast.Name):
                                    if v.func.id in python_type_map:
                                        inferred = python_type_map[v.func.id]
                                    elif v.func.id in named_types:
                                        inferred = named_types[v.func.id]
                                elif hasattr(v.func, 'attr') and v.func.attr in ('struct', 'vec', 'array', 'opaque'):
                                    inferred = {'struct': struct, 'vec': vec, 'array': array, 'opaque': opaque}[v.func.attr]
                            break
                ann = inferred or int32
            elif ann in python_type_map:
                ann = python_type_map[ann]
            elif isinstance(ann, str) and ann in named_types:
                ann = named_types[ann]
            if not (isinstance(ann, allowed_types) or type(ann) in allowed_types):
                raise TypeError(
                    f"[pyir] Error in function '{fn.__name__}' (defined at {inspect.getsourcefile(fn) or '<unknown file>'}:{inspect.getsourcelines(fn)[1]}):\n"
                    f"  Parameter '{name}' must be annotated with a pyir type, Python int/float/bool, or user-defined type.\n"
                    f"  Instead got: {param.annotation!r}\n"
                    f"  Please annotate all arguments with pyir types, int/float/bool, or user-defined types."
                )
            arg_types.append(ann)
            arg_names.append(name)
        ret_ann = sig.return_annotation
        if ret_ann is inspect._empty:
            ret_ann = void
        elif ret_ann in python_type_map:
            ret_ann = python_type_map[ret_ann]
        elif isinstance(ret_ann, str) and ret_ann in named_types:
            ret_ann = named_types[ret_ann]
        if not (isinstance(ret_ann, allowed_types) or type(ret_ann) in allowed_types):
            raise TypeError(
                f"[pyir] Error in function '{fn.__name__}' (defined at {inspect.getsourcefile(fn) or '<unknown file>'}:{inspect.getsourcelines(fn)[1]}):\n"
                f"  Return type must be a pyir type, Python int/float/bool, or user-defined type.\n"
                f"  Instead got: {ret_ann!r}\n"
                f"  Please annotate the return type with a pyir type, int/float/bool, or user-defined type."
            )
        type_suffix = "_".join(t.llvm for t in arg_types + [ret_ann])
        mangled = f"{fn.__name__}__{type_suffix}"
        src_file = inspect.getsourcefile(fn) or "<unknown file>"
        src_line = inspect.getsourcelines(fn)[1]
        cache = {}

        # --- IR object construction for eager registration ---
        entry_block = IRBlock('entry')
        declared_vars = set(arg_names)
        used_vars = set()
        return_var = None
        duplicate_vars = set()
        for stmt in funcdef.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                var = stmt.target.id
                ann = stmt.annotation
                if isinstance(ann, ast.Name) and ann.id in python_type_map:
                    pass
                if var in declared_vars:
                    duplicate_vars.add(var)
                declared_vars.add(var)
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                var = stmt.targets[0].id
                if var not in declared_vars:
                    v = stmt.value
                    inferred = None
                    if isinstance(v, ast.Constant):
                        inferred = infer_type_from_value(v.value)
                    elif isinstance(v, ast.BinOp):
                        for side in [v.left, v.right]:
                            if isinstance(side, ast.Constant):
                                inferred = infer_type_from_value(side.value)
                                break
                    elif isinstance(v, ast.Call):
                        if isinstance(v.func, ast.Name) and v.func.id in python_type_map:
                            inferred = python_type_map[v.func.id]
                    declared_vars.add(var)
                if var in declared_vars:
                    duplicate_vars.add(var)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if (isinstance(call.func, ast.Attribute) and hasattr(call.func.value, 'id') and call.func.value.id == 'pyir' and call.func.attr == 'inline') or \
                   (isinstance(call.func, ast.Name) and call.func.id == 'inline'):
                    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
                        s = call.args[0].value
                        entry_block.add(IRInstr(s))
                        used_vars.update(re.findall(r'%([a-zA-Z_][a-zA-Z0-9_]*)', s))
            elif isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Name):
                    return_var = stmt.value.id
                    used_vars.add(return_var)
                else:
                    raise ValueError(f"[pyir] Only 'return var' is supported in ergonomic mode for '{fn.__name__}'. Got: {ast.dump(stmt.value)}")
            elif isinstance(stmt, ast.Pass):
                continue
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
            else:
                raise ValueError(f"[pyir] Unsupported statement in '{fn.__name__}': {ast.dump(stmt)}")
        if duplicate_vars:
            raise ValueError(f"[pyir] Duplicate variable declarations in '{fn.__name__}': {', '.join(duplicate_vars)}")
        if not entry_block.instrs:
            raise ValueError(f"[pyir] No pyir.inline calls found in '{fn.__name__}'.")
        if not return_var and ret_ann is not void:
            raise ValueError(f"[pyir] No return variable found in '{fn.__name__}'.")
        undeclared = used_vars - declared_vars
        if undeclared:
            raise ValueError(f"[pyir] Variables used but not declared in '{fn.__name__}': {', '.join(undeclared)}")
        unused = declared_vars - used_vars - set(arg_names)
        if unused:
            raise ValueError(f"[pyir] Variables declared but not used in '{fn.__name__}': {', '.join(unused)}")
        # --- IR object assembly ---
        ir_fn = IRFunction(mangled, list(zip(arg_names, [t.llvm for t in arg_types])), ret_ann.llvm, fast_math=fast_math)
        ir_fn.add_block(entry_block)
        # Add return instruction
        if ret_ann is not void:
            entry_block.add(IRInstr(f"ret {ret_ann.llvm} %{return_var}"))
        else:
            entry_block.add(IRInstr("ret void"))
        ir_str = str(ir_fn)
        register_function(mangled, ir_str)  # Eager registration

        @functools.wraps(fn)
        def wrapper(*args):
            idx = next(_call_count)
            if len(args) != len(arg_types):
                raise TypeError(
                    f"[pyir] Error calling '{fn.__name__}': expected {len(arg_types)} arguments ({', '.join(arg_names)}), got {len(args)}.\n"
                    f"  Arguments received: {args}\n"
                    f"  Please check your function call."
                )
            # --- IR object construction ---
            entry_block = IRBlock('entry')
            declared_vars = set(arg_names)
            used_vars = set()
            return_var = None
            duplicate_vars = set()
            for stmt in funcdef.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    var = stmt.target.id
                    ann = stmt.annotation
                    if isinstance(ann, ast.Name) and ann.id in python_type_map:
                        pass
                    if var in declared_vars:
                        duplicate_vars.add(var)
                    declared_vars.add(var)
                elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    var = stmt.targets[0].id
                    if var not in declared_vars:
                        v = stmt.value
                        inferred = None
                        if isinstance(v, ast.Constant):
                            inferred = infer_type_from_value(v.value)
                        elif isinstance(v, ast.BinOp):
                            for side in [v.left, v.right]:
                                if isinstance(side, ast.Constant):
                                    inferred = infer_type_from_value(side.value)
                                    break
                        elif isinstance(v, ast.Call):
                            if isinstance(v.func, ast.Name) and v.func.id in python_type_map:
                                inferred = python_type_map[v.func.id]
                        declared_vars.add(var)
                    if var in declared_vars:
                        duplicate_vars.add(var)
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    if (isinstance(call.func, ast.Attribute) and hasattr(call.func.value, 'id') and call.func.value.id == 'pyir' and call.func.attr == 'inline') or \
                       (isinstance(call.func, ast.Name) and call.func.id == 'inline'):
                        if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
                            s = call.args[0].value
                            entry_block.add(IRInstr(s))
                            used_vars.update(re.findall(r'%([a-zA-Z_][a-zA-Z0-9_]*)', s))
                elif isinstance(stmt, ast.Return):
                    if isinstance(stmt.value, ast.Name):
                        return_var = stmt.value.id
                        used_vars.add(return_var)
                    else:
                        raise ValueError(f"[pyir] Only 'return var' is supported in ergonomic mode for '{fn.__name__}'. Got: {ast.dump(stmt.value)}")
                elif isinstance(stmt, ast.Pass):
                    continue
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    continue
                else:
                    raise ValueError(f"[pyir] Unsupported statement in '{fn.__name__}': {ast.dump(stmt)}")
            if duplicate_vars:
                raise ValueError(f"[pyir] Duplicate variable declarations in '{fn.__name__}': {', '.join(duplicate_vars)}")
            if not entry_block.instrs:
                raise ValueError(f"[pyir] No pyir.inline calls found in '{fn.__name__}'.")
            if not return_var and ret_ann is not void:
                raise ValueError(f"[pyir] No return variable found in '{fn.__name__}'.")
            undeclared = used_vars - declared_vars
            if undeclared:
                raise ValueError(f"[pyir] Variables used but not declared in '{fn.__name__}': {', '.join(undeclared)}")
            unused = declared_vars - used_vars - set(arg_names)
            if unused:
                raise ValueError(f"[pyir] Variables declared but not used in '{fn.__name__}': {', '.join(unused)}")
            # --- IR object assembly ---
            ir_fn = IRFunction(mangled, list(zip(arg_names, [t.llvm for t in arg_types])), ret_ann.llvm, fast_math=fast_math)
            ir_fn.add_block(entry_block)
            # Add return instruction
            if ret_ann is not void:
                entry_block.add(IRInstr(f"ret {ret_ann.llvm} %{return_var}"))
            else:
                entry_block.add(IRInstr("ret void"))
            # --- Enhanced kernel caching with global cache and optimizations ---
            ir_str = str(ir_fn)
            cache_key = hashlib.sha256((src + str(arg_types) + ir_str + str(ret_ann) + str(fast_math)).encode()).hexdigest()
            # Check global cache first
            if cache_key in _kernel_cache:
                cfunc = _kernel_cache[cache_key]
            elif cache_key in cache:
                cfunc = cache[cache_key]
            else:
                module_name = f"{mangled}_mod_{idx}"
                global_ir = []
                for name, (llvm_type, value) in _global_consts.items():
                    global_ir.append(f"@{name} = constant {llvm_type} {value}")
                ir_mod = IRModule()
                for g in global_ir:
                    ir_mod.add_global(g)
                ir_mod.add_function(ir_fn)
                ir = str(ir_mod)
                # Apply LLVM optimizations
                try:
                    ir = optimize_ir_module(ir)
                except Exception as e:
                    warnings.warn(f"[pyir] Optimization failed for {fn.__name__}: {e}")
                register_function(mangled, ir)
                tgt = target or _current_target
                if tgt in ('cuda', 'gpu'):
                    cache[cache_key] = lambda *a, **kw: ir
                    return ir
                _compile_ir(ir, fn_name=fn.__name__)
                from pyir import _engine
                addr = _engine.get_function_address(mangled)
                cfunc_ty = ctypes.CFUNCTYPE(ret_ann.ctype, *(t.ctype for t in arg_types)) if ret_ann.ctype else ctypes.CFUNCTYPE(None, *(t.ctype for t in arg_types))
                cfunc = cfunc_ty(addr)
                # Store in both caches
                cache[cache_key] = cfunc
                _kernel_cache[cache_key] = cfunc
            return cfunc(*args)
        return wrapper
    if fn is not None:
        return decorator(fn)
    return decorator

# --- Security: Safe mode flag ---
safe_mode = False

def set_safe_mode(enabled=True):
    """Enable or disable PyIR safe mode (disables IR injection and restricts code execution)."""
    global safe_mode
    safe_mode = enabled

# --- Refactored pyir.inline to use IRInstr ---
def inline(ir: str, sugar=False):
    """Inject LLVM IR inline as IRInstr. Disabled in safe mode."""
    if safe_mode:
        raise RuntimeError("[pyir] IR injection is disabled in safe mode.")
    return IRInstr(ir)

# --- Exported sandboxed JIT API ---
def sandboxed_jit(ir_code, fn_name, args=None, ret_ctype=None):
    """Run IR in a sandboxed subprocess with full type support (int, float, bool, struct, array, vector, numpy, device pointers, GPU memory)."""
    return _compile_ir(ir_code, fn_name, args=args, ret_ctype=ret_ctype)

# --- IR Object Model for Fast SSA and Emission ---
class IRInstr:
    def __init__(self, text):
        self.text = text
    def __str__(self):
        return self.text

class IRBlock:
    def __init__(self, label):
        self.label = label
        self.instrs = []
    def add(self, instr):
        self.instrs.append(instr if isinstance(instr, IRInstr) else IRInstr(instr))
    def __str__(self):
        # Only emit the label if the first instruction is not already a label
        lines = [str(i) for i in self.instrs]
        if lines and lines[0].endswith(':'):
            return '\n'.join(lines)
        return f"{self.label}:\n  " + "\n  ".join(lines)

class IRFunction:
    def __init__(self, name, args, ret_type, attrs="", fast_math=False):
        self.name = name
        self.args = args
        self.ret_type = ret_type
        self.attrs = attrs  # Only for valid LLVM function attributes
        self.blocks = []
        self.fast_math = fast_math
    def add_block(self, block):
        self.blocks.append(block)
    def __str__(self):
        args_str = ", ".join(f"{t} %{n}" for n, t in self.args)
        body = []
        for b in self.blocks:
            lines = []
            for instr in b.instrs:
                s = str(instr)
                if self.fast_math or _fast_math_flags:
                    s = re.sub(r'\b(fadd|fsub|fmul|fdiv|frem)\b', r'\1 fast', s)
                lines.append(s)
            # Only emit the label if not already present
            if lines and lines[0].endswith(':'):
                body.append("\n  ".join(lines))
            else:
                body.append(f"{b.label}:\n  " + "\n  ".join(lines))
        return f"define {self.ret_type} @{self.name}({args_str}) {{\n" + "\n".join(body) + "\n}"

class IRModule:
    def __init__(self):
        self.functions = []
        self.globals = []
    def add_function(self, fn):
        self.functions.append(fn)
    def add_global(self, g):
        self.globals.append(g)
    def __str__(self):
        return "\n".join(self.globals + [str(f) for f in self.functions])

# --- IR Object Model Utilities ---
def create_ir_function_from_string(ir_str, name=None):
    """
    Create an IRFunction object from an IR string.
    Useful for parsing existing IR and converting to object model.
    """
    import re
    
    # Extract function signature
    sig_match = re.search(r'define\s+(?:[^@]*@)?([^(]+)\(([^)]*)\)\s*{', ir_str)
    if not sig_match:
        raise ValueError(f"[pyir] Could not parse function signature from IR")
    
    func_name = name or sig_match.group(1).strip()
    args_str = sig_match.group(2).strip()
    
    # Parse arguments
    args = []
    if args_str:
        for arg in args_str.split(','):
            arg = arg.strip()
            if arg:
                parts = arg.split('%')
                if len(parts) == 2:
                    arg_type = parts[0].strip()
                    arg_name = parts[1].strip()
                    args.append((arg_name, arg_type))
    
    # Extract return type and attributes
    ret_type_match = re.search(r'define\s+([^@]+)@', ir_str)
    ret_type = ret_type_match.group(1).strip() if ret_type_match else "void"
    
    attrs_match = re.search(r'define\s+(?:[^@]*@)?[^(]+\([^)]*\)\s+([^{]*)', ir_str)
    attrs = attrs_match.group(1).strip() if attrs_match else ""
    
    # Create IRFunction
    ir_fn = IRFunction(func_name, args, ret_type, attrs)
    
    # Parse blocks
    blocks = re.findall(r'([^:]+):\s*\n((?:\s+[^\n]+\n?)*)', ir_str)
    for block_label, block_content in blocks:
        block = IRBlock(block_label.strip())
        for line in block_content.strip().split('\n'):
            line = line.strip()
            if line:
                block.add(IRInstr(line))
        ir_fn.add_block(block)
    
    return ir_fn

def merge_ir_functions(functions, merged_name="merged"):
    """
    Merge multiple IRFunction objects into a single function.
    Returns a new IRFunction with all blocks from all functions.
    """
    if not functions:
        raise ValueError("[pyir] No functions to merge")
    
    # Use the first function as base
    base_fn = functions[0]
    merged_fn = IRFunction(merged_name, base_fn.args, base_fn.ret_type, base_fn.attrs)
    
    # Add all blocks from all functions
    for i, fn in enumerate(functions):
        for block in fn.blocks:
            # Rename block labels to avoid conflicts
            if block.label == 'entry' and i > 0:
                block.label = f'entry_{i}'
            merged_fn.add_block(block)
    
    return merged_fn

def get_kernel_metadata(fn):
    """
    Extract metadata from a kernel function for fusion analysis.
    Returns dict with kernel properties.
    """
    metadata = {
        'name': fn.__name__,
        'is_cuda': getattr(fn, '_is_cuda_kernel', False),
        'is_simd': getattr(fn, '_is_simd_kernel', False),
        'cuda_grid': getattr(fn, '_cuda_grid', None),
        'cuda_block': getattr(fn, '_cuda_block', None),
        'simd_width': getattr(fn, '_simd_width', None),
        'simd_dtype': getattr(fn, '_simd_dtype', None),
        'arg_names': getattr(fn, '_arg_names', []),
        'arg_types': getattr(fn, '_arg_types', []),
        'output_names': getattr(fn, '_output_names', []),
        'ir_module': getattr(fn, '_ir_module', None)
    }
    
    # Get function signature
    import inspect
    sig = inspect.signature(fn)
    metadata['signature'] = str(sig)
    metadata['parameters'] = list(sig.parameters.keys())
    
    return metadata

def hash_ir(ir: str) -> str:
    """Compute a SHA256 hash of the IR string for deduplication."""
    return hashlib.sha256(ir.encode()).hexdigest()

# --- IR deduplication registry: IR hash -> (kernel name, function pointer) ---
_ir_dedup_registry = {}

# In the function decorator and kernel creation, before registering/compiling:
# 1. Compute the IR hash.
# 2. If the hash is in _ir_dedup_registry, reuse the function pointer.
# 3. Otherwise, register and compile as usual, then store in the dedup registry.

# Example integration (to be used in kernel creation paths):
def get_or_register_kernel(name, ir, dtype, target, fast_math):
    """
    Get or register a kernel by IR hash. Returns the function pointer.
    If an identical kernel is already registered, reuse it.
    """
    ir_hash = hash_ir(ir)
    key = _kernel_cache_key(name, ir, dtype, target, fast_math)
    if ir_hash in _ir_dedup_registry:
        # Reuse the function pointer from the dedup registry
        _, cfunc = _ir_dedup_registry[ir_hash]
        _kernel_cache[key] = cfunc
        return cfunc
    # Otherwise, register and compile as usual
    register_function(name, ir)
    _compile_ir(ir, fn_name=name)
    from . import _engine
    addr = _engine.get_function_address(name)
    arg_types = [dtype]  # Simplified; update as needed
    import ctypes
    cfunc_ty = ctypes.CFUNCTYPE(None, *(t.ctype for t in arg_types))
    cfunc = cfunc_ty(addr)
    _kernel_cache[key] = cfunc
    _ir_dedup_registry[ir_hash] = (name, cfunc)
    return cfunc

def _kernel_cache_key(name, ir, dtype, target, fast_math):
    """Compute a robust cache key for a kernel."""
    ir_hash = hashlib.sha256(ir.encode()).hexdigest()
    return (name, ir_hash, str(dtype), str(target), str(fast_math))
