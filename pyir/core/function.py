"""
pyir.core.function: Ergonomic function decorator and registration utilities for PyIR
"""
import functools
import inspect
import textwrap
import ast
import re
import hashlib
import warnings
import itertools
import threading
import ctypes
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, RLock
import weakref

from typing import Callable, Any, Dict, List, Optional, NamedTuple
from .._engine import pyir_debug
from .ir import *
from .consts import _global_consts
from .aot import _current_target
from .cache import _kernel_cache
from ..typing import *
from .jit import jit_compile_ir
from .registry import _function_registry, register_function
from .optims import optimize_ir_module, OptimizationPreset

_call_count = itertools.count()

import threading
_atomic_counter = itertools.count()
_atomic_counter_lock = threading.Lock()

def atomic_increment():
    """Atomic increment operation."""
    with _atomic_counter_lock:
        return next(_atomic_counter)

_ast_cache = {}
_ast_cache_lock = threading.Lock()

_type_inference_cache = {}
_type_inference_lock = threading.Lock()

_signature_cache = {}
_signature_cache_lock = threading.Lock()

_optimized_type_map = {
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
    'i8': int8, 'i16': int16, 'i32': int32, 'i64': int64,
    'f16': float16, 'f32': float32, 'f64': float64,
    'c64': complex64, 'c128': complex128,
}

_allowed_types_set = {
    IntType, FloatType, VectorType, StructType, ArrayType, 
    VoidType, PointerType, FunctionPointerType, OpaqueType
}

class FunctionMetadata(NamedTuple):
    """Metadata for a function that will be compiled lazily."""
    fn: Callable
    arg_types: List[Any]
    arg_names: List[str]
    ret_type: Any
    mangled_name: str
    fast_math: bool
    opt_level: Optional[int]
    preset: Optional[OptimizationPreset]
    no_optims: bool
    target: Optional[str]
    src_ast: ast.AST
    funcdef: ast.FunctionDef
    src_file: str
    src_line: int
    created_time: float

_function_metadata_registry: Dict[str, FunctionMetadata] = {}
_metadata_registry_lock = threading.Lock()

_compiled_functions: Dict[str, Any] = {}
_compiled_functions_lock = threading.Lock()

_async_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pyir_compiler")
_async_executor_lock = threading.Lock()

def get_async_executor():
    """Get or create the async executor."""
    global _async_executor
    with _async_executor_lock:
        if _async_executor is None or _async_executor._shutdown:
            _async_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pyir_compiler")
    return _async_executor

def get_cached_ast(fn):
    """Get cached AST for a function, or parse and cache it."""
    fn_id = id(fn)
    
    # Fast path: check without lock first
    if fn_id in _ast_cache:
        return _ast_cache[fn_id]
    
    # Only lock if we need to parse
    with _ast_cache_lock:
        if fn_id in _ast_cache:
            return _ast_cache[fn_id]
        
        # Parse and cache
        src = inspect.getsource(fn)
        src = textwrap.dedent(src)
        src_ast = ast.parse(src)
        _ast_cache[fn_id] = src_ast
        
        # Limit cache size atomically
        if len(_ast_cache) > 1000:
            oldest_keys = list(_ast_cache.keys())[:100]
            for key in oldest_keys:
                del _ast_cache[key]
        
        return src_ast

def fast_type_inference(fn, param_name, funcdef):
    """Optimized type inference with atomic caching."""
    cache_key = (id(fn), param_name)
    
    # Fast path: check without lock first
    if cache_key in _type_inference_cache:
        return _type_inference_cache[cache_key]
    
    # Only lock if we need to compute
    with _type_inference_lock:
        if cache_key in _type_inference_cache:
            return _type_inference_cache[cache_key]
    
    # Fast path: check for direct assignments first
    for stmt in funcdef.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            if stmt.targets[0].id == param_name:
                v = stmt.value
            if isinstance(v, ast.Constant):
                inferred = infer_type_from_value(v.value)
                with _type_inference_lock:
                    _type_inference_cache[cache_key] = inferred
                return inferred
            elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name):
                if v.func.id in _optimized_type_map:
                    inferred = _optimized_type_map[v.func.id]
                    with _type_inference_lock:
                        _type_inference_cache[cache_key] = inferred
                    return inferred
    
    # Default fallback
    inferred = int32
    with _type_inference_lock:
        _type_inference_cache[cache_key] = inferred
    return inferred

def fast_type_resolution(ann):
    """Optimized type resolution with minimal lookups."""
    if ann is inspect._empty:
        return int32
    
    # Fast path: direct type objects (including pyir types)
    if type(ann) in _allowed_types_set or hasattr(ann, 'llvm'):
        return ann
    
    # Fast path: optimized type map
    if ann in _optimized_type_map:
        return _optimized_type_map[ann]
    
    # Fast path: string types
    if isinstance(ann, str):
        if ann in _optimized_type_map:
            return _optimized_type_map[ann]
        if ann in named_types:
            return named_types[ann]
    
    # Fallback to original logic
    if ann in python_type_map:
        return python_type_map[ann]
    
    # Handle pyir type objects that might not be in the sets above
    if hasattr(ann, 'llvm') and hasattr(ann, 'ctype'):
        return ann
    
    return ann

def fast_signature_processing(fn, funcdef):
    """Optimized signature processing with atomic caching."""
    cache_key = id(fn)
    
    # Fast path: check without lock first
    if cache_key in _signature_cache:
        return _signature_cache[cache_key]
    
    # Only lock if we need to compute
    with _signature_cache_lock:
        if cache_key in _signature_cache:
            return _signature_cache[cache_key]
    
    sig = inspect.signature(fn)
    arg_types = []
    arg_names = []
    
    for name, param in sig.parameters.items():
        ann = fast_type_resolution(param.annotation)
        
        # Fast type inference for unannotated parameters
        if ann is inspect._empty:
            ann = fast_type_inference(fn, name, funcdef)
        
        # Fast type validation
        if not (type(ann) in _allowed_types_set or hasattr(ann, 'llvm')):
            raise TypeError(
                f"[pyir] Error in function '{fn.__name__}': "
                f"Parameter '{name}' must be annotated with a pyir type, Python int/float/bool, or user-defined type. "
                f"Instead got: {param.annotation!r}"
            )
        
        arg_types.append(ann)
        arg_names.append(name)
    
    # Fast return type resolution
    ret_ann = fast_type_resolution(sig.return_annotation)
    if not (type(ret_ann) in _allowed_types_set or hasattr(ret_ann, 'llvm')):
        raise TypeError(
            f"[pyir] Error in function '{fn.__name__}': "
            f"Return type must be a pyir type, int/float/bool, or user-defined type. "
            f"Instead got: {ret_ann!r}"
        )
    
    result = (arg_types, arg_names, ret_ann)
    with _signature_cache_lock:
        _signature_cache[cache_key] = result
    
    return result

def fast_mangling(fn_name, arg_types, ret_type):
    """Optimized function name mangling."""
    # Pre-compute type strings to avoid repeated .llvm calls
    type_strings = [t.llvm for t in arg_types + [ret_type]]
    return f"{fn_name}__{'_'.join(type_strings)}"

def register_function_metadata(metadata: FunctionMetadata):
    """Register function metadata with atomic operation."""
    with _metadata_registry_lock:
        _function_metadata_registry[metadata.mangled_name] = metadata
        if pyir_debug:
            print(f"[pyir] Registered function metadata: {metadata.mangled_name}")

def get_function_metadata(mangled_name: str) -> Optional[FunctionMetadata]:
    """Get function metadata by mangled name."""
    with _metadata_registry_lock:
        return _function_metadata_registry.get(mangled_name)

async def async_compile_function_from_metadata(metadata: FunctionMetadata) -> Any:
    """
    Asynchronously compile a function from its metadata.
    """
    # Check if already compiled (fast path)
    if metadata.mangled_name in _compiled_functions:
        return _compiled_functions[metadata.mangled_name]
    
    # Use executor for background compilation
    executor = get_async_executor()
    loop = asyncio.get_event_loop()
    
    def compile_sync():
        return compile_function_from_metadata(metadata)
    
    # Run compilation in background thread
    compiled_func = await loop.run_in_executor(executor, compile_sync)
    return compiled_func

def compile_function_from_metadata(metadata: FunctionMetadata) -> Any:
    """
    Compile a function from its metadata. This is the core of lazy compilation.
    """
    # Check if already compiled (fast path)
    if metadata.mangled_name in _compiled_functions:
        return _compiled_functions[metadata.mangled_name]
    
    # Only lock if we need to compile
    with _compiled_functions_lock:
        if metadata.mangled_name in _compiled_functions:
            return _compiled_functions[metadata.mangled_name]
    
    # Get the already registered IR object
    if metadata.mangled_name not in _function_registry:
        raise ValueError(f"[pyir] No IR found for function '{metadata.fn.__name__}' with mangled name '{metadata.mangled_name}'. This should not happen.")
    
    ir_obj = _function_registry[metadata.mangled_name]
    # Convert to string for LLVM/JIT
    ir_str = str(ir_obj)
    
    # Get source code for cache key
    src = inspect.getsource(metadata.fn)
    cache_key = hashlib.sha256((src + str(metadata.arg_types) + ir_str + str(metadata.ret_type) + str(metadata.fast_math) + str(metadata.opt_level) + str(metadata.preset) + str(metadata.no_optims)).encode()).hexdigest()
    
    # Check global cache first
    if cache_key in _kernel_cache:
        compiled_func = _kernel_cache[cache_key]
        with _compiled_functions_lock:
            _compiled_functions[metadata.mangled_name] = compiled_func
        return compiled_func
    
    # Compile and cache - use the IR string directly for JIT
    tgt = metadata.target or _current_target
    if tgt in ('cuda', 'gpu'):
        compiled_func = lambda *a, **kw: ir_str
        _kernel_cache[cache_key] = compiled_func
        with _compiled_functions_lock:
            _compiled_functions[metadata.mangled_name] = compiled_func
        return compiled_func
    
    jit_compile_ir(ir_str, fn_name=metadata.fn.__name__)
    from pyir import _engine
    addr = _engine.get_function_address(metadata.mangled_name)
    cfunc_ty = ctypes.CFUNCTYPE(metadata.ret_type.ctype, *(t.ctype for t in metadata.arg_types)) if metadata.ret_type.ctype else ctypes.CFUNCTYPE(None, *(t.ctype for t in metadata.arg_types))
    compiled_func = cfunc_ty(addr)
    
    # Store in global cache and compiled functions registry
    _kernel_cache[cache_key] = compiled_func
    with _compiled_functions_lock:
        _compiled_functions[metadata.mangled_name] = compiled_func
    
    if pyir_debug:
        print(f"[pyir] Compiled function {metadata.mangled_name} (lazy compilation)")
    
    return compiled_func

def generate_ir_object_from_metadata(metadata: FunctionMetadata):
    entry_block = IRBlock('entry')
    declared_vars = set(metadata.arg_names)
    used_vars = set()
    return_var = None
    duplicate_vars = set()
    for stmt in metadata.funcdef.body:
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
                raise ValueError(f"[pyir] Only 'return var' is supported in ergonomic mode for '{metadata.fn.__name__}'. Got: {ast.dump(stmt.value)}")
        elif isinstance(stmt, ast.Pass):
            continue
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        else:
            raise ValueError(f"[pyir] Unsupported statement in '{metadata.fn.__name__}': {ast.dump(stmt)}")
    if duplicate_vars:
        raise ValueError(f"[pyir] Duplicate variable declarations in '{metadata.fn.__name__}': {', '.join(duplicate_vars)}")
    if not entry_block.instrs:
        raise ValueError(f"[pyir] No pyir.inline calls found in '{metadata.fn.__name__}'.")
    if not return_var and metadata.ret_type is not void:
        raise ValueError(f"[pyir] No return variable found in '{metadata.fn.__name__}': {', '.join(undeclared)}")
    undeclared = used_vars - declared_vars
    if undeclared:
        raise ValueError(f"[pyir] Variables used but not declared in '{metadata.fn.__name__}': {', '.join(undeclared)}")
    unused = declared_vars - used_vars - set(metadata.arg_names)
    if unused:
        raise ValueError(f"[pyir] Variables declared but not used in '{metadata.fn.__name__}': {', '.join(unused)}")
    ir_fn = IRFunction(metadata.mangled_name, list(zip(metadata.arg_names, [t.llvm for t in metadata.arg_types])), metadata.ret_type.llvm, fast_math=metadata.fast_math)
    ir_fn.add_block(entry_block)
    # Add return instruction
    if metadata.ret_type is not void:
        entry_block.add(IRInstr(f"ret {metadata.ret_type.llvm} %{return_var}"))
    else:
        entry_block.add(IRInstr("ret void"))
    return ir_fn

def function(fn=None, *, target=None, cuda_kernel=False, simd=False, simd_width=None, simd_dtype=None, fast_math=True, opt_level=None, preset=None, no_optims=False):
    """
    Decorator for ergonomic LLVM IR kernels with hybrid IR registration and lazy compilation.
    
    This implementation provides:
    - Fast startup: Function metadata and IR are registered immediately
    - Lazy compilation: Functions are only compiled when first called
    - Caching: Compiled functions are cached for subsequent calls
    - Async compilation: Background compilation for better responsiveness
    
    Optimization options:
        - fast_math: Enable fast-math optimizations (default: True)
        - opt_level: LLVM optimization level (0-3, default: global)
        - preset: OptimizationPreset (e.g., OptimizationPreset.PERFORMANCE, OptimizationPreset.SAFE)
        - no_optims: Disable all optimizations including autovectorization (default: False)
    
    By default, PyIR automatically applies:
        - Fast-math optimizations for floating-point operations
        - LLVM optimizations (level 2 by default)
        - Loop vectorization and SLP vectorization
        - Function inlining and loop unrolling
    
    Use no_optims=True only for debugging or when you need exact bit-for-bit reproducibility.
    """
    def decorator(fn):
        src_ast = get_cached_ast(fn)
        funcdef = src_ast.body[0]
        
        # Fast signature processing with caching
        arg_types, arg_names, ret_ann = fast_signature_processing(fn, funcdef)
        
        # Fast mangling
        mangled = fast_mangling(fn.__name__, arg_types, ret_ann)
        
        # Fast file/line extraction (cached)
        src_file = getattr(fn, '_pyir_src_file', None)
        if src_file is None:
            src_file = inspect.getsourcefile(fn) or "<unknown file>"
            fn._pyir_src_file = src_file
        
        src_line = getattr(fn, '_pyir_src_line', None)
        if src_line is None:
            src_line = inspect.getsourcelines(fn)[1]
            fn._pyir_src_line = src_line
        
        # Create and register metadata immediately (fast startup)
        metadata = FunctionMetadata(
            fn=fn,
            arg_types=arg_types,
            arg_names=arg_names,
            ret_type=ret_ann,
            mangled_name=mangled,
            fast_math=fast_math,
            opt_level=opt_level,
            preset=preset,
            no_optims=no_optims,
            target=target,
            src_ast=src_ast,
            funcdef=funcdef,
            src_file=src_file,
            src_line=src_line,
            created_time=time.time()
        )
        
        register_function_metadata(metadata)
        
        # Generate IR object and register it immediately
        ir_fn = generate_ir_object_from_metadata(metadata)
        # Store the IR object in the registry
        register_function(metadata.mangled_name, ir_fn)
        
        # --- LAZY COMPILATION: Compile only when called ---
        @functools.wraps(fn)
        def wrapper(*args):
            if len(args) != len(arg_types):
                raise TypeError(
                    f"[pyir] Error calling '{fn.__name__}': expected {len(arg_types)} arguments ({', '.join(arg_names)}), got {len(args)}.\n"
                    f"  Arguments received: {args}\n"
                    f"  Please check your function call."
                )
            
            # Fast path: check if already compiled
            if mangled in _compiled_functions:
                compiled_func = _compiled_functions[mangled]
            else:
                compiled_func = compile_function_from_metadata(metadata)
            
            return compiled_func(*args)
        
        # Mark async kernels
        if inspect.iscoroutinefunction(fn):
            wrapper._is_async_kernel = True
        
        # Store metadata reference for introspection
        wrapper._metadata = metadata
        
        return wrapper
    
    if fn is not None:
        return decorator(fn)
    return decorator

# Utility functions for the hybrid approach
def get_compiled_function_count() -> int:
    """Get the number of compiled functions."""
    with _compiled_functions_lock:
        return len(_compiled_functions)

def get_metadata_function_count() -> int:
    """Get the number of registered function metadata."""
    with _metadata_registry_lock:
        return len(_function_metadata_registry)

def clear_compiled_functions():
    """Clear all compiled functions (metadata remains)."""
    with _compiled_functions_lock:
        _compiled_functions.clear()

def clear_caches():
    """Clear all caches for memory management."""
    with _ast_cache_lock:
        _ast_cache.clear()
    with _type_inference_lock:
        _type_inference_cache.clear()
    with _signature_cache_lock:
        _signature_cache.clear()

def shutdown_async_executor():
    """Shutdown the async executor."""
    global _async_executor
    with _async_executor_lock:
        if _async_executor and not _async_executor._shutdown:
            _async_executor.shutdown(wait=False)
            _async_executor = None

def get_function_stats() -> Dict[str, Any]:
    """Get statistics about function registration and compilation."""
    with _metadata_registry_lock:
        with _compiled_functions_lock:
            return {
                'metadata_count': len(_function_metadata_registry),
                'compiled_count': len(_compiled_functions),
                'compilation_ratio': len(_compiled_functions) / max(len(_function_metadata_registry), 1),
                'metadata_names': list(_function_metadata_registry.keys()),
                'compiled_names': list(_compiled_functions.keys()),
                'ast_cache_size': len(_ast_cache),
                'type_inference_cache_size': len(_type_inference_cache),
                'signature_cache_size': len(_signature_cache),
                'async_executor_active': _async_executor is not None and not _async_executor._shutdown
            }

# Export the registry for use by fusion and other modules
__all__ = [
    "function", "register_function", "_function_registry",
    "register_function_metadata", "get_function_metadata", "compile_function_from_metadata",
    "async_compile_function_from_metadata", "get_compiled_function_count", 
    "get_metadata_function_count", "clear_compiled_functions", "clear_caches",
    "shutdown_async_executor", "get_function_stats"
]
