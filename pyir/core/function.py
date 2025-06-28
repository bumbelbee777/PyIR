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
from .._engine import pyir_debug, _local
from .ir import *
from .consts import _global_consts
from .aot import _current_target
from .cache import _kernel_cache
from ..typing import *
from .jit import jit_compile_ir
from .registry import _function_registry, register_function
from .optims import optimize_ir_module, OptimizationPreset
from pyir.fusion import fuse_kernels
from ..helpers import fast_type_resolution, fast_mangling

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
    VoidType, PointerType, FunctionPointerType, OpaqueType, tuple
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
    
    # Handle tuple types (for return types)
    if ann is tuple:
        return tuple
    
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
    if not (type(ret_ann) in _allowed_types_set or hasattr(ret_ann, 'llvm') or ret_ann is tuple):
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
    type_strings = []
    for t in arg_types:
        if hasattr(t, 'llvm'):
            type_strings.append(t.llvm)
        elif t is tuple:
            type_strings.append('tuple')
        else:
            type_strings.append(str(t))
    # Special handling for tuple return type
    if ret_type is tuple:
        # Try to infer tuple size from the function signature (if available)
        tuple_size = 2  # Default to 2 if not available
        # This is a hack, but works for our ergonomic kernels
        type_strings.append('tuple2')
    elif hasattr(ret_type, 'llvm'):
        type_strings.append(ret_type.llvm)
    else:
        type_strings.append(str(ret_type))
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

# Define Tuple2f at the module level for tuple returns
class Tuple2f(ctypes.Structure):
    _fields_ = [('x', ctypes.c_float), ('y', ctypes.c_float)]

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
    
    # Handle tuple returns: wrap IRFunction in IRModule to emit struct type declarations
    if metadata.ret_type is tuple:
        from .ir import IRModule
        ir_module = IRModule()
        ir_module.add_function(ir_obj)
        ir_str = str(ir_module)
    else:
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
    # Handle tuple return: use Tuple2f as the return type (for 2-element float tuples)
    if metadata.ret_type is tuple:
        cfunc_ty = ctypes.CFUNCTYPE(None, *(t.ctype for t in metadata.arg_types), ctypes.POINTER(Tuple2f))
    else:
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
    """Generate IR object from function metadata with proper SSA variable handling."""
    entry_block = IRBlock('entry')
    declared_vars = set(metadata.arg_names)
    used_vars = set()
    return_var = None
    duplicate_vars = set()
    ssa_counter = {}  # Track SSA variable counters
    
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
                    # Process SSA variable assignments to ensure uniqueness
                    lines = s.strip().split('\n')
                    processed_lines = []
                    var_mapping = {}  # Track variable name mappings
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith('%') and '=' in line:
                            # Extract variable name and generate unique SSA name
                            var_match = re.match(r'(%[a-zA-Z_][a-zA-Z0-9_]*)\s*=', line)
                            if var_match:
                                var_name = var_match.group(1)
                                base_name = var_name[1:]  # Remove % prefix
                                
                                # Track this variable assignment
                                if base_name not in ssa_counter:
                                    ssa_counter[base_name] = 0
                                ssa_counter[base_name] += 1
                                
                                # Create unique name for this assignment
                                if ssa_counter[base_name] == 1:
                                    new_var = var_name  # Keep original name for first assignment
                                else:
                                    new_var = f"%{base_name}{ssa_counter[base_name]}"
                                
                                # Store mapping from original to new name
                                var_mapping[var_name] = new_var
                                # Add the new SSA variable to declared_vars (strip %)
                                declared_vars.add(new_var[1:])
                                # Replace the assignment target
                                line = re.sub(rf'{re.escape(var_name)}(?=\s*=)', new_var, line)
                                
                                # Replace any uses of previous variables in the RHS
                                for old_var, new_var_prev in var_mapping.items():
                                    if old_var != var_name:  # Don't replace the current assignment target
                                        line = re.sub(rf'{re.escape(old_var)}(?![a-zA-Z0-9_])', new_var_prev, line)
                        
                        processed_lines.append(line)
                    
                    s = '\n'.join(processed_lines)
                    entry_block.add(IRInstr(s))
                    used_vars.update(re.findall(r'%([a-zA-Z_][a-zA-Z0-9_]*)', s))
        elif isinstance(stmt, ast.Return):
            if isinstance(stmt.value, ast.Name):
                return_var = stmt.value.id
                used_vars.add(return_var)
            elif isinstance(stmt.value, ast.Tuple):
                # Only collect variable names and mark for tuple return lowering
                return_vars = []
                for elt in stmt.value.elts:
                    if isinstance(elt, ast.Name):
                        return_vars.append(elt.id)
                        used_vars.add(elt.id)
                    else:
                        raise ValueError(f"[pyir] Only variable names are supported in tuple returns for '{metadata.fn.__name__}'. Got: {ast.dump(elt)}")
                # Mark for tuple lowering, but do not emit any IR here
                tuple_var_names = return_vars
                return_var = None
            else:
                raise ValueError(f"[pyir] Only 'return var' or 'return (var1, var2, ...)' is supported in ergonomic mode for '{metadata.fn.__name__}'. Got: {ast.dump(stmt.value)}")
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
    
    # Check for undeclared variables
    undeclared = used_vars - declared_vars
    if undeclared:
        raise ValueError(f"[pyir] Variables used but not declared in '{metadata.fn.__name__}': {', '.join(undeclared)}")
    
    # Check for unused variables (only if we have a single return variable)
    if return_var is not None:
        unused = declared_vars - used_vars - set(metadata.arg_names)
        if unused:
            raise ValueError(f"[pyir] Variables declared but not used in '{metadata.fn.__name__}': {', '.join(unused)}")
    
    # Check return variable (only for single returns, not tuple returns)
    if return_var is None and metadata.ret_type is not void and metadata.ret_type is not tuple:
        raise ValueError(f"[pyir] No return variable found in '{metadata.fn.__name__}'")

    tuple_return = False
    tuple_size = 0
    tuple_struct_name = None
    tuple_var_names = []
    if metadata.ret_type is tuple:
        # Find tuple size and variable names from the return statement
        for stmt in metadata.funcdef.body:
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Tuple):
                tuple_size = len(stmt.value.elts)
                tuple_var_names = [elt.id for elt in stmt.value.elts]
                break
        tuple_struct_name = f'%struct.{metadata.fn.__name__}_tuple{tuple_size}'
        tuple_struct_type = '{' + ', '.join(['float'] * tuple_size) + '}'
        tuple_return = True

    # Prepare IR function signature
    ir_arg_types = [t.llvm for t in metadata.arg_types]
    ir_arg_names = list(metadata.arg_names)
    if tuple_return:
        ir_arg_types.append(f'{tuple_struct_name}*')
        ir_arg_names.append('out')
        ir_ret_type = 'void'
    else:
        ir_ret_type = metadata.ret_type.llvm

    ir_fn = IRFunction(metadata.mangled_name, list(zip(ir_arg_names, ir_arg_types)), ir_ret_type, fast_math=metadata.fast_math)
    ir_fn.add_block(entry_block)

    # Add return/store instruction
    if tuple_return:
        # Build the struct step by step
        prev_var = None
        for i, var_name in enumerate(tuple_var_names):
            if i == 0:
                prev_var = 'struct_result'
                entry_block.add(IRInstr(f'%{prev_var} = insertvalue {tuple_struct_name} undef, float %{var_name}, {i}'))
            else:
                current_var = f'struct_result_{i}'
                entry_block.add(IRInstr(f'%{current_var} = insertvalue {tuple_struct_name} %{prev_var}, float %{var_name}, {i}'))
                prev_var = current_var
        # Store the struct to the output pointer
        entry_block.add(IRInstr(f'store {tuple_struct_name} %{prev_var}, {tuple_struct_name}* %out'))
        entry_block.add(IRInstr('ret void'))
    elif return_var is not None:
        if metadata.ret_type is not void:
            # Find the final variable name for the return variable
            final_return_var = return_var
            if return_var in ssa_counter and ssa_counter[return_var] > 1:
                final_return_var = f"{return_var}{ssa_counter[return_var]}"
            entry_block.add(IRInstr(f"ret {metadata.ret_type.llvm} %{final_return_var}"))
        else:
            entry_block.add(IRInstr("ret void"))
    # Note: tuple returns are already handled above

    # If tuple_return, emit the struct type at the top
    if tuple_return:
        ir_fn._struct_type_decl = f'{tuple_struct_name} = type {tuple_struct_type}'
    return ir_fn

# Autofusion infrastructure
_autofusion_queue = []
_autofusion_maxlen = 2  # Default fusion window size
_autofusion_history = set()  # Track fused kernel name tuples to avoid redundant fusions
_fusion_cache = {}  # Cache fused kernels by mangled name tuple

def set_autofusion_window(size):
    """Set the autofusion window size (number of recent kernels to consider)."""
    global _autofusion_maxlen
    _autofusion_maxlen = size

def can_autofuse(fns):
    """
    Smarter heuristic: check if all functions are compatible for fusion.
    - Same argument names/types
    - Same return type
    - Same device/kernel attributes (e.g., simd, cuda, async)
    - All are pure/elementwise (no side effects)
    - Not already fused together
    """
    if len(fns) < 2:
        if pyir_debug:
            print(f"[pyir.autofusion] Too few functions: {len(fns)}")
        return False
    # Check for per-function opt-out
    if any(getattr(fn, '_no_autofusion', False) for fn in fns):
        if pyir_debug:
            print(f"[pyir.autofusion] Function opted out: {[f.__name__ for f in fns if getattr(f, '_no_autofusion', False)]}")
        return False
    # All must have the same argument names, types, and return type
    sigs = [inspect.signature(fn) for fn in fns]
    arg_names = [tuple(sig.parameters.keys()) for sig in sigs]
    arg_types = [tuple(p.annotation for p in sig.parameters.values()) for sig in sigs]
    ret_types = [sig.return_annotation for sig in sigs]
    if not (all(arg_names[0] == names for names in arg_names) and
            all(arg_types[0] == types for types in arg_types) and
            all(ret_types[0] == r for r in ret_types)):
        if pyir_debug:
            print(f"[pyir.autofusion] Signature mismatch: names={arg_names}, types={arg_types}, ret_types={ret_types}")
        return False
    
    # Check device/kernel attributes (simd, cuda, async) - must be EXACTLY the same
    def get_attr(fn, attr):
        # Check on wrapper, _metadata, and original function
        val = getattr(fn, attr, False)
        meta = getattr(fn, '_metadata', None)
        if meta is not None:
            val = val or getattr(meta, attr, False)
            orig = getattr(meta, 'fn', None)
            if orig is not None:
                val = val or getattr(orig, attr, False)
        orig = getattr(fn, 'fn', None)
        if orig is not None:
            val = val or getattr(orig, attr, False)
        return val
    
    # Check each attribute individually - all functions must have the same value
    simd_attrs = [get_attr(fn, '_is_simd_kernel') for fn in fns]
    cuda_attrs = [get_attr(fn, '_is_cuda_kernel') for fn in fns]
    async_attrs = [get_attr(fn, '_is_async_kernel') for fn in fns]
    
    if pyir_debug:
        print(f"[pyir.autofusion] Attributes: simd={simd_attrs}, cuda={cuda_attrs}, async={async_attrs}")
    
    if not (all(simd_attrs[0] == a for a in simd_attrs) and
            all(cuda_attrs[0] == a for a in cuda_attrs) and
            all(async_attrs[0] == a for a in async_attrs)):
        if pyir_debug:
            print(f"[pyir.autofusion] Attribute mismatch: simd={simd_attrs}, cuda={cuda_attrs}, async={async_attrs}")
        return False
    
    # Check for elementwise/pure (must not have _has_side_effects on wrapper or original)
    def has_side_effects(fn):
        # Check on wrapper, _metadata, and original function
        val = getattr(fn, '_has_side_effects', False)
        meta = getattr(fn, '_metadata', None)
        if meta is not None:
            val = val or getattr(meta, '_has_side_effects', False)
            orig = getattr(meta, 'fn', None)
            if orig is not None:
                val = val or getattr(orig, '_has_side_effects', False)
        orig = getattr(fn, 'fn', None)
        if orig is not None:
            val = val or getattr(orig, '_has_side_effects', False)
        # Also check if the original function has side effects set after definition
        if hasattr(fn, '__wrapped__'):
            val = val or getattr(fn.__wrapped__, '_has_side_effects', False)
        return val
    
    # If ANY function has side effects, don't fuse
    side_effects = [has_side_effects(fn) for fn in fns]
    if pyir_debug:
        print(f"[pyir.autofusion] Side effects: {side_effects}")
    if any(side_effects):
        if pyir_debug:
            print(f"[pyir.autofusion] Side effects detected: {[f.__name__ for f, se in zip(fns, side_effects) if se]}")
        return False
    
    # Check fusion history to avoid redundant fusions
    kernel_names = tuple(getattr(fn, '__name__', str(fn)) for fn in fns)
    if kernel_names in _autofusion_history:
        if pyir_debug:
            print(f"[pyir.autofusion] Already fused: {kernel_names}")
        return False
    
    if pyir_debug:
        print(f"[pyir.autofusion] Can autofuse: {kernel_names}")
    return True

def function(fn=None, *, target=None, cuda_kernel=False, simd=False, simd_width=None, simd_dtype=None, fast_math=True, opt_level=None, preset=None, no_optims=False, autofusion=True):
    """
    Decorator for ergonomic LLVM IR kernels with hybrid IR registration and lazy compilation.
    Now supports robust autofusion: automatically fuses compatible recently-called kernels.
    
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
        arg_types, arg_names, ret_ann = fast_signature_processing(fn, funcdef)
        mangled = fast_mangling(fn.__name__, arg_types, ret_ann)
        src_file = getattr(fn, '_pyir_src_file', None)
        if src_file is None:
            src_file = inspect.getsourcefile(fn) or "<unknown file>"
            fn._pyir_src_file = src_file
        src_line = getattr(fn, '_pyir_src_line', None)
        if src_line is None:
            src_line = inspect.getsourcelines(fn)[1]
            fn._pyir_src_line = src_line
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
        ir_fn = generate_ir_object_from_metadata(metadata)
        register_function(metadata.mangled_name, ir_fn)

        tuple_return = (ret_ann is tuple)
        tuple_size = 0
        if tuple_return:
            for stmt in funcdef.body:
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Tuple):
                    tuple_size = len(stmt.value.elts)
                    break
            # Only support 2-element float tuples for now
            from pyir.core.function import Tuple2f  # Use the module-level Tuple2f

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args):
                fused = getattr(wrapper, '_fused', None)
                if fused is not None and fused is not wrapper and getattr(fused, '_is_fused', False):
                    if getattr(fused, '_is_async_kernel', False):
                        return await fused(*args)
                    else:
                        return fused(*args)
                # Only run autofusion logic if not a fused kernel and not in a fused kernel call
                if autofusion and not getattr(fn, '_no_autofusion', False) and not getattr(_local, 'in_autofusion', False) and not getattr(wrapper, '_is_fused', False) and not getattr(_local, 'in_fused_kernel', False):
                    if pyir_debug:
                        print(f"[pyir.autofusion] Running autofusion for {wrapper.__name__}, in_fused_kernel={getattr(_local, 'in_fused_kernel', False)}")
                    _local.in_autofusion = True
                    try:
                        if pyir_debug:
                            print(f"[pyir.autofusion] Adding {wrapper.__name__} to queue")
                        _autofusion_queue.append(wrapper)
                        if len(_autofusion_queue) > _autofusion_maxlen:
                            _autofusion_queue.pop(0)
                        for window in range(_autofusion_maxlen, 1, -1):
                            candidates = list(_autofusion_queue)[-window:]
                            if pyir_debug:
                                print(f"[pyir.autofusion] Checking window {window}: {[f.__name__ for f in candidates]}")
                            if can_autofuse(candidates):
                                mangled_names = tuple(getattr(f, '_metadata', None).mangled_name if hasattr(f, '_metadata') and getattr(f, '_metadata', None) is not None else getattr(f, '__name__', str(f)) for f in candidates)
                                is_async = any(getattr(f, '_is_async_kernel', False) for f in candidates)
                                fusion_key = (mangled_names, is_async)
                                if fusion_key in _fusion_cache:
                                    fused = _fusion_cache[fusion_key]
                                    if pyir_debug:
                                        print(f"[pyir.autofusion] Using cached fused kernel for: {fusion_key}")
                                    if is_async:
                                        fused._is_async_kernel = True
                                else:
                                    kernel_names = tuple(getattr(f, '__name__', str(f)) for f in candidates)
                                    if pyir_debug:
                                        print(f"[pyir.autofusion] Attempting fusion for: {kernel_names}")
                                    try:
                                        fused = fuse_kernels(candidates)
                                        fused._fusion_history = getattr(fused, '_fusion_history', []) + [kernel_names]
                                        fused._is_fused = True
                                        if is_async:
                                            fused._is_async_kernel = True
                                        _fusion_cache[fusion_key] = fused
                                        if pyir_debug:
                                            print(f"[pyir.autofusion] Fused kernels: {kernel_names}")
                                    except Exception as e:
                                        if pyir_debug:
                                            print(f"[pyir.autofusion] Fusion failed: {e}")
                                        continue
                                for f in candidates:
                                    if f is not fused:
                                        f._fused = fused
                                _autofusion_history.add(mangled_names)
                                break
                    finally:
                        _local.in_autofusion = False
                if len(args) != len(arg_types):
                    raise TypeError(
                        f"[pyir] Error calling '{fn.__name__}': expected {len(arg_types)} arguments ({', '.join(arg_names)}), got {len(args)}.\n"
                        f"  Arguments received: {args}\n"
                        f"  Please check your function call."
                    )
                if mangled in _compiled_functions:
                    compiled_func = _compiled_functions[mangled]
                else:
                    compiled_func = await async_compile_function_from_metadata(metadata)
                executor = get_async_executor()
                loop = asyncio.get_event_loop()
                if tuple_return:
                    out_struct = Tuple2f()
                    await loop.run_in_executor(executor, compiled_func, *(list(args) + [ctypes.byref(out_struct)]))
                    return (out_struct.x, out_struct.y)
                else:
                    return await loop.run_in_executor(executor, compiled_func, *args)
        else:
            @functools.wraps(fn)
            def wrapper(*args):
                fused = getattr(wrapper, '_fused', None)
                if fused is not None and fused is not wrapper and getattr(fused, '_is_fused', False):
                    if getattr(fused, '_is_async_kernel', False):
                        return asyncio.run(fused(*args))
                    else:
                        return fused(*args)
                # Only run autofusion logic if not a fused kernel and not in a fused kernel call
                if autofusion and not getattr(fn, '_no_autofusion', False) and not getattr(_local, 'in_autofusion', False) and not getattr(wrapper, '_is_fused', False) and not getattr(_local, 'in_fused_kernel', False):
                    if pyir_debug:
                        print(f"[pyir.autofusion] Running autofusion for {wrapper.__name__}, in_fused_kernel={getattr(_local, 'in_fused_kernel', False)}")
                    _local.in_autofusion = True
                    try:
                        if pyir_debug:
                            print(f"[pyir.autofusion] Adding {wrapper.__name__} to queue")
                        _autofusion_queue.append(wrapper)
                        if len(_autofusion_queue) > _autofusion_maxlen:
                            _autofusion_queue.pop(0)
                        for window in range(_autofusion_maxlen, 1, -1):
                            candidates = list(_autofusion_queue)[-window:]
                            if pyir_debug:
                                print(f"[pyir.autofusion] Checking window {window}: {[f.__name__ for f in candidates]}")
                            if can_autofuse(candidates):
                                mangled_names = tuple(getattr(f, '_metadata', None).mangled_name if hasattr(f, '_metadata') and getattr(f, '_metadata', None) is not None else getattr(f, '__name__', str(f)) for f in candidates)
                                is_async = any(getattr(f, '_is_async_kernel', False) for f in candidates)
                                fusion_key = (mangled_names, is_async)
                                if fusion_key in _fusion_cache:
                                    fused = _fusion_cache[fusion_key]
                                    if pyir_debug:
                                        print(f"[pyir.autofusion] Using cached fused kernel for: {fusion_key}")
                                    if is_async:
                                        fused._is_async_kernel = True
                                else:
                                    kernel_names = tuple(getattr(f, '__name__', str(f)) for f in candidates)
                                    if pyir_debug:
                                        print(f"[pyir.autofusion] Attempting fusion for: {kernel_names}")
                                    try:
                                        fused = fuse_kernels(candidates)
                                        fused._fusion_history = getattr(fused, '_fusion_history', []) + [kernel_names]
                                        fused._is_fused = True
                                        if is_async:
                                            fused._is_async_kernel = True
                                        _fusion_cache[fusion_key] = fused
                                        if pyir_debug:
                                            print(f"[pyir.autofusion] Fused kernels: {kernel_names}")
                                    except Exception as e:
                                        if pyir_debug:
                                            print(f"[pyir.autofusion] Fusion failed: {e}")
                                        continue
                                for f in candidates:
                                    if f is not fused:
                                        f._fused = fused
                                _autofusion_history.add(mangled_names)
                                break
                    finally:
                        _local.in_autofusion = False
                if len(args) != len(arg_types):
                    raise TypeError(
                        f"[pyir] Error calling '{fn.__name__}': expected {len(arg_types)} arguments ({', '.join(arg_names)}), got {len(args)}.\n"
                        f"  Arguments received: {args}\n"
                        f"  Please check your function call."
                    )
                if mangled in _compiled_functions:
                    compiled_func = _compiled_functions[mangled]
                else:
                    compiled_func = compile_function_from_metadata(metadata)
                if tuple_return:
                    out_struct = Tuple2f()
                    compiled_func(*(list(args) + [ctypes.byref(out_struct)]))
                    return (out_struct.x, out_struct.y)
                else:
                    return compiled_func(*args)
        # Propagate device/kernel attributes based on decorator parameters
        if simd:
            wrapper._is_simd_kernel = True
        if cuda_kernel:
            wrapper._is_cuda_kernel = True
        if inspect.iscoroutinefunction(fn):
            wrapper._is_async_kernel = True
        # Always set _fused attribute (default None)
        wrapper._fused = None
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
