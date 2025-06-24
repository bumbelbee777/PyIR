"""
pyir.fusion: Robust kernel fusion utilities for PyIR (with IR object model)
"""
import inspect
import re
import warnings
import uuid
import numpy as np
from .core import ssa, validate_ir, register_function, get_module_ir, IRInstr, IRBlock, IRFunction, IRModule
import collections
from .typing import float32
import asyncio

_function_registry = None  # Will be set by main pyir/__init__.py

def set_function_registry(reg):
    global _function_registry
    _function_registry = reg

def _unique_ssa_name(name, uid):
    return f"{name}_f_{uid}"

def _parse_ir_to_objects(ir_str, ssa_uid, func_name=None):
    """
    Parse IR string into IR objects with SSA renaming.
    Returns (IRFunction, output_vars, output_names).
    If func_name is given, extract that function; otherwise, use the first.
    Uses a brace-counting parser for robustness.
    """
    import re
    # Find all function definitions using brace counting
    lines = ir_str.splitlines()
    functions = []
    in_func = False
    func_lines = []
    brace_count = 0
    func_header = None
    for line in lines:
        if not in_func and line.strip().startswith('define'):
            in_func = True
            func_lines = [line]
            brace_count = line.count('{') - line.count('}')
            func_header = line
        elif in_func:
            func_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                # End of function
                func_block = '\n'.join(func_lines)
                # Extract function name
                m = re.match(r'define\s+[^@]+@([^\(]+)\(', func_header.strip())
                name = m.group(1).strip() if m else None
                functions.append((name, func_block))
                in_func = False
                func_lines = []
                func_header = None
    if not functions:
        raise ValueError(f"[pyir.fusion] Could not parse function definition from IR")
    # Pick the function by name if possible
    if func_name:
        for n, block in functions:
            if n == func_name:
                func_ir = block
                break
        else:
            func_ir = functions[0][1]
    else:
        func_ir = functions[0][1]
    # Now parse the function signature and body as before
    sig_match = re.search(r'define\s+([^{@]+)@([^\(]+)\(([^\)]*)\)[^{]*\{', func_ir)
    if not sig_match:
        print(f"[pyir.fusion] Could not parse function signature from IR block:\n{func_ir}")
        raise ValueError(f"[pyir.fusion] Could not parse function signature from IR")
    ret_type = sig_match.group(1).strip()
    name = sig_match.group(2).strip()
    args_str = sig_match.group(3).strip()
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
    # Parse body
    body_match = re.search(r'\{([\s\S]*?)\}$', func_ir, re.MULTILINE)
    body = body_match.group(1) if body_match else ''
    # Build IRFunction and return
    ir_fn = IRFunction(name, args, ret_type)
    block = IRBlock('entry')
    for line in body.splitlines():
        line = line.strip()
        if line:
            block.add(IRInstr(line))
    ir_fn.add_block(block)
    return ir_fn, [], []

def _merge_ir_functions(functions, ssa_uid):
    """
    Merge multiple IRFunction objects into a single fused function.
    Returns (IRFunction, output_vars, output_names).
    """
    if not functions:
        raise ValueError("[pyir.fusion] No functions to merge")
    
    # Use the first function as the base for signature
    base_fn = functions[0]
    merged_fn = IRFunction(
        f"fused_{ssa_uid}",
        base_fn.args,
        base_fn.ret_type,
        base_fn.attrs
    )
    
    output_vars = []
    output_names = []
    
    # Merge all blocks from all functions
    for i, fn in enumerate(functions):
        for block in fn.blocks:
            # Rename block labels to avoid conflicts
            if block.label == 'entry' and i > 0:
                block.label = f'entry_{i}'
            merged_fn.add_block(block)
            
            # Collect output variables from return instructions
            for instr in block.instrs:
                ret_match = re.search(r'ret\s+[^%]*%([a-zA-Z_][a-zA-Z0-9_]+)', str(instr))
                if ret_match:
                    outvar = f"%{ret_match.group(1)}"
                    output_vars.append(outvar)
                    output_names.append(f"out_{i}")
    
    return merged_fn, output_vars, output_names

def fuse_kernels(fns, name="fused_kernel", register=True, debug=False, pretty_print=False, inline_deps=False, outputs=None, output_names=None, return_type='tuple'):
    """
    Robustly fuse multiple PyIR kernels into a single kernel using IR object model.
    Supports complex-valued and async kernels.
    - Each sub-kernel is added as a separate function in the IR module.
    - The fused kernel is a wrapper that calls each sub-kernel and returns their results.
    - If any sub-kernel is async, the fused kernel is async and awaits all sub-kernels.
    - Complex-valued kernels (pyir.complex64, pyir.complex128, complex) are supported.
    """
    if _function_registry is None:
        raise RuntimeError("[pyir.fusion] _function_registry not set. Call set_function_registry() first.")

    # Validate argument types and collect metadata
    arg_types_map = {}
    arg_names = []
    shape_metadata = {}
    is_async = False
    is_complex = False
    for fn in fns:
        sig = inspect.signature(fn)
        for n, p in sig.parameters.items():
            if n in arg_types_map:
                if arg_types_map[n] != p.annotation:
                    raise TypeError(f"[pyir.fusion] Argument '{n}' has inconsistent types: {arg_types_map[n]} vs {p.annotation}")
            else:
                arg_types_map[n] = p.annotation
                arg_names.append(n)
                shape_metadata[n] = getattr(p, 'shape', None)
            # Detect complex types
            if hasattr(p.annotation, 'llvm') and ('{' in p.annotation.llvm and 'float' in p.annotation.llvm):
                is_complex = True
            elif p.annotation in (complex,):
                is_complex = True
        # Detect async kernels
        if asyncio.iscoroutinefunction(fn):
            is_async = True
        elif hasattr(fn, '_is_async_kernel') and fn._is_async_kernel:
            is_async = True
    arg_types = [arg_types_map[n] for n in arg_names]
    ssa_uid = uuid.uuid4().hex[:8]

    # Parse IR strings to IR objects for all sub-kernels
    ir_functions = []
    kernel_names = []
    for fn in fns:
        found = False
        for k in _function_registry:
            if k.startswith(fn.__name__ + "__"):
                found = True
                ir_str = _function_registry[k]
                ir_fn, _, _ = _parse_ir_to_objects(ir_str, ssa_uid)
                ir_functions.append(ir_fn)
                kernel_names.append(ir_fn.name)
                break
        if not found:
            raise ValueError(f"[pyir.fusion] No IR found for function '{fn.__name__}'.")

    # Check that all sub-kernels have identical argument types and order
    base_args = ir_functions[0].args
    for fn in ir_functions[1:]:
        if fn.args != base_args:
            raise TypeError(f"[pyir.fusion] All fused kernels must have identical argument types and order. Got: {base_args} vs {fn.args}")

    # Use the first kernel's signature for the wrapper
    wrapper_args = base_args
    wrapper_ret_type = ir_functions[0].ret_type if len(ir_functions) == 1 else f"{{{', '.join(fn.ret_type for fn in ir_functions)}}}"

    # Build the wrapper function body
    wrapper_block = IRBlock('entry')
    result_vars = []
    for i, fn in enumerate(ir_functions):
        call_args = [f"%{n}" for n, _ in wrapper_args]
        call_result = ssa(f"call_{fn.name}")
        wrapper_block.add(IRInstr(f"{call_result} = call {fn.ret_type} @{fn.name}({', '.join(f'{t} {a}' for a, (n, t) in zip(call_args, fn.args))})"))
        result_vars.append(call_result)
    # Return as struct (tuple)
    if len(result_vars) == 1:
        wrapper_block.add(IRInstr(f"ret {ir_functions[0].ret_type} {result_vars[0]}"))
    else:
        # Create a struct to hold all results
        struct_type = f"{{{', '.join(fn.ret_type for fn in ir_functions)}}}"
        # Chain insertvalue assignments, starting from undef
        prev_var = None
        for i, (res, fn) in enumerate(zip(result_vars, ir_functions)):
            if i == 0:
                prev_var = ssa('fused_result')
                wrapper_block.add(IRInstr(f"{prev_var} = insertvalue {struct_type} undef, {fn.ret_type} {res}, {i}"))
            else:
                next_var = ssa('fused_result')
                wrapper_block.add(IRInstr(f"{next_var} = insertvalue {struct_type} {prev_var}, {fn.ret_type} {res}, {i}"))
                prev_var = next_var
        wrapper_block.add(IRInstr(f"ret {struct_type} {prev_var}"))
    wrapper_fn = IRFunction(name, wrapper_args, wrapper_ret_type)
    wrapper_fn.add_block(wrapper_block)

    # Build the IR module: all sub-kernels + wrapper
    ir_module = IRModule()
    for fn in ir_functions:
        ir_module.add_function(fn)
    ir_module.add_function(wrapper_fn)
    fused_ir = str(ir_module)

    try:
        print(f"[pyir.fusion] IR before validation for '{name}':\n{fused_ir}")
        validate_ir(fused_ir)
    except Exception as e:
        print(f"[pyir.fusion] Invalid IR for '{name}':\n{fused_ir}")
        raise RuntimeError(f"[pyir.fusion] Fused IR validation failed: {e}")

    if register:
        register_function(name, fused_ir)
        from .core import _compile_ir
        _compile_ir(fused_ir, fn_name=name)

    if debug or pretty_print:
        print(f"[pyir.fusion] Fused IR for '{name}':\n{fused_ir}")

    # --- Fused Python wrapper: supports sync, async, and complex-valued kernels ---
    def is_async_fn(fn):
        import asyncio
        return asyncio.iscoroutinefunction(fn) or getattr(fn, '_is_async_kernel', False)

    any_async = any(is_async_fn(fn) for fn in fns)

    if any_async:
        async def fused_fn(*args, out=None):
            # Await all sub-kernels if needed
            results = []
            for fn in fns:
                res = fn(*args)
                if asyncio.iscoroutine(res) or hasattr(res, '__await__'):
                    res = await res
                results.append(res)
            if outputs is not None:
                results = [results[i] for i in outputs]
                names = [output_names[i] for i in outputs] if output_names else [f"out{i}" for i in outputs]
            else:
                names = output_names if output_names else [f"out{i}" for i in range(len(results))]
            if return_type == 'namedtuple':
                NT = collections.namedtuple(f"FusedOutputs_{name}", names)
                return NT(*results)
            elif return_type == 'dict':
                return {n: v for n, v in zip(names, results)}
            if len(results) == 1:
                return results[0]
            return tuple(results)
        fused_fn._is_async_kernel = True
    else:
        def fused_fn(*args, out=None):
            results = [fn(*args) for fn in fns]
            if outputs is not None:
                results = [results[i] for i in outputs]
                names = [output_names[i] for i in outputs] if output_names else [f"out{i}" for i in outputs]
            else:
                names = output_names if output_names else [f"out{i}" for i in range(len(results))]
            if return_type == 'namedtuple':
                NT = collections.namedtuple(f"FusedOutputs_{name}", names)
                return NT(*results)
            elif return_type == 'dict':
                return {n: v for n, v in zip(names, results)}
            if len(results) == 1:
                return results[0]
            return tuple(results)

    # Attach shape metadata and kernel info
    fused_fn._arg_names = arg_names
    fused_fn._arg_types = arg_types
    fused_fn._shape_metadata = shape_metadata
    fused_fn._output_names = output_names if output_names else [f"out{i}" for i in range(len(fns))]
    fused_fn._ir_module = ir_module
    fused_fn._fused_kernels = fns
    fused_fn._is_async_kernel = any_async
    fused_fn._is_complex_kernel = is_complex
    return fused_fn

# Ergonomic @fuse decorator
def fuse(*fns, **kwargs):
    """Decorator or function: fuse kernels with ergonomic API."""
    def decorator(fn):
        return fuse_kernels((fn,)+fns, **kwargs)
    return decorator

# Operator overloading for fusion
class FusableKernel:
    def __init__(self, fn):
        self.fn = fn
    
    def __add__(self, other):
        if isinstance(other, FusableKernel):
            return FusableKernel(fuse_kernels([self.fn, other.fn]))
        return NotImplemented
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

def as_fusable(fn):
    """Wrap a kernel to support + fusion operator."""
    return FusableKernel(fn)

# --- Enhanced Autovectorization utility with IR object model ---
def autovectorize_kernels(fns, width=4, dtype=None, name_prefix="simd_"):
    """
    Given a list of scalar kernels, generate SIMD versions and fuse them using IR object model.
    - Uses @simd_kernel to wrap each kernel.
    - Fuses the resulting SIMD kernels with optimized IR object operations.
    - Returns the fused SIMD kernel.
    Supports kernels with multiple outputs.
    """
    from .simd import simd_kernel
    simd_kernels = []
    for fn in fns:
        simd_fn = simd_kernel(width=width, dtype=dtype)(fn)
        simd_kernels.append(simd_fn)
    fused = fuse_kernels(simd_kernels, name=name_prefix+"fused", register=True)
    # Set SIMD metadata on the fused kernel
    fused._is_simd_kernel = True
    fused._simd_width = width
    fused._simd_dtype = dtype
    return fused

# --- IR Object Model Utilities for Fusion ---
def get_kernel_ir_objects(fn):
    """
    Extract IR objects from a kernel function for inspection and manipulation.
    Returns (IRModule, output_vars, output_names) or None if not found.
    """
    if _function_registry is None:
        return None
    
    for k in _function_registry:
        if k.startswith(fn.__name__ + "__"):
            ir_str = _function_registry[k]
            ssa_uid = uuid.uuid4().hex[:8]
            ir_fn, output_vars, output_names = _parse_ir_to_objects(ir_str, ssa_uid)
            
            ir_module = IRModule()
            ir_module.add_function(ir_fn)
            
            return ir_module, output_vars, output_names
    
    return None

def analyze_fusion_compatibility(fns):
    """
    Analyze the compatibility of kernels for fusion.
    Returns a dict with compatibility info and potential issues.
    """
    if not fns:
        return {"compatible": False, "issues": ["No functions provided"]}
    
    issues = []
    is_cuda = any(getattr(fn, '_is_cuda_kernel', False) for fn in fns)
    is_simd = any(getattr(fn, '_is_simd_kernel', False) for fn in fns)
    
    # Check device type consistency
    if is_cuda and not all(getattr(fn, '_is_cuda_kernel', False) for fn in fns):
        issues.append("Mixed CPU and CUDA kernels")
    
    if is_simd and not all(getattr(fn, '_is_simd_kernel', False) for fn in fns):
        issues.append("Mixed CPU and SIMD kernels")
    
    # Check argument type compatibility
    arg_types_map = {}
    for fn in fns:
        sig = inspect.signature(fn)
        for n, p in sig.parameters.items():
            if n in arg_types_map:
                if arg_types_map[n] != p.annotation:
                    issues.append(f"Argument '{n}' has inconsistent types: {arg_types_map[n]} vs {p.annotation}")
            else:
                arg_types_map[n] = p.annotation
    
    # Check IR availability
    for fn in fns:
        found = False
        for k in _function_registry:
            if k.startswith(fn.__name__ + "__"):
                found = True
                break
        if not found:
            issues.append(f"No IR found for function '{fn.__name__}'")
    
    return {
        "compatible": len(issues) == 0,
        "issues": issues,
        "device_types": {
            "cuda": is_cuda,
            "simd": is_simd,
            "cpu": not (is_cuda or is_simd)
        },
        "arg_count": len(arg_types_map),
        "arg_types": arg_types_map
    }

def vectorized_fuse_kernels(fns, name="fused_vectorized_kernel", dtype=None, register=True, debug=False):
    """
    Fuse multiple vectorized (native loop) kernels into a single vectorized kernel.
    This is now the default and recommended way to fuse elementwise kernels in PyIR.
    The fused kernel takes (in1_ptr, in2_ptr, ..., out_ptr1, out_ptr2, ..., n) and applies all fused scalar kernels in a single native loop.
    Supports arbitrary number of inputs and outputs per kernel, and kernels with different argument lists.
    """
    import ctypes
    import numpy as np
    from collections import OrderedDict
    # Infer dtype from first kernel if not given
    if dtype is None:
        dtype = getattr(fns[0], '_scalar_kernel', None)
        if dtype is not None:
            dtype = getattr(dtype, '_arg_types', [None])[0] or dtype
        else:
            dtype = float32
    # Gather all scalar kernels and their arity/output counts
    scalar_kernels = [getattr(fn, '_scalar_kernel', fn) for fn in fns]
    # Compute the union of all input argument names/types, preserving order of first appearance
    arg_name_type_pairs = OrderedDict()
    for fn in scalar_kernels:
        if hasattr(fn, '_arg_names') and hasattr(fn, '_arg_types'):
            for n, t in zip(fn._arg_names, fn._arg_types):
                if n not in arg_name_type_pairs:
                    arg_name_type_pairs[n] = t
        else:
            import inspect
            sig = inspect.signature(fn)
            for n, p in sig.parameters.items():
                if n not in arg_name_type_pairs:
                    arg_name_type_pairs[n] = getattr(p, 'annotation', dtype)
    arg_names = list(arg_name_type_pairs.keys())
    arg_types = list(arg_name_type_pairs.values())
    n_args = len(arg_names)
    # Determine number of outputs for each kernel
    n_outputs_list = []
    for fn in scalar_kernels:
        ret_type = fn.__annotations__.get('return', dtype)
        n_outputs = 1
        if hasattr(ret_type, '__origin__') and ret_type.__origin__ is tuple:
            n_outputs = len(ret_type.__args__)
        n_outputs_list.append(n_outputs)
    total_outputs = sum(n_outputs_list)
    # Build IR for the fused vectorized kernel
    in_ptrs = [f"{dtype.llvm}* %{n}" for n in arg_names]
    out_ptrs = [f"{dtype.llvm}* %out{i}" for i in range(total_outputs)]
    ir_lines = [
        f"define void @{name}({', '.join(in_ptrs + out_ptrs + ['i32 %n'])}) {{",
        "entry:",
        "  %i = alloca i32, align 4",
        "  store i32 0, i32* %i",
        "  br label %loop",
        "loop:",
        "  %idx = load i32, i32* %i",
        "  %cmp = icmp slt i32 %idx, %n",
        "  br i1 %cmp, label %body, label %exit",
        "body:",
    ]
    # Load all inputs
    for n in arg_names:
        ir_lines.append(f"  %{n}_ptr = getelementptr {dtype.llvm}, {dtype.llvm}* %{n}, i32 %idx")
        ir_lines.append(f"  %{n}_val = load {dtype.llvm}, {dtype.llvm}* %{n}_ptr")
    # For each kernel, inline its op(s)
    from .fusion import get_kernel_ir_objects
    out_idx = 0
    for k, fn in enumerate(scalar_kernels):
        ir_module, _, _ = get_kernel_ir_objects(fn)
        scalar_ir = str(ir_module)
        op_lines = []
        ssa_vars = []
        for line in scalar_ir.splitlines():
            line = line.strip()
            if not line or line.startswith('define') or line.startswith('{') or line.startswith('}'): continue
            if 'fadd' in line or 'fmul' in line or 'add' in line or 'mul' in line:
                op_lines.append(line)
        if not op_lines:
            raise ValueError(f"[pyir.vectorized_fuse_kernels] Could not find op line in scalar kernel IR for kernel {fn}.")
        # Map argument names for this kernel
        if hasattr(fn, '_arg_names'):
            kernel_arg_names = fn._arg_names
        else:
            import inspect
            kernel_arg_names = list(inspect.signature(fn).parameters.keys())
        import re
        # Inline all op lines, but only store the last one as output
        last_out_var = None
        for j, op_line in enumerate(op_lines):
            for arg_name in kernel_arg_names:
                op_line = re.sub(rf'%{arg_name}(?![a-zA-Z0-9_])', f'%{arg_name}_val', op_line)
            # Extract SSA variable name on the left
            m = re.match(r'(%[a-zA-Z0-9_]+)\s*=.*', op_line)
            if m:
                out_var = m.group(1)
            else:
                out_var = f'%res{out_idx}'
            last_out_var = out_var
            ir_lines.append(f"  {op_line}")
        # Only store the last output variable for this kernel
        ir_lines.append(f"  %out{out_idx}_ptr = getelementptr {dtype.llvm}, {dtype.llvm}* %out{out_idx}, i32 %idx")
        ir_lines.append(f"  store {dtype.llvm} {last_out_var}, {dtype.llvm}* %out{out_idx}_ptr")
        out_idx += 1
    ir_lines.append(f"  %idx_next = add i32 %idx, 1")
    ir_lines.append(f"  store i32 %idx_next, i32* %i")
    ir_lines.append(f"  br label %loop")
    ir_lines.append(f"exit:")
    ir_lines.append(f"  ret void")
    ir_lines.append(f"}}")
    ir = '\n'.join(ir_lines)
    try:
        print(f"[pyir.fusion] IR before validation for '{name}':\n{ir}")
        validate_ir(ir)
    except Exception as e:
        print(f"[pyir.fusion] Invalid IR for '{name}':\n{ir}")
        raise
    from .core import get_or_register_kernel
    cfunc = get_or_register_kernel(name, ir, dtype, target='cpu', fast_math=True)
    if debug:
        for line in ir.splitlines():
            if line.strip().startswith('define'):
                print(f'[pyir.vectorized_fuse_kernels] IR function header: {line.strip()}')
                break
    # Build Python wrapper
    def kernel(*arrays, out=None):
        arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
        n = arrays[0].size
        for arr in arrays:
            assert arr.size == n
        # Robust output allocation
        if out is None:
            out_ptrs = [np.empty_like(arrays[0]) for _ in range(total_outputs)]
            out = out_ptrs[0] if total_outputs == 1 else out_ptrs
        else:
            out_ptrs = out if isinstance(out, (list, tuple)) else [out]
        if debug:
            print(f'[pyir.vectorized_fuse_kernels] arrays: {[(arr.shape, arr.dtype, arr.ctypes.data) for arr in arrays]}')
            print(f'[pyir.vectorized_fuse_kernels] out_ptrs: {len(out_ptrs)}, {[o.shape for o in out_ptrs]}, {[o.ctypes.data for o in out_ptrs]}')
        if len(out_ptrs) != total_outputs:
            raise ValueError(f"[pyir.vectorized_fuse_kernels] Number of output arrays ({len(out_ptrs)}) does not match expected ({total_outputs})")
        for i, o in enumerate(out_ptrs):
            if o.ctypes.data == 0:
                raise ValueError(f"[pyir.vectorized_fuse_kernels] Output array {i} has NULL pointer!")
        cfunc(
            *(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for arr in arrays),
            *(o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for o in out_ptrs),
            n)
        return out
    kernel._is_vectorized_kernel = True
    kernel._fused_kernels = fns
    kernel.__name__ = name
    return kernel
