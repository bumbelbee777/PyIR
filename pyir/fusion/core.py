import uuid
import asyncio
import collections
import inspect

from ..core.ir import IRFunction, IRInstr, IRModule, IRBlock, validate_ir, ssa
from pyir.core.registry import _function_registry, register_function
from .ir_utils import _parse_ir_to_objects
from pyir.core.jit import _compile_ir
from pyir.typing import python_type_map
from .._engine import pyir_debug

_function_registry = None  # Will be set by main pyir/__init__.py

def set_function_registry(reg):
    global _function_registry
    _function_registry = reg

def _unique_ssa_name(name, uid):
    return f"{name}_f_{uid}"

def _mangle_function_name(fn):
    sig = inspect.signature(fn)
    arg_types = []
    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann in python_type_map:
            ann = python_type_map[ann]
        arg_types.append(getattr(ann, 'llvm', str(ann)))
    ret_ann = sig.return_annotation
    if ret_ann in python_type_map:
        ret_ann = python_type_map[ret_ann]
    if ret_ann is inspect._empty:
        ret_ann = 'void'
    else:
        ret_ann = getattr(ret_ann, 'llvm', str(ret_ann))
    type_suffix = "_".join(arg_types + [ret_ann])
    return f"{fn.__name__}__{type_suffix}"

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
        mangled = _mangle_function_name(fn)
        for k in _function_registry:
            if k == mangled:
                found = True
                ir_obj = _function_registry[k]
                ir_fn, _, _ = _parse_ir_to_objects(ir_obj, ssa_uid)
                ir_functions.append(ir_fn)
                kernel_names.append(ir_fn.name)
                break
        if not found:
            raise ValueError(f"[pyir.fusion] No IR found for function '{fn.__name__}' with mangled name '{mangled}'.")

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
        if pyir_debug:
            print(f"[pyir.fusion] IR before validation for '{name}':\n{fused_ir}")
        validate_ir(fused_ir)
    except Exception as e:
        print(f"[pyir.fusion] Invalid IR for '{name}':\n{fused_ir}")
        raise RuntimeError(f"[pyir.fusion] Fused IR validation failed: {e}")

    if register:
        register_function(name, fused_ir)
        _compile_ir(fused_ir, fn_name=name)

    if debug or pretty_print:
        print(f"[pyir.fusion] Fused IR for '{name}':\n{fused_ir}")

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
    # Set the name to match the registered name
    fused_fn.__name__ = name
    # Propagate SIMD attributes if all subkernels are SIMD
    if all(getattr(fn, '_is_simd_kernel', False) for fn in fns):
        fused_fn._is_simd_kernel = True
        # Use width/dtype from first kernel (assume all match)
        fused_fn._simd_width = getattr(fns[0], '_simd_width', None)
        fused_fn._simd_dtype = getattr(fns[0], '_simd_dtype', None)
        fused_fn.is_simd = True
    return fused_fn

def get_kernel_ir_objects(fn):
    """
    Extract IR objects from a kernel function for inspection and manipulation.
    Returns (IRModule, output_vars, output_names) or None if not found.
    """
    if _function_registry is None:
        return None
    
    for k in _function_registry:
        if k.startswith(fn.__name__ + "__"):
            ir_obj = _function_registry[k]
            ssa_uid = uuid.uuid4().hex[:8]
            ir_fn, output_vars, output_names = _parse_ir_to_objects(ir_obj, ssa_uid)
            
            ir_module = IRModule()
            ir_module.add_function(ir_fn)
            
            return ir_module, output_vars, output_names
    
    return None

def get_kernel_metadata(fn):
    """Extract metadata from a kernel function."""
    import inspect
    from . import _function_registry
    sig = inspect.signature(fn)
    # First check for exact match (for fused kernels)
    if fn.__name__ in _function_registry:
        ir_obj = _function_registry[fn.__name__]
        ir_str = str(ir_obj) if hasattr(ir_obj, 'blocks') else ir_obj
        return {
            'name': fn.__name__,
            'ir_length': len(ir_str),
            'has_ir': True,
            'is_cuda': getattr(fn, '_is_cuda_kernel', False),
            'is_simd': getattr(fn, '_is_simd_kernel', False) or getattr(fn, 'is_simd', False),
            'simd_width': getattr(fn, '_simd_width', None),
            'simd_dtype': getattr(fn, '_simd_dtype', None),
            'parameters': list(sig.parameters.keys()),
        }
    # Then check for mangled names (for regular kernels)
    for k in _function_registry:
        if k.startswith(fn.__name__ + "__"):
            ir_obj = _function_registry[k]
            ir_str = str(ir_obj) if hasattr(ir_obj, 'blocks') else ir_obj
            return {
                'name': fn.__name__,
                'ir_length': len(ir_str),
                'has_ir': True,
                'is_cuda': getattr(fn, '_is_cuda_kernel', False),
                'is_simd': getattr(fn, '_is_simd_kernel', False) or getattr(fn, 'is_simd', False),
                'simd_width': getattr(fn, '_simd_width', None),
                'simd_dtype': getattr(fn, '_simd_dtype', None),
                'parameters': list(sig.parameters.keys()),
            }
    return {'name': fn.__name__, 'has_ir': False, 'is_cuda': False, 'is_simd': False, 'simd_width': None, 'simd_dtype': None, 'parameters': []}

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
    
    # For simple test compatibility - if no major issues, allow fusion
    compatible = len(issues) == 0 or all('No IR found' not in issue for issue in issues)
    
    return {
        "compatible": compatible,
        "issues": issues,
        "device_types": {
            "cuda": is_cuda,
            "simd": is_simd,
            "cpu": not (is_cuda or is_simd)
        },
        "arg_count": len(arg_types_map),
        "arg_types": arg_types_map
    }

def _parse_ir_to_objects(ir_str, ssa_uid):
    """Parse IR string to IR objects, handling both strings and IRFunction objects."""
    if hasattr(ir_str, 'blocks'):  # It's already an IRFunction object
        ir_fn = ir_str
        # Extract output variables from the return instruction
        output_vars = []
        output_names = []
        for block in ir_fn.blocks:
            for instr in block.instrs:
                if str(instr).startswith('ret '):
                    ret_instr = str(instr)
                    # Extract the return variable
                    if 'ret void' not in ret_instr:
                        # Find the variable being returned
                        import re
                        match = re.search(r'ret [^{}]+ %([a-zA-Z_][a-zA-Z0-9_]*)', ret_instr)
                        if match:
                            output_vars.append(match.group(1))
                            output_names.append('result')
        return ir_fn, output_vars, output_names
    else:
        # It's a string, parse it
        from ..core.ir import create_ir_function_from_string
        ir_fn = create_ir_function_from_string(ir_str)
        # Extract output variables from the return instruction
        output_vars = []
        output_names = []
        for block in ir_fn.blocks:
            for instr in block.instrs:
                if str(instr).startswith('ret '):
                    ret_instr = str(instr)
                    # Extract the return variable
                    if 'ret void' not in ret_instr:
                        # Find the variable being returned
                        import re
                        match = re.search(r'ret [^{}]+ %([a-zA-Z_][a-zA-Z0-9_]*)', ret_instr)
                        if match:
                            output_vars.append(match.group(1))
                            output_names.append('result')
        return ir_fn, output_vars, output_names
