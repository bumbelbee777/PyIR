import ctypes
import numpy as np

from ..core.ir import validate_ir
from ..typing import float32, float64, int32, int64, complex64, complex128, void
from .._engine import pyir_debug

def pyir_type_to_llvm_and_ctypes(pyir_type):
    # Accept both class and instance
    t = pyir_type
    if hasattr(t, 'llvm') and hasattr(t, 'ctype'):
        return t.llvm, t.ctype
    if t in ('float32', float32):
        return 'float', ctypes.c_float
    if t in ('float64', float64):
        return 'double', ctypes.c_double
    if t in ('int32', int32):
        return 'i32', ctypes.c_int32
    if t in ('int64', int64):
        return 'i64', ctypes.c_int64
    if t in ('complex64', complex64):
        return '{float, float}', ctypes.c_float * 2
    if t in ('complex128', complex128):
        return '{double, double}', ctypes.c_double * 2
    raise NotImplementedError(f"Type {pyir_type} not supported")

def vectorized_fuse_kernels(fns, name="fused_vectorized_kernel", dtype=None, register=True, debug=False, no_optims=False):
    """
    Fuse multiple vectorized (native loop) kernels into a single vectorized kernel.
    This is now the default and recommended way to fuse elementwise kernels in PyIR.
    The fused kernel takes (in1_ptr, in2_ptr, ..., out_ptr1, out_ptr2, ..., n) and applies all fused scalar kernels in a single native loop.
    Supports arbitrary number of inputs and outputs per kernel, and kernels with different argument lists.
    """
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
                # PATCH: Skip 'out' parameter for void functions since they use output pointers
                ret_type = fn.__annotations__.get('return', dtype)
                if (ret_type is void or (hasattr(ret_type, '__name__') and ret_type.__name__ == 'void')) and n == 'out':
                    continue
                if n not in arg_name_type_pairs:
                    arg_name_type_pairs[n] = t
        else:
            import inspect
            sig = inspect.signature(fn)
            for n, p in sig.parameters.items():
                # PATCH: Skip 'out' parameter for void functions since they use output pointers
                ret_type = fn.__annotations__.get('return', dtype)
                if (ret_type is void or (hasattr(ret_type, '__name__') and ret_type.__name__ == 'void')) and n == 'out':
                    continue
                if n not in arg_name_type_pairs:
                    arg_name_type_pairs[n] = getattr(p, 'annotation', dtype)
    arg_names = list(arg_name_type_pairs.keys())
    arg_types = list(arg_name_type_pairs.values())
    n_args = len(arg_names)
    # Determine number of outputs for each kernel
    n_outputs_list = []
    output_types = []
    for fn in scalar_kernels:
        ret_type = fn.__annotations__.get('return', dtype)
        if pyir_debug:
            print(f"[DEBUG] Function {getattr(fn, '__name__', 'unknown')} return type: {ret_type}")
        
        # PATCH: For void functions, look at the original function signature to find output pointer
        if (ret_type is void or (hasattr(ret_type, '__name__') and ret_type.__name__ == 'void')):
            # Look at the original function signature to find the output pointer
            if hasattr(fn, '_arg_names') and hasattr(fn, '_arg_types'):
                for n, t in zip(fn._arg_names, fn._arg_types):
                    if n == 'out' and hasattr(t, 'base_type'):
                        output_type = t.base_type
                        if pyir_debug:
                            print(f"[DEBUG] Found output pointer type: {output_type}")
                        output_types.append(output_type)
                        break
                else:
                    # Fallback: use complex64 as output type
                    output_type = complex64
                    if pyir_debug:
                        print(f"[DEBUG] Using fallback output type: {output_type}")
                    output_types.append(output_type)
            else:
                # Fallback: use complex64 as output type
                output_type = complex64
                if pyir_debug:
                    print(f"[DEBUG] Using fallback output type: {output_type}")
                output_types.append(output_type)
            n_outputs = 1
        else:
            n_outputs = 1
            if hasattr(ret_type, '__origin__') and ret_type.__origin__ is tuple:
                n_outputs = len(ret_type.__args__)
                output_types.extend(ret_type.__args__)
            else:
                output_types.append(ret_type)
        n_outputs_list.append(n_outputs)
    
    print(f"[DEBUG] Final output_types: {output_types}")
    
    total_outputs = sum(n_outputs_list)
    # Build IR for the fused vectorized kernel
    in_ptrs = []
    for n, t in zip(arg_names, arg_types):
        llvm_type, _ = pyir_type_to_llvm_and_ctypes(t)
        in_ptrs.append(f"{llvm_type}* %{n}")
    out_ptrs = []
    for i, t in enumerate(output_types):
        llvm_type, _ = pyir_type_to_llvm_and_ctypes(t)
        # PATCH: Always use single pointer for outputs, and never void*
        if llvm_type == 'void':
            # If we somehow got void, use i8* as a fallback
            llvm_type = 'i8'
        out_ptrs.append(f"{llvm_type}* %out{i}")
        if pyir_debug:
            print(f"[DEBUG] Output {i}: type={t}, llvm_type={llvm_type}")
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
    for n, t in zip(arg_names, arg_types):
        llvm_type, _ = pyir_type_to_llvm_and_ctypes(t)
        ir_lines.append(f"  %{n}_ptr = getelementptr {llvm_type}, {llvm_type}* %{n}, i32 %idx")
        # PATCH: For struct types, load the struct value after getelementptr
        if llvm_type.startswith('{'):
            ir_lines.append(f"  %{n}_val = load {llvm_type}, {llvm_type}* %{n}_ptr")
        else:
            ir_lines.append(f"  %{n}_val = load {llvm_type}, {llvm_type}* %{n}_ptr")
    # For each kernel, inline its op(s)
    from .core import get_kernel_ir_objects
    out_idx = 0
    for k, fn in enumerate(scalar_kernels):
        result = get_kernel_ir_objects(fn)
        if result is None:
            raise ValueError(f"[pyir.vectorized_kernel] No IR found for kernel {getattr(fn, '__name__', repr(fn))} (type: {type(fn)}). Is it registered? Function registry: {getattr(fn, '_function_registry', 'N/A')}")
        ir_module, _, _ = result
        scalar_ir = str(ir_module)
        op_lines = []
        for line in scalar_ir.splitlines():
            line = line.strip()
            if not line or line.startswith('define') or line.startswith('{') or line.startswith('}'): continue
            if 'fadd' in line or 'fmul' in line or 'add' in line or 'mul' in line or 'sub' in line or 'div' in line:
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
        # 1. Find all SSA variable assignments in op_lines
        ssa_vars = set()
        for op_line in op_lines:
            m = re.match(r'(%[a-zA-Z0-9_]+)\s*=.*', op_line)
            if m:
                ssa_vars.add(m.group(1))
        # 2. Build a mapping to unique names per kernel
        var_rename_map = {orig_var: f"{orig_var}_{k}" for orig_var in ssa_vars}
        # 3. Replace all occurrences (LHS and RHS) in all op_lines
        renamed_op_lines = []
        for op_line in op_lines:
            # Replace argument names
            for arg_name in kernel_arg_names:
                op_line = re.sub(rf'%{arg_name}(?![a-zA-Z0-9_])', f'%{arg_name}_val', op_line)
            # Replace all SSA variable names (LHS and RHS)
            for orig_var, new_var in var_rename_map.items():
                op_line = re.sub(rf'(?<![a-zA-Z0-9_]){re.escape(orig_var)}(?![a-zA-Z0-9_])', new_var, op_line)
            renamed_op_lines.append(op_line)
        # Detect op and output type
        op_type = None
        for op_line in renamed_op_lines:
            if 'fadd' in op_line: op_type = 'fadd'
            elif 'fmul' in op_line: op_type = 'fmul'
            elif 'add' in op_line: op_type = 'add'
            elif 'mul' in op_line: op_type = 'mul'
            elif 'sub' in op_line: op_type = 'sub'
            elif 'div' in op_line: op_type = 'div'
        out_type = output_types[out_idx]
        if _is_struct_type(out_type):
            # Find input argument names (skip "out")
            input_arg_names = [n for n in kernel_arg_names if n != "out"]
            assert len(input_arg_names) == 2, f"Expected 2 input args, got {input_arg_names}"
            lhs_var = f"%{input_arg_names[0]}_val"
            rhs_var = f"%{input_arg_names[1]}_val"
            struct_ir, struct_result = emit_struct_op_ir(lhs_var, rhs_var, f"structop_{out_idx}_{k}", out_type, op_type)
            ir_lines.extend(struct_ir)
            last_out_var = struct_result
        else:
            # Inline all renamed op lines, only store the last one as output
            last_out_var = None
            for op_line in renamed_op_lines:
                m = re.match(r'(%[a-zA-Z0-9_]+)\s*=.*', op_line)
                if m:
                    out_var = m.group(1)
                else:
                    out_var = f'%res{out_idx}'
                last_out_var = out_var
                ir_lines.append(f"  {op_line}")
        # Store result to output
        llvm_type, _ = pyir_type_to_llvm_and_ctypes(output_types[out_idx])
        ir_lines.append(f"  %out{out_idx}_ptr = getelementptr {llvm_type}, {llvm_type}* %out{out_idx}, i32 %idx")
        ir_lines.append(f"  store {llvm_type} {last_out_var}, {llvm_type}* %out{out_idx}_ptr")
        out_idx += 1
    ir_lines.append(f"  %idx_next = add i32 %idx, 1")
    ir_lines.append(f"  store i32 %idx_next, i32* %i")
    ir_lines.append(f"  br label %loop")
    ir_lines.append(f"exit:")
    ir_lines.append(f"  ret void")
    ir_lines.append(f"}}")
    ir = '\n'.join(ir_lines)
    try:
        if pyir_debug:
            print(f"[pyir.fusion] IR before validation for '{name}':\n{ir}")
        validate_ir(ir)
    except Exception as e:
        print(f"[pyir.fusion] Invalid IR for '{name}':\n{ir}")
        raise
    from pyir.core.cache import get_or_register_kernel
    # Build ctypes signature for the fused kernel
    arg_ctypes = []
    for t in arg_types:
        _, ctype = pyir_type_to_llvm_and_ctypes(t)
        arg_ctypes.append(ctypes.POINTER(ctype))
    for t in output_types:
        _, ctype = pyir_type_to_llvm_and_ctypes(t)
        arg_ctypes.append(ctypes.POINTER(ctype))
    arg_ctypes.append(ctypes.c_int)
    cfunc = get_or_register_kernel(name, ir, dtype, target='cpu', fast_math=True, arg_ctypes=arg_ctypes, no_optims=no_optims)
    if debug:
        for line in ir.splitlines():
            if line.strip().startswith('define'):
                print(f'[pyir.vectorized_fuse_kernels] IR function header: {line.strip()}')
                break
    # Build Python wrapper
    def kernel(*arrays, out=None):
        # Map types for each input
        arrays = [np.ascontiguousarray(np.asarray(arr, dtype=np.result_type(t.ctype))) for arr, t in zip(arrays, arg_types)]
        n = arrays[0].size
        for arr in arrays:
            assert arr.size == n
        # Robust output allocation
        if out is None:
            out_ptrs = [np.empty_like(arrays[0], dtype=np.result_type(t.ctype)) for t in output_types]
            out = out_ptrs[0] if total_outputs == 1 else out_ptrs
        else:
            out_ptrs = out if isinstance(out, (list, tuple)) else [out]
        out_ptrs = [np.ascontiguousarray(o, dtype=np.result_type(t.ctype)) for o, t in zip(out_ptrs, output_types)]
        if debug:
            print(f'[pyir.vectorized_fuse_kernels] arrays: {[(arr.shape, arr.dtype, type(arr), arr.ctypes.data) for arr in arrays]}')
            print(f'[pyir.vectorized_fuse_kernels] out_ptrs: {len(out_ptrs)}, {[o.shape for o in out_ptrs]}, {[type(o) for o in out_ptrs]}, {[o.ctypes.data for o in out_ptrs]}')
        if len(out_ptrs) != total_outputs:
            raise ValueError(f"[pyir.vectorized_fuse_kernels] Number of output arrays ({len(out_ptrs)}) does not match expected ({total_outputs})")
        for i, o in enumerate(out_ptrs):
            if o.ctypes.data == 0:
                raise ValueError(f"[pyir.vectorized_fuse_kernels] Output array {i} has NULL pointer!")
        cfunc(
            *(arr.ctypes.data_as(ctypes.POINTER(pyir_type_to_llvm_and_ctypes(t)[1])) for arr, t in zip(arrays, arg_types)),
            *(o.ctypes.data_as(ctypes.POINTER(pyir_type_to_llvm_and_ctypes(t)[1])) for o, t in zip(out_ptrs, output_types)),
            n)
        return out
    kernel._is_vectorized_kernel = True
    kernel._fused_kernels = fns
    kernel.__name__ = name
    return kernel

def _is_struct_type(t):
    return hasattr(t, 'llvm') and t.llvm.startswith('{')

def _struct_field_types(t):
    return getattr(t, 'field_types', None)

def emit_struct_op_ir(lhs_var, rhs_var, out_var_base, t, op, indent='  '):
    """
    Recursively emit IR for struct types (including complex), applying op to each field.
    Returns (ir_lines, result_var)
    """
    ir_lines = []
    if isinstance(t, type):
        t = t()
    if not _is_struct_type(t):
        # Scalar: just emit op
        result_var = f"%{out_var_base}"
        ir_lines.append(f"{indent}{result_var} = {op} {t.llvm} {lhs_var}, {rhs_var}")
        return ir_lines, result_var
    # Struct: recursively extract fields, operate, and reassemble
    field_types = _struct_field_types(t)
    if field_types is None:
        raise TypeError(f"[pyir.vectorized] Type {t} (repr: {repr(t)}) is marked as struct but has no field_types attribute. This likely means a type is not a valid StructType or ComplexType. Got: {t} (type: {type(t)})")
    prev_var = 'undef'
    for i, ft in enumerate(field_types):
        lhs_f = f"%{out_var_base}_lhs_f{i}"
        rhs_f = f"%{out_var_base}_rhs_f{i}"
        res_f = f"%{out_var_base}_f{i}"
        # extractvalue
        ir_lines.append(f"{indent}{lhs_f} = extractvalue {t.llvm} {lhs_var}, {i}")
        ir_lines.append(f"{indent}{rhs_f} = extractvalue {t.llvm} {rhs_var}, {i}")
        # recurse
        sub_ir, sub_result = emit_struct_op_ir(lhs_f, rhs_f, f"{out_var_base}_f{i}_op", ft, op, indent)
        ir_lines.extend(sub_ir)
        # insertvalue
        if i == 0:
            prev_var = f"%{out_var_base}_struct"
            ir_lines.append(f"{indent}{prev_var} = insertvalue {t.llvm} undef, {ft.llvm} {sub_result}, {i}")
        else:
            next_var = f"%{out_var_base}_struct{i}"
            ir_lines.append(f"{indent}{next_var} = insertvalue {t.llvm} {prev_var}, {ft.llvm} {sub_result}, {i}")
            prev_var = next_var
    return ir_lines, prev_var
