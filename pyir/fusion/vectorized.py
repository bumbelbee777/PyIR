import ctypes
import numpy as np

from ..core.ir import validate_ir
from ..typing import float32

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
    from pyir.core.cache import get_or_register_kernel
    # Build ctypes signature for the fused kernel
    float_ptr = ctypes.POINTER(ctypes.c_float)
    arg_ctypes = [float_ptr for _ in arg_names] + [float_ptr for _ in range(total_outputs)] + [ctypes.c_int]
    cfunc = get_or_register_kernel(name, ir, dtype, target='cpu', fast_math=True, arg_ctypes=arg_ctypes, no_optims=no_optims)
    if debug:
        for line in ir.splitlines():
            if line.strip().startswith('define'):
                print(f'[pyir.vectorized_fuse_kernels] IR function header: {line.strip()}')
                break
    # Build Python wrapper
    def kernel(*arrays, out=None):
        arrays = [np.ascontiguousarray(np.asarray(arr, dtype=np.float32)) for arr in arrays]
        n = arrays[0].size
        for arr in arrays:
            assert arr.size == n
        # Robust output allocation
        if out is None:
            out_ptrs = [np.empty_like(arrays[0]) for _ in range(total_outputs)]
            out = out_ptrs[0] if total_outputs == 1 else out_ptrs
        else:
            out_ptrs = out if isinstance(out, (list, tuple)) else [out]
        out_ptrs = [np.ascontiguousarray(o, dtype=np.float32) for o in out_ptrs]
        if debug:
            print(f'[pyir.vectorized_fuse_kernels] arrays: {[(arr.shape, arr.dtype, type(arr), arr.ctypes.data) for arr in arrays]}')
            print(f'[pyir.vectorized_fuse_kernels] out_ptrs: {len(out_ptrs)}, {[o.shape for o in out_ptrs]}, {[type(o) for o in out_ptrs]}, {[o.ctypes.data for o in out_ptrs]}')
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