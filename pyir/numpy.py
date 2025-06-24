import ctypes
import inspect
import functools
import concurrent.futures
import asyncio
import os
try:
    import numpy as np
except ImportError:
    np = None
from .typing import int8, int16, int32, int64, float16, float32, float64, complex64, complex128, ptr
from .core import IRInstr, IRBlock, IRFunction, IRModule, function, register_function, validate_ir, sandboxed_jit
from .policy import ExecutionPolicy, validate_policy, SandboxedPolicy

def from_numpy_dtype(dtype):
    """Map a NumPy dtype to a PyIR type."""
    if np is None:
        raise ImportError("NumPy is not available.")
    dt = np.dtype(dtype)
    if dt == np.int8:
        return int8
    if dt == np.int16:
        return int16
    if dt == np.int32:
        return int32
    if dt == np.int64:
        return int64
    if dt == np.float16:
        return float16
    if dt == np.float32:
        return float32
    if dt == np.float64:
        return float64
    if dt == np.complex64:
        return complex64
    if dt == np.complex128:
        return complex128
    raise TypeError(f"[pyir.from_numpy_dtype] Unsupported dtype: {dtype}")

def as_ctypes(arr):
    """Get a ctypes pointer from a NumPy array."""
    if np is None:
        raise ImportError("NumPy is not available.")
    return arr.ctypes.data_as(ctypes.POINTER(from_numpy_dtype(arr.dtype).ctype))

def as_pointer(arr):
    """Get a raw pointer value (int) from a NumPy array."""
    if np is None:
        raise ImportError("NumPy is not available.")
    return arr.ctypes.data

def ptr_from_numpy(arr):
    """Get the PyIR pointer type for a NumPy array."""
    return ptr(from_numpy_dtype(arr.dtype))

def numpy_kernel(pyir_func, debug=False, policy="vectorized", num_threads=None):
    """
    Wrap a @pyir.function as a NumPy-compatible kernel with execution policy support.
    Supported policies: vectorized (default), serial, parallel, async, sandboxed, or SandboxedPolicy(backend).
    For ASYNC policy, returns an asyncio.Future. You can await it or add callbacks.
    """
    policy = validate_policy(policy)
    if isinstance(policy, SandboxedPolicy):
        backend = policy.backend
        if backend == ExecutionPolicy.SERIAL:
            # Sandbox each elementwise call (slow, secure)
            def sandboxed_serial_kernel(*arrays, out=None):
                import numpy as np
                arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
                n = arrays[0].size
                for arr in arrays:
                    assert arr.size == n
                if out is None:
                    out = np.empty_like(arrays[0])
                from .fusion import get_kernel_ir_objects
                ir_module, fn_name, _ = get_kernel_ir_objects(pyir_func)
                ir_code = str(ir_module)
                for i in range(n):
                    args = [arr[i] for arr in arrays]
                    out[i] = sandboxed_jit(ir_code, fn_name, args=args, ret_ctype=None)
                return out
            return sandboxed_serial_kernel
        elif backend == ExecutionPolicy.VECTORIZED:
            # Sandbox the entire vectorized kernel in a subprocess (fast, secure)
            def sandboxed_vectorized_kernel(*arrays, out=None):
                import numpy as np
                arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
                n = arrays[0].size
                for arr in arrays:
                    assert arr.size == n
                if out is None:
                    out = np.empty_like(arrays[0])
                from .fusion import get_kernel_ir_objects
                ir_module, fn_name, _ = get_kernel_ir_objects(pyir_func)
                ir_code = str(ir_module)
                if isinstance(out, (list, tuple)):
                    raise NotImplementedError("Sandboxed vectorized execution only supports single-output kernels for now.")
                result = sandboxed_jit(ir_code, fn_name, args=[arr for arr in arrays], ret_ctype=None)
                out[...] = result
                return out
            return sandboxed_vectorized_kernel
        else:
            raise NotImplementedError(f"Sandboxed execution with backend {backend} is not supported.")
    if policy == ExecutionPolicy.VECTORIZED:
        return vectorized_kernel(pyir_func, debug=debug)
    elif policy == ExecutionPolicy.SERIAL:
        # Fallback: Python loop (slow, for reference)
        def serial_kernel(*arrays, out=None):
            import numpy as np
            arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
            n = arrays[0].size
            for arr in arrays:
                assert arr.size == n
            if out is None:
                out = np.empty_like(arrays[0])
            for i in range(n):
                args = [arr[i] for arr in arrays]
                out[i] = pyir_func(*args)
            return out
        return serial_kernel
    elif policy == ExecutionPolicy.PARALLEL:
        def parallel_kernel(*arrays, out=None):
            import numpy as np
            arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
            n = arrays[0].size
            for arr in arrays:
                assert arr.size == n
            if out is None:
                out = np.empty_like(arrays[0])
            if isinstance(out, (list, tuple)):
                raise NotImplementedError("Parallel execution only supports single-output kernels for now.")
            vec_kernel = vectorized_kernel(pyir_func, debug=debug)
            num_workers = num_threads or min(32, (os.cpu_count() or 1))
            chunk_size = (n + num_workers - 1) // num_workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i in range(0, n, chunk_size):
                    def run_chunk(start=i, end=min(i+chunk_size, n)):
                        vec_kernel(*(arr[start:end] for arr in arrays), out=out[start:end])
                    futures.append(executor.submit(run_chunk))
                concurrent.futures.wait(futures)
            return out
        return parallel_kernel
    elif policy == ExecutionPolicy.ASYNC:
        def async_kernel(*arrays, out=None, loop=None):
            import numpy as np
            arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
            n = arrays[0].size
            for arr in arrays:
                assert arr.size == n
            if out is None:
                out = np.empty_like(arrays[0])
            if isinstance(out, (list, tuple)):
                raise NotImplementedError("Async execution only supports single-output kernels for now.")
            vec_kernel = vectorized_kernel(pyir_func, debug=debug)
            loop = loop or asyncio.get_event_loop()
            future = loop.create_future()
            def run():
                try:
                    result = vec_kernel(*arrays, out=out)
                    loop.call_soon_threadsafe(future.set_result, result)
                except Exception as e:
                    loop.call_soon_threadsafe(future.set_exception, e)
            asyncio.get_running_loop().run_in_executor(None, run)
            return future
        return async_kernel
    else:
        raise NotImplementedError(f"Execution policy {policy} not supported in numpy_kernel.")

def numpy_reduction(pyir_func):
    """
    Decorator: Wrap a @pyir.function as a NumPy-compatible reduction kernel.
    Usage:
        @pyir.numpy_reduction
        @pyir.function
        def mysum(a: pyir.float32, b: pyir.float32) -> pyir.float32:
            return pyir.inline('result = fadd float %a, %b', sugar=True)
        mysum(np_array)  # returns np.float32
    """
    if np is None:
        raise ImportError("NumPy is not available.")
    sig = inspect.signature(pyir_func)
    arg_types = [param.annotation for param in sig.parameters.values()]
    ret_type = sig.return_annotation
    @functools.wraps(pyir_func)
    def wrapper(arr, axis=None, out=None):
        arr = np.asarray(arr)
        if axis is None:
            arr = arr.ravel()
            result = arr[0]
            for i in range(1, arr.size):
                result = pyir_func(result, arr[i])
            return result
        else:
            return np.apply_along_axis(lambda x: wrapper(x, axis=None), axis, arr)
    return wrapper

# --- Native vectorized kernel generator ---
def vectorized_kernel(scalar_kernel, name=None, dtype=float32, debug=False, policy="vectorized"):
    """
    Generate a native vectorized kernel from a scalar PyIR kernel, with execution policy support.
    """
    policy = validate_policy(policy)
    if policy != ExecutionPolicy.VECTORIZED:
        raise NotImplementedError(f"Only vectorized policy is supported in vectorized_kernel (got {policy}).")
    import ctypes
    import numpy as np
    # Infer types and names
    if name is None:
        name = f"vectorized_{scalar_kernel.__name__}"
    arg_types = getattr(scalar_kernel, '_arg_types', None)
    if arg_types is None:
        import inspect
        sig = inspect.signature(scalar_kernel)
        arg_types = [param.annotation for param in sig.parameters.values()]
    n_args = len(arg_types)
    # Determine number of outputs (handle tuple return)
    ret_type = scalar_kernel.__annotations__.get('return', dtype)
    n_outputs = 1
    is_tuple_output = False
    if hasattr(ret_type, '__origin__') and ret_type.__origin__ is tuple:
        n_outputs = len(ret_type.__args__)
        is_tuple_output = True
    # Build IR for the vectorized kernel
    in_ptrs = [f"{dtype.llvm}* %a{i}" for i in range(n_args)]
    out_ptrs = [f"{dtype.llvm}* %out{i}" for i in range(n_outputs)]
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
    for j in range(n_args):
        ir_lines.append(f"  %a{j}_ptr = getelementptr {dtype.llvm}, {dtype.llvm}* %a{j}, i32 %idx")
        ir_lines.append(f"  %a{j}_val = load {dtype.llvm}, {dtype.llvm}* %a{j}_ptr")
    # Inline the scalar kernel's IR, replacing %a, %b, ... with %a0_val, %a1_val, ...
    from .fusion import get_kernel_ir_objects
    ir_module, _, _ = get_kernel_ir_objects(scalar_kernel)
    scalar_ir = str(ir_module)
    op_lines = []
    output_vars = []
    last_out_var = None
    for line in scalar_ir.splitlines():
        line = line.strip()
        if not line or line.startswith('define') or line.startswith('{') or line.startswith('}'): continue
        if 'fadd' in line or 'fmul' in line or 'add' in line or 'mul' in line:
            op_lines.append(line)
    if not op_lines:
        raise ValueError("[pyir.vectorized_kernel] Could not find op line in scalar kernel IR.")
    # Map argument names: assume scalar kernel uses %a, %b, %c, ...
    arg_names = [chr(97 + j) for j in range(n_args)]  # 'a', 'b', ...
    ssa_vars = []
    import re
    for k, op_line in enumerate(op_lines):
        for j, arg_name in enumerate(arg_names):
            op_line = re.sub(rf'%{arg_name}(?![a-zA-Z0-9_])', f'%a{j}_val', op_line)
            op_line = re.sub(rf'%a{j}(?![a-zA-Z0-9_])', f'%a{j}_val', op_line)
        # Extract SSA variable name on the left
        m = re.match(r'(%[a-zA-Z0-9_]+)\s*=.*', op_line)
        if m:
            out_var = m.group(1)
            last_out_var = out_var
        else:
            out_var = f'%res{k}' if n_outputs > 1 else '%res'
            last_out_var = out_var
        ssa_vars.append(out_var)
        ir_lines.append(f"  {op_line}")
    # Store all outputs (for now, only support single output)
    for k in range(n_outputs):
        out_var = last_out_var if n_outputs == 1 else (ssa_vars[k] if k < len(ssa_vars) else (f'%res{k}' if n_outputs > 1 else '%res'))
        ir_lines.append(f"  %out{k}_ptr = getelementptr {dtype.llvm}, {dtype.llvm}* %out{k}, i32 %idx")
        ir_lines.append(f"  store {dtype.llvm} {out_var}, {dtype.llvm}* %out{k}_ptr")
    ir_lines.append(f"  %idx_next = add i32 %idx, 1")
    ir_lines.append(f"  store i32 %idx_next, i32* %i")
    ir_lines.append(f"  br label %loop")
    ir_lines.append(f"exit:")
    ir_lines.append(f"  ret void")
    ir_lines.append(f"}}")
    ir = '\n'.join(ir_lines)
    if debug:
        print(f'[pyir.vectorized_kernel] Generated IR for {name}:\n{ir}')
    validate_ir(ir)
    # Use deduplication and caching
    from .core import get_or_register_kernel
    cfunc = get_or_register_kernel(name, ir, dtype, target='cpu', fast_math=True)
    # Debug: print all registered function names
    from .core import _function_registry
    if debug:
        print(f'[pyir.vectorized_kernel] Registered functions: {list(_function_registry.keys())}')
    # Build Python wrapper
    def kernel(*arrays, out=None):
        arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
        n = arrays[0].size
        for arr in arrays:
            assert arr.size == n
        # Robust output allocation
        if out is None:
            out_ptrs = [np.empty_like(arrays[0]) for _ in range(n_outputs)]
            out = out_ptrs[0] if n_outputs == 1 else out_ptrs
        else:
            out_ptrs = out if isinstance(out, (list, tuple)) else [out]
        # Debug print
        if debug:
            print(f'[pyir.vectorized_kernel] arrays: {[(arr.shape, arr.dtype, arr.ctypes.data) for arr in arrays]}')
            print(f'[pyir.vectorized_kernel] out_ptrs: {len(out_ptrs)}, {[o.shape for o in out_ptrs]}, {[o.ctypes.data for o in out_ptrs]}')
        import ctypes
        cfunc_argtypes = [ctypes.POINTER(ctypes.c_float)] * (len(arrays) + len(out_ptrs)) + [ctypes.c_int32]
        cfunc.restype = None
        cfunc.argtypes = cfunc_argtypes
        cfunc(
            *(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for arr in arrays),
            *(o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for o in out_ptrs),
            n)
        if debug:
            print(f'[pyir.vectorized_kernel] output after execution: {out}')
        return out
    kernel._is_vectorized_kernel = True
    kernel._scalar_kernel = scalar_kernel
    kernel.__name__ = name
    return kernel

# --- Async orchestration helper ---
def async_gather(*futures, loop=None, return_exceptions=False):
    """
    Orchestrate multiple async kernel launches. Equivalent to asyncio.gather.
    Usage: results = await pyir.async_gather(fut1, fut2, ...)
    """
    loop = loop or asyncio.get_event_loop()
    return asyncio.gather(*futures, loop=loop, return_exceptions=return_exceptions)
