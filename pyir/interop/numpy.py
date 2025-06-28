import ctypes
import inspect
import functools
import concurrent.futures
import asyncio
import os
import threading
import numpy as np
from typing import List, Optional, Union, Tuple
try:
    import numpy as np
except ImportError:
    np = None
from ..typing import int8, int16, int32, int64, float16, float32, float64, complex64, complex128, ptr
from ..core import IRInstr, IRBlock, IRFunction, IRModule, function, register_function, validate_ir, pyir_debug
from ..security.policy import ExecutionPolicy, validate_policy, SandboxedPolicy
from ..security.sandbox import sandboxed_jit
from ..fusion import get_kernel_ir_objects

# Atomic counters for parallel operations
_parallel_counter = 0
_parallel_counter_lock = threading.Lock()

# Parallel execution settings
_parallel_enabled = True
_max_parallel_workers = min(16, os.cpu_count() or 4)
_parallel_executor = None
_parallel_executor_lock = threading.Lock()

def get_parallel_executor():
    """Get or create the global parallel executor."""
    global _parallel_executor
    with _parallel_executor_lock:
        if _parallel_executor is None:
            _parallel_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_max_parallel_workers)
        return _parallel_executor

def set_parallel_execution(enabled: bool = True, max_workers: int = None):
    """Enable/disable parallel execution and set max workers."""
    global _parallel_enabled, _max_parallel_workers
    _parallel_enabled = enabled
    if max_workers is not None:
        _max_parallel_workers = max_workers
        # Recreate executor with new worker count
        global _parallel_executor
        with _parallel_executor_lock:
            if _parallel_executor is not None:
                _parallel_executor.shutdown(wait=False)
            _parallel_executor = None

def atomic_parallel_counter():
    """Get atomic counter for tracking parallel operations."""
    global _parallel_counter
    with _parallel_counter_lock:
        _parallel_counter += 1
        return _parallel_counter

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

def numpy_kernel(pyir_func, debug=False, policy="vectorized", num_threads=None, no_optims=False):
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
                from ..fusion import get_kernel_ir_objects
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
                from ..fusion import get_kernel_ir_objects
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
        return vectorized_kernel(pyir_func, debug=debug, no_optims=no_optims)
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
            vec_kernel = vectorized_kernel(pyir_func, debug=debug, no_optims=no_optims)
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
            vec_kernel = vectorized_kernel(pyir_func, debug=debug, no_optims=no_optims)
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

def vectorized_kernel(scalar_kernel, debug=False, no_optims=False):
    """
    Generate a true vectorized kernel by creating a vectorized IR wrapper that loops over the input arrays and calls the scalar kernel for each element.
    Uses logic from pyir.fusion.vectorized.vectorized_fuse_kernels for a single kernel.
    """
    import ctypes
    import numpy as np
    from pyir.fusion.vectorized import vectorized_fuse_kernels
    # Use the vectorized_fuse_kernels logic for a single kernel
    fused_kernel = vectorized_fuse_kernels([scalar_kernel], name=f"{getattr(scalar_kernel, '__name__', 'kernel')}_vectorized", debug=debug, no_optims=no_optims)
    return fused_kernel

def async_gather(*futures, loop=None, return_exceptions=False):
    """
    Orchestrate multiple async kernel launches. Equivalent to asyncio.gather.
    Usage: results = await pyir.async_gather(fut1, fut2, ...)
    """
    loop = loop or asyncio.get_event_loop()
    return asyncio.gather(*futures, loop=loop, return_exceptions=return_exceptions)

def parallel_numpy_kernel(pyir_func, chunk_size: int = 10000, num_threads: int = None):
    """
    Create a parallel NumPy kernel using atomic operations and async execution.
    
    Args:
        pyir_func: PyIR function to parallelize
        chunk_size: Size of chunks for parallel processing
        num_threads: Number of threads to use (default: auto-detect)
    
    Returns:
        Parallel kernel function
    """
    if not _parallel_enabled:
        return numpy_kernel(pyir_func)
    
    # Ensure function is compiled
    pyir_func(1.0, 2.0) if pyir_func.__code__.co_argcount == 2 else pyir_func(1.0, 2.0, 3.0)
    
    def parallel_kernel(*arrays, out=None):
        if np is None:
            raise ImportError("NumPy is not available.")
        
        arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
        n = arrays[0].size
        
        # Validate array sizes
        for arr in arrays:
            assert arr.size == n
        
        if out is None:
            out = np.empty_like(arrays[0])
        
        # Use single-threaded for small arrays
        if n < chunk_size:
            return numpy_kernel(pyir_func)(*arrays, out=out)
        
        # Parallel processing
        num_workers = num_threads or min(_max_parallel_workers, (n + chunk_size - 1) // chunk_size)
        executor = get_parallel_executor()
        
        # Create chunks
        chunks = []
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunk_arrays = [arr[i:end] for arr in arrays]
            chunk_out = out[i:end]
            chunks.append((chunk_arrays, chunk_out))
        
        # Process chunks in parallel
        futures = []
        for chunk_arrays, chunk_out in chunks:
            future = executor.submit(numpy_kernel(pyir_func), *chunk_arrays, out=chunk_out)
            futures.append(future)
        
        # Wait for completion
        concurrent.futures.wait(futures)
        
        # Check for errors
        for future in futures:
            if future.exception():
                raise future.exception()
        
        return out
    
    return parallel_kernel

async def async_numpy_kernel(pyir_func, chunk_size: int = 10000, num_threads: int = None):
    """
    Create an async NumPy kernel for non-blocking execution.
    
    Args:
        pyir_func: PyIR function to async-ize
        chunk_size: Size of chunks for parallel processing
        num_threads: Number of threads to use
    
    Returns:
        Async kernel function
    """
    if not _parallel_enabled:
        # Fallback to synchronous execution
        sync_kernel = numpy_kernel(pyir_func)
        async def fallback_kernel(*arrays, out=None):
            return sync_kernel(*arrays, out=out)
        return fallback_kernel
    
    # Ensure function is compiled
    pyir_func(1.0, 2.0) if pyir_func.__code__.co_argcount == 2 else pyir_func(1.0, 2.0, 3.0)
    
    async def async_kernel(*arrays, out=None):
        if np is None:
            raise ImportError("NumPy is not available.")
        
        arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
        n = arrays[0].size
        
        # Validate array sizes
        for arr in arrays:
            assert arr.size == n
        
        if out is None:
            out = np.empty_like(arrays[0])
        
        # Use single-threaded for small arrays
        if n < chunk_size:
            return numpy_kernel(pyir_func)(*arrays, out=out)
        
        # Async parallel processing
        loop = asyncio.get_event_loop()
        executor = get_parallel_executor()
        
        # Create chunks
        chunks = []
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunk_arrays = [arr[i:end] for arr in arrays]
            chunk_out = out[i:end]
            chunks.append((chunk_arrays, chunk_out))
        
        # Process chunks asynchronously
        tasks = []
        for chunk_arrays, chunk_out in chunks:
            task = loop.run_in_executor(
                executor,
                numpy_kernel(pyir_func),
                *chunk_arrays,
                out=chunk_out
            )
            tasks.append(task)
        
        # Wait for completion
        await asyncio.gather(*tasks)
        
        return out
    
    return async_kernel

def atomic_reduction_kernel(pyir_func, reduction_op='sum'):
    """
    Create an atomic reduction kernel for parallel reductions.
    
    Args:
        pyir_func: PyIR function for reduction
        reduction_op: Reduction operation ('sum', 'max', 'min', 'prod')
    
    Returns:
        Atomic reduction kernel function
    """
    if np is None:
        raise ImportError("NumPy is not available.")
    
    # Ensure function is compiled
    pyir_func(1.0, 2.0)
    
    def atomic_kernel(arr, axis=None, out=None):
        arr = np.asarray(arr)
        
        if axis is None:
            # Flatten and use atomic operations
            arr_flat = arr.ravel()
            n = arr_flat.size
            
            if n == 0:
                return 0.0
            
            # Use parallel processing for large arrays
            if n > 10000 and _parallel_enabled:
                chunk_size = 10000
                num_workers = min(_max_parallel_workers, (n + chunk_size - 1) // chunk_size)
                executor = get_parallel_executor()
                
                # Process chunks in parallel
                futures = []
                for i in range(0, n, chunk_size):
                    end = min(i + chunk_size, n)
                    chunk = arr_flat[i:end]
                    future = executor.submit(lambda x: pyir_func(x[0], x[1]) if len(x) > 1 else x[0], chunk)
                    futures.append(future)
                
                # Collect results
                results = [future.result() for future in futures]
                
                # Combine results
                if reduction_op == 'sum':
                    return sum(results)
                elif reduction_op == 'max':
                    return max(results)
                elif reduction_op == 'min':
                    return min(results)
                elif reduction_op == 'prod':
                    import math
                    return math.prod(results)
            else:
                # Sequential processing
                result = arr_flat[0]
                for i in range(1, arr_flat.size):
                    result = pyir_func(result, arr_flat[i])
                return result
        else:
            # Apply along axis
            return np.apply_along_axis(lambda x: atomic_kernel(x, axis=None), axis, arr)
    
    return atomic_kernel
