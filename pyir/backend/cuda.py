"""
pyir.cuda: CUDA/GPU experimental helpers for PyIR
"""

import functools
import warnings
from ..security.safe_mode import safe_mode

try:
    import numpy as np
    import numba.cuda as nbcuda
    _has_numba = True
except ImportError:
    _has_numba = False

def cuda_malloc(shape, dtype=np.float32):
    """Allocate device memory (returns device array). Runtime checks in safe mode."""
    if not _has_numba:
        raise ImportError("Numba is required for CUDA memory management.")
    arr = nbcuda.device_array(shape, dtype=dtype)
    if safe_mode and arr is None:
        raise RuntimeError("[pyir] cuda_malloc failed: device array is None (safe mode)")
    return arr

def cuda_free(devptr):
    """Free device memory (no-op for Numba, GC handles it)."""
    del devptr

def cuda_memcpy_htod(dst, src):
    """Copy host to device."""
    if not _has_numba:
        raise ImportError("Numba is required for CUDA memcpy.")
    dst.copy_to_device(src)

def cuda_memcpy_dtoh(dst, src):
    """Copy device to host."""
    if not _has_numba:
        raise ImportError("Numba is required for CUDA memcpy.")
    src.copy_to_host(dst)

def grid_stride_loop(start, end, body, label_base='gsloop'):
    """Emit IR for a grid-stride loop (CUDA idiom)."""
    tid = '%tid_x'
    block_dim = '%block_dim'
    grid_dim = '%grid_dim'
    ir = [
        f"{tid} = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()",
        f"{block_dim} = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()",
        f"{grid_dim} = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()",
        f"%stride = mul i32 {block_dim}, {grid_dim}",
        f"%i = add i32 {tid}, 0",
        f"br label %{label_base}_cond",
        f"{label_base}_cond:",
        f"  %cond = icmp slt i32 %i, {end}",
        f"  br i1 %cond, label %{label_base}_body, label %{label_base}_end",
        f"{label_base}_body:",
        f"  " + body('%i'),
        f"  %i_next = add i32 %i, %stride",
        f"  %i = %i_next",
        f"  br label %{label_base}_cond",
        f"{label_base}_end:"
    ]
    return '\n'.join(ir)

def cuda_kernel(grid=(1,1,1), block=(1,1,1)):
    """
    Decorator to mark a function as a CUDA kernel and store launch configuration.
    Usage:
        @pyir.cuda_kernel(grid=(16,1,1), block=(256,1,1))
        def mykernel(...): ...
    """
    def decorator(fn):
        fn._is_cuda_kernel = True
        fn._cuda_grid = grid
        fn._cuda_block = block
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn("[pyir] CUDA kernel launch is not yet implemented. This is a stub.")
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def cuda_tid_x():
    """Emit IR for threadIdx.x (experimental, CUDA only)."""
    return '%tid_x = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()'

def cuda_tid_y():
    """Emit IR for threadIdx.y (experimental, CUDA only)."""
    return '%tid_y = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()'

def cuda_tid_z():
    """Emit IR for threadIdx.z (experimental, CUDA only)."""
    return '%tid_z = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()'

def cuda_block_idx():
    """Emit IR for blockIdx.x (experimental, CUDA only)."""
    return '%block_idx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()'

def cuda_block_dim():
    """Emit IR for blockDim.x (experimental, CUDA only)."""
    return '%block_dim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()'

def cuda_grid_dim():
    """Emit IR for gridDim.x (experimental, CUDA only)."""
    return '%grid_dim = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()'
