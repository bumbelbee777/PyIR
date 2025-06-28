from .core import fuse_kernels

def autovectorize_kernels(fns, width=4, dtype=None, name_prefix="simd_"):
    """
    Given a list of scalar kernels, generate SIMD versions and fuse them using IR object model.
    - Uses @simd_kernel to wrap each kernel.
    - Fuses the resulting SIMD kernels with optimized IR object operations.
    - Returns the fused SIMD kernel.
    Supports kernels with multiple outputs.
    """
    from ..backend.simd import simd_kernel
    simd_kernels = []
    for fn in fns:
        # Apply simd_kernel to the existing function (modifies it in place)
        simd_fn = simd_kernel(width=width, dtype=dtype)(fn)
        simd_kernels.append(simd_fn)
    fused = fuse_kernels(simd_kernels, name=name_prefix+"fused", register=True)
    return fused