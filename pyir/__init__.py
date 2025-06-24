from .core import *
from .helpers import *
from .fusion import *
from .ad import *
from .numpy import *
from .inspect import *
from .cuda import *
from .simd import *
from .policy import ExecutionPolicy, validate_policy

# Set up function registry for fusion
from .core import _function_registry
from .fusion import set_function_registry
set_function_registry(_function_registry)

# Export additional fusion utilities
from .fusion import (
    fuse_kernels, fuse, as_fusable, FusableKernel,
    autovectorize_kernels, get_kernel_ir_objects, analyze_fusion_compatibility,
    vectorized_fuse_kernels
)

# Export IR object model utilities
from .core import (
    IRInstr, IRBlock, IRFunction, IRModule,
    create_ir_function_from_string, merge_ir_functions,
    get_kernel_metadata, clear_kernel_cache, get_cache_stats,
    set_fast_math, optimize_ir_module
)

# Export SIMD utilities
from .simd import (
    simd_kernel, autovectorize_kernel, autovectorize_numpy_kernel,
    create_simd_ir_function, vectorize_ir_block,
    detect_simd_capabilities, get_optimal_simd_width,
    vadd, vsub, vmul, vdiv, vload, vstore, vfma, vmax, vmin, vabs, vsqrt
)

from .typing import (
    float16, float32, float64, complex64, complex128,
    int8, int16, int32, int64, bool,
    struct, fnptr, opaque, ptr, void,
    vec
)
