from .core import *
from .helpers import *
from .fusion import *
from .grad import *
from .interop.numpy import *
from .debug.inspect import *
from .backend.cuda import *
from .backend.simd import *
from .security.policy import ExecutionPolicy, validate_policy

# Set up function registry for fusion
from .core import _function_registry
from .fusion import set_function_registry
set_function_registry(_function_registry)

# Export additional fusion utilities
from .fusion import (
    fuse_kernels, as_fusable, FusableKernel,
    autovectorize_kernels, get_kernel_ir_objects, analyze_fusion_compatibility,
    vectorized_fuse_kernels, get_kernel_metadata
)

# Export IR object model utilities
from .core import (
    IRInstr, IRBlock, IRFunction, IRModule,
    create_ir_function_from_string, merge_ir_functions,
    clear_kernel_cache, get_cache_stats,
    set_fast_math,
)

# Export hybrid function registration utilities
from .core import (
    register_function_metadata, get_function_metadata, compile_function_from_metadata,
    async_compile_function_from_metadata, get_compiled_function_count, 
    get_metadata_function_count, clear_compiled_functions, clear_caches,
    shutdown_async_executor, get_function_stats
)

# Export SIMD utilities
from .backend.simd import (
    simd_kernel, autovectorize_kernel, autovectorize_numpy_kernel,
    create_simd_ir_function, vectorize_ir_block,
    detect_simd_capabilities, get_optimal_simd_width,
    vadd, vsub, vmul, vdiv, vload, vstore, vfma, vmax, vmin, vabs, vsqrt
)

from .typing import (
    float16, float32, float64, complex64, complex128,
    int8, int16, int32, int64, bool,
    struct, fnptr, opaque, ptr, void,
    vec, vec4f, vec8f, vec4d, vec4i, vec8i
)
