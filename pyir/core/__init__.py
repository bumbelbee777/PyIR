from .ir import (
    ssa, define_macro, validate_ir, IRInstr, IRBlock, _function_registry,
    IRFunction, IRModule, inline, create_ir_function_from_string, merge_ir_functions,
    _compile_ir, named_types,
)
from .function import function, register_function, _function_registry, register_function_metadata, get_function_metadata, compile_function_from_metadata, async_compile_function_from_metadata, get_compiled_function_count, get_metadata_function_count, clear_compiled_functions, clear_caches, shutdown_async_executor, get_function_stats
from .jit import jit_compile_ir
from .cache import (
    should_use_disk_cache, HybridCache, cache_priority, get_or_register_kernel,
    prefetch_kernel, clear_kernel_cache, get_cache_stats, set_fast_math, _cpu_kernel_cache,
    _gpu_kernel_cache, cpu_kernel_cache, gpu_kernel_cache, _kernel_cache,
    _disk_cache_path, _kernel_cache_key, load_kernel_from_disk,
)
from .aot import (
    aot_compile_kernel, load_aot_kernel
)
from .optims import set_fast_math
from pyir._engine import _engine

# Global debug flag for verbose output
pyir_debug = False

# Any remaining constants, registries, or helpers can be added here as needed.

PYIR_CACHE_DIR = '.pyir_cache'

__all__ = [
    'function', 'register_function', '_function_registry',
    'register_function_metadata', 'get_function_metadata', 'compile_function_from_metadata',
    'async_compile_function_from_metadata', 'get_compiled_function_count', 
    'get_metadata_function_count', 'clear_compiled_functions', 'clear_caches',
    'shutdown_async_executor', 'get_function_stats',
    'jit_compile_ir', '_compile_ir',
    '_kernel_cache', '_cpu_kernel_cache', '_gpu_kernel_cache',
    'cpu_kernel_cache', 'gpu_kernel_cache',
    'PYIR_CACHE_DIR',
    'aot_compile_kernel', 'load_aot_kernel', '_disk_cache_path', '_kernel_cache_key',
    '_engine', 'load_kernel_from_disk',
]
