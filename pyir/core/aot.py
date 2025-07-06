import concurrent
import itertools

from .ir import _compile_ir, register_function
from .cache import load_kernel_from_disk, save_kernel_to_disk, _kernel_cache_key
from .optims import optimize_ir_module, OptimizationPreset
from .jit import jit_compile_ir as _jit_compile_ir

_current_target = 'cpu'

_fast_math_flags = True

def aot_compile_kernel(fn, dtype=None, target=None, fast_math=None, async_mode=False, opt_level=None, preset=None, no_optims=False):
    """
    Compile a kernel and save IR/binary to disk cache. If async_mode, run in background and return a Future.
    Automatically adds needed LLVM intrinsic declarations if used but not declared (via jit_compile_ir).
    """
    def _compile():
        if hasattr(fn, '_cached_ir'):
            ir = fn._cached_ir
        else:
            from ..fusion import get_kernel_ir_objects
            ir_module, _, _ = get_kernel_ir_objects(fn)
            ir = str(ir_module)
            fn._cached_ir = ir
        name = fn.__name__
        dtype_ = dtype or getattr(fn, '_arg_types', [None])[0] or 'float32'
        target_ = target or _current_target
        fast_math_ = fast_math if fast_math is not None else _fast_math_flags
        key = _kernel_cache_key(name, ir, dtype_, target_, fast_math_, no_optims)
        existing = load_kernel_from_disk(key)
        if existing is not None:
            return key
        if not no_optims:
            # Use performance preset by default for automatic optimization
            optimization_preset = preset or OptimizationPreset.PERFORMANCE
            ir_optimized = optimize_ir_module(ir, fast_math=fast_math_, preset=optimization_preset, opt_level=opt_level)
        else:
            # When no_optims is True, use debug preset (no optimizations)
            ir_optimized = optimize_ir_module(ir, preset=OptimizationPreset.DEBUG)
        # Compile and get binary (simulate: use IR as binary for now)
        register_function(name, ir_optimized)
        _jit_compile_ir(ir_optimized, fn_name=name)
        # Save to disk
        save_kernel_to_disk(key, ir_optimized, binary=ir_optimized.encode(), meta={
            'name': name, 
            'dtype': dtype_, 
            'target': target_, 
            'fast_math': fast_math_,
            'preset': preset.name if preset else None,
            'opt_level': opt_level,
            'no_optims': no_optims
        })
        return key
    if async_mode:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return executor.submit(_compile)
    else:
        return _compile()

def load_aot_kernel(fn, dtype=None, target=None, fast_math=None, opt_level=None, preset=None, no_optims=False):
    """
    Load a kernel from disk cache if available, else compile and cache it. Uses memmap for binary if available.
    """
    if hasattr(fn, '_cached_ir'):
        ir = fn._cached_ir
    else:
        from ..fusion import get_kernel_ir_objects
        ir_module, _, _ = get_kernel_ir_objects(fn)
        ir = str(ir_module)
        fn._cached_ir = ir
    name = fn.__name__
    dtype_ = dtype or getattr(fn, '_arg_types', [None])[0] or 'float32'
    target_ = target or _current_target
    fast_math_ = fast_math if fast_math is not None else _fast_math_flags
    key = _kernel_cache_key(name, ir, dtype_, target_, fast_math_, no_optims)
    disk = load_kernel_from_disk(key)
    if disk is not None:
        # Use memmap binary if available (simulate: just use IR for now)
        # In real use, would load shared object or JIT from binary
        return disk
    # Not on disk: compile and cache
    aot_compile_kernel(fn, dtype, target, fast_math, opt_level=opt_level, preset=preset, no_optims=no_optims)
    return load_kernel_from_disk(key)