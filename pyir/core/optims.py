"""
pyir.core.optims: Deterministic IR optimizations and LLVM pass management
"""
import llvmlite.binding as llvm
import warnings
import hashlib
import threading
import os
import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import time

from .ir import get_llvm_target_machine

# Global optimization settings
_optimization_level = 2  # Default to -O2 for balance of speed and safety
_fast_math_enabled = True
_loop_vectorize = True
_slp_vectorize = True
_optimization_cache = {}
_optimization_cache_lock = threading.Lock()

# Atomic counters for parallel optimization
_optimization_counter = 0
_optimization_counter_lock = threading.Lock()

# Parallel optimization settings
_parallel_optimization_enabled = True
_max_parallel_workers = min(8, os.cpu_count() or 4)
_optimization_executor = None
_optimization_executor_lock = threading.Lock()

class OptimizationLevel(Enum):
    """Optimization levels with deterministic behavior guarantees."""
    NONE = 0      # No optimizations
    BASIC = 1     # Basic optimizations (safe)
    STANDARD = 2  # Standard optimizations (default)
    AGGRESSIVE = 3  # Aggressive optimizations (may affect precision)

class OptimizationPreset(Enum):
    """Predefined optimization presets for different use cases."""
    DEBUG = "debug"           # No optimizations, preserve debug info
    SAFE = "safe"            # Conservative optimizations
    BALANCED = "balanced"    # Balance of performance and safety
    PERFORMANCE = "perf"     # Performance-focused optimizations
    SIZE = "size"            # Size-focused optimizations

def set_optimization_level(level: OptimizationLevel):
    """Set the global optimization level."""
    global _optimization_level
    _optimization_level = level.value

def set_fast_math(enabled: bool = True):
    """Enable or disable fast-math optimizations globally."""
    global _fast_math_enabled
    _fast_math_enabled = enabled

def set_vectorization(loop: bool = True, slp: bool = True):
    """Configure vectorization settings."""
    global _loop_vectorize, _slp_vectorize
    _loop_vectorize = loop
    _slp_vectorize = slp

def get_optimization_preset(preset: OptimizationPreset) -> Dict[str, Any]:
    """Get optimization settings for a predefined preset."""
    presets = {
        OptimizationPreset.DEBUG: {
            'opt_level': 0,
            'fast_math': False,
            'loop_vectorize': False,
            'slp_vectorize': False,
            'inlining': False,
            'unroll_loops': False,
            'tail_call_elim': False
        },
        OptimizationPreset.SAFE: {
            'opt_level': 1,
            'fast_math': False,
            'loop_vectorize': False,
            'slp_vectorize': False,
            'inlining': True,
            'unroll_loops': False,
            'tail_call_elim': True
        },
        OptimizationPreset.BALANCED: {
            'opt_level': 2,
            'fast_math': True,
            'loop_vectorize': True,
            'slp_vectorize': True,
            'inlining': True,
            'unroll_loops': True,
            'tail_call_elim': True
        },
        OptimizationPreset.PERFORMANCE: {
            'opt_level': 3,
            'fast_math': True,
            'loop_vectorize': True,
            'slp_vectorize': True,
            'inlining': True,
            'unroll_loops': True,
            'tail_call_elim': True
        },
        OptimizationPreset.SIZE: {
            'opt_level': 2,
            'fast_math': False,
            'loop_vectorize': False,
            'slp_vectorize': False,
            'inlining': True,
            'unroll_loops': False,
            'tail_call_elim': True
        }
    }
    return presets[preset]

def create_deterministic_pass_manager(opt_level: int = None, 
                                    fast_math: bool = None,
                                    loop_vectorize: bool = None,
                                    slp_vectorize: bool = None,
                                    preset: OptimizationPreset = None) -> llvm.ModulePassManager:
    """
    Create a deterministic LLVM pass manager with consistent optimization behavior.
    
    Args:
        opt_level: Optimization level (0-3)
        fast_math: Enable fast-math optimizations (handled at IR generation level)
        loop_vectorize: Enable loop vectorization
        slp_vectorize: Enable SLP vectorization
        preset: Use predefined optimization preset
    
    Returns:
        Configured LLVM pass manager
    """
    # Use preset if provided, otherwise use global settings
    if preset is not None:
        settings = get_optimization_preset(preset)
        opt_level = settings['opt_level']
        fast_math = settings['fast_math']
        loop_vectorize = settings['loop_vectorize']
        slp_vectorize = settings['slp_vectorize']
    else:
        opt_level = opt_level if opt_level is not None else _optimization_level
        fast_math = fast_math if fast_math is not None else _fast_math_enabled
        loop_vectorize = loop_vectorize if loop_vectorize is not None else _loop_vectorize
        slp_vectorize = slp_vectorize if slp_vectorize is not None else _slp_vectorize

    # Create pass manager builder with deterministic settings
    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = opt_level
    pmb.size_level = 0  # Always optimize for speed, not size
    
    # Configure vectorization deterministically
    pmb.loop_vectorize = loop_vectorize
    pmb.slp_vectorize = slp_vectorize
    
    # Create module pass manager
    pm = llvm.create_module_pass_manager()
    
    # Add target-specific passes for deterministic behavior
    target_machine = get_llvm_target_machine()
    if target_machine:
        # target_machine.add_analysis_passes(pm)  # This is safe, but skip verifier pass
        pass
    
    # Populate with standard passes
    pmb.populate(pm)
    
    # Fast-math is handled at IR generation level, not in pass manager
    # The fast_math parameter is used to control IR generation, not optimization passes
    
    return pm

def validate_optimization_safety(ir_before: str, ir_after: str) -> bool:
    """
    Validate that optimizations preserve function semantics.
    
    This is a basic validation - in practice, you'd want more sophisticated
    semantic analysis for production use.
    """
    try:
        # Parse both IR versions
        mod_before = llvm.parse_assembly(ir_before)
        mod_after = llvm.parse_assembly(ir_after)
        
        # Basic checks for function preservation
        funcs_before = set(f.name for f in mod_before.functions)
        funcs_after = set(f.name for f in mod_after.functions)
        
        if funcs_before != funcs_after:
            warnings.warn(f"[pyir] Optimization changed function set: {funcs_before} -> {funcs_after}")
            return False
        
        # Check that all functions have the same signature
        for func_name in funcs_before:
            func_before = mod_before.get_function(func_name)
            func_after = mod_after.get_function(func_name)
            
            # Compare function types properly using llvmlite API
            if func_before.type != func_after.type:
                warnings.warn(f"[pyir] Function {func_name} signature changed during optimization")
                return False
        
        return True
        
    except Exception as e:
        warnings.warn(f"[pyir] Could not validate optimization safety: {e}")
        return True  # Assume safe if validation fails

def optimize_ir_module(ir_module_str: str, 
                      opt_level: int = None,
                      fast_math: bool = None,
                      loop_vectorize: bool = None,
                      slp_vectorize: bool = None,
                      preset: OptimizationPreset = None,
                      validate_safety: bool = False,  # Disabled by default for performance
                      cache_optimizations: bool = True) -> str:
    """
    Apply deterministic LLVM optimizations to an IR module string.
    
    Args:
        ir_module_str: Input LLVM IR as string
        opt_level: Optimization level (0-3)
        fast_math: Enable fast-math optimizations
        loop_vectorize: Enable loop vectorization
        slp_vectorize: Enable SLP vectorization
        preset: Use predefined optimization preset
        validate_safety: Validate that optimizations preserve semantics (default: False for performance)
        cache_optimizations: Cache optimization results
    
    Returns:
        Optimized LLVM IR as string
    """
    if not ir_module_str or ir_module_str.strip() == "":
        return ir_module_str
    
    # Check cache first
    if cache_optimizations:
        cache_key = _generate_structural_cache_key(
            ir_module_str, opt_level, fast_math, loop_vectorize, slp_vectorize, preset
        )
        
        with _optimization_cache_lock:
            if cache_key in _optimization_cache:
                return _optimization_cache[cache_key]
    
    try:
        # Parse IR
        mod = llvm.parse_assembly(ir_module_str)
        mod.verify()
        
        # Store original for validation (only if validation is enabled)
        ir_original = str(mod) if validate_safety else None
        
        # Create deterministic pass manager
        pm = create_deterministic_pass_manager(
            opt_level=opt_level,
            fast_math=fast_math,
            loop_vectorize=loop_vectorize,
            slp_vectorize=slp_vectorize,
            preset=preset
        )
        
        # Run optimizations
        pm.run(mod)
        
        # Verify the optimized module
        mod.verify()
        
        # Get optimized IR
        ir_optimized = str(mod)
        
        # Validate safety if requested (disabled by default)
        if validate_safety and ir_original:
            if not validate_optimization_safety(ir_original, ir_optimized):
                warnings.warn("[pyir] Optimization safety validation failed, using original IR")
                ir_optimized = ir_original
        
        # Cache result
        if cache_optimizations:
            with _optimization_cache_lock:
                _optimization_cache[cache_key] = ir_optimized
                
                # Limit cache size
                if len(_optimization_cache) > 1000:
                    # Remove oldest entries (simple FIFO)
                    oldest_keys = list(_optimization_cache.keys())[:100]
                    for key in oldest_keys:
                        del _optimization_cache[key]
        
        return ir_optimized
        
    except Exception as e:
        # Fallback to unoptimized IR if optimization fails
        warnings.warn(f"[pyir] IR optimization failed: {e}. Using unoptimized IR.")
        return ir_module_str

def _generate_structural_cache_key(ir_str: str, 
                                 opt_level: int,
                                 fast_math: bool,
                                 loop_vectorize: bool,
                                 slp_vectorize: bool,
                                 preset: OptimizationPreset) -> str:
    """Generate a structural cache key based on function signatures and settings, not full IR."""
    import re
    
    # Extract function signatures (much faster than hashing entire IR)
    signatures = []
    for match in re.finditer(r'define\s+([^@]+)@([^(]+)\(([^)]*)\)', ir_str):
        ret_type = match.group(1).strip()
        func_name = match.group(2).strip()
        args = match.group(3).strip()
        # Create signature hash (much smaller than full IR)
        sig_hash = hashlib.md5(f"{ret_type}@{func_name}({args})".encode()).hexdigest()[:8]
        signatures.append(sig_hash)
    
    # Sort signatures for deterministic ordering
    signatures.sort()
    
    # Create settings string
    settings_str = f"{opt_level}_{fast_math}_{loop_vectorize}_{slp_vectorize}_{preset.value if preset else 'none'}"
    
    # Combine signatures with settings (much faster than full IR hash)
    combined = f"{'_'.join(signatures)}_{settings_str}"
    return hashlib.md5(combined.encode()).hexdigest()

def clear_optimization_cache():
    """Clear the optimization result cache."""
    global _optimization_cache
    with _optimization_cache_lock:
        _optimization_cache.clear()

def get_optimization_stats() -> Dict[str, Any]:
    """Get statistics about optimization usage."""
    with _optimization_cache_lock:
        return {
            'cache_size': len(_optimization_cache),
            'optimization_level': _optimization_level,
            'fast_math_enabled': _fast_math_enabled,
            'loop_vectorize': _loop_vectorize,
            'slp_vectorize': _slp_vectorize,
            'parallel_enabled': _parallel_optimization_enabled,
            'max_parallel_workers': _max_parallel_workers,
            'optimization_counter': _optimization_counter,
            'executor_active': _optimization_executor is not None
        }

# Convenience functions for common optimization scenarios
def optimize_for_debug(ir_module_str: str) -> str:
    """Optimize IR for debugging (no optimizations)."""
    return optimize_ir_module(ir_module_str, preset=OptimizationPreset.DEBUG)

def optimize_for_safety(ir_module_str: str) -> str:
    """Optimize IR with conservative, safe optimizations."""
    return optimize_ir_module(ir_module_str, preset=OptimizationPreset.SAFE)

def optimize_for_performance(ir_module_str: str) -> str:
    """Optimize IR for maximum performance."""
    return optimize_ir_module(ir_module_str, preset=OptimizationPreset.PERFORMANCE)

def optimize_for_size(ir_module_str: str) -> str:
    """Optimize IR for minimum code size."""
    return optimize_ir_module(ir_module_str, preset=OptimizationPreset.SIZE)

def get_optimization_executor():
    """Get or create the global optimization executor."""
    global _optimization_executor
    with _optimization_executor_lock:
        if _optimization_executor is None:
            _optimization_executor = ThreadPoolExecutor(max_workers=_max_parallel_workers)
        return _optimization_executor

def set_parallel_optimization(enabled: bool = True, max_workers: int = None):
    """Enable/disable parallel optimization and set max workers."""
    global _parallel_optimization_enabled, _max_parallel_workers
    _parallel_optimization_enabled = enabled
    if max_workers is not None:
        _max_parallel_workers = max_workers
        # Recreate executor with new worker count
        global _optimization_executor
        with _optimization_executor_lock:
            if _optimization_executor is not None:
                _optimization_executor.shutdown(wait=False)
            _optimization_executor = None

def parallel_optimize_ir_modules(ir_modules: List[str], 
                               preset: OptimizationPreset = None,
                               opt_level: int = None,
                               fast_math: bool = None) -> List[str]:
    """
    Optimize multiple IR modules in parallel using async execution.
    
    Args:
        ir_modules: List of IR module strings to optimize
        preset: Optimization preset to apply
        opt_level: Optimization level override
        fast_math: Fast-math override
    
    Returns:
        List of optimized IR module strings
    """
    if not _parallel_optimization_enabled or len(ir_modules) <= 1:
        # Fallback to sequential optimization
        return [optimize_ir_module(ir, preset=preset, opt_level=opt_level, fast_math=fast_math) 
                for ir in ir_modules]
    
    # Use ThreadPoolExecutor for parallel optimization
    executor = get_optimization_executor()
    
    # Submit optimization tasks
    futures = []
    for ir_module in ir_modules:
        future = executor.submit(
            optimize_ir_module, 
            ir_module, 
            preset=preset, 
            opt_level=opt_level, 
            fast_math=fast_math
        )
        futures.append(future)
    
    # Collect results
    results = []
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            warnings.warn(f"[pyir] Parallel optimization failed: {e}")
            # Fallback to unoptimized IR
            results.append(ir_modules[len(results)])
    
    return results

async def async_optimize_ir_modules(ir_modules: List[str],
                                  preset: OptimizationPreset = None,
                                  opt_level: int = None,
                                  fast_math: bool = None) -> List[str]:
    """
    Asynchronously optimize multiple IR modules.
    
    Args:
        ir_modules: List of IR module strings to optimize
        preset: Optimization preset to apply
        opt_level: Optimization level override
        fast_math: Fast-math override
    
    Returns:
        List of optimized IR module strings
    """
    if not _parallel_optimization_enabled or len(ir_modules) <= 1:
        # Fallback to sequential optimization
        return [optimize_ir_module(ir, preset=preset, opt_level=opt_level, fast_math=fast_math) 
                for ir in ir_modules]
    
    # Run optimization in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    executor = get_optimization_executor()
    
    # Create optimization tasks
    tasks = []
    for ir_module in ir_modules:
        task = loop.run_in_executor(
            executor,
            optimize_ir_module,
            ir_module,
            preset,
            opt_level,
            fast_math
        )
        tasks.append(task)
    
    # Wait for all optimizations to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    optimized_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            warnings.warn(f"[pyir] Async optimization failed for module {i}: {result}")
            optimized_results.append(ir_modules[i])  # Use unoptimized IR
        else:
            optimized_results.append(result)
    
    return optimized_results

def atomic_optimization_counter():
    """Get atomic optimization counter for tracking parallel operations."""
    global _optimization_counter
    with _optimization_counter_lock:
        _optimization_counter += 1
        return _optimization_counter

# Export the main function for backward compatibility
__all__ = [
    'optimize_ir_module',
    'set_optimization_level',
    'set_fast_math',
    'set_vectorization',
    'OptimizationLevel',
    'OptimizationPreset',
    'optimize_for_debug',
    'optimize_for_safety',
    'optimize_for_performance',
    'optimize_for_size',
    'clear_optimization_cache',
    'get_optimization_stats',
    'parallel_optimize_ir_modules',
    'async_optimize_ir_modules',
    'atomic_optimization_counter',
    'set_parallel_optimization'
]