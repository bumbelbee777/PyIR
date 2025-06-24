import pyir
import numpy as np
import os
import shutil
import tempfile
import time
import pytest
from pyir import core as pyir_core
from pyir.fusion import get_kernel_ir_objects
import gc

def test_add():
    @pyir.function
    def add(a: pyir.int32, b: pyir.int32) -> pyir.int32:
        result: int
        pyir.inline("""
            %result = add i32 %a, %b
        """)
        return result
    assert add(2, 3) == 5

def test_numpy_kernel():
    @pyir.function
    def muladd(a: pyir.float32, b: pyir.float32, c: pyir.float32) -> pyir.float32:
        tmp: float
        result: float
        pyir.inline("""
            %tmp = fmul float %a, %b
            %result = fadd float %tmp, %c
        """)
        return result
    elemwise = pyir.numpy_kernel(muladd, debug=True)
    x = np.arange(5, dtype=np.float32)
    y = np.ones(5, dtype=np.float32)
    z = np.full(5, 2.0, dtype=np.float32)
    np.testing.assert_allclose(elemwise(x, y, z), np.array([2, 3, 4, 5, 6], dtype=np.float32))

def test_ad():
    @pyir.function
    def square(x: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fmul float %x, %x
        """)
        return result
    grad_fn = pyir.grad(square)
    dx, = grad_fn(3.0)
    assert np.isclose(dx, 6.0)

def test_fusion():
    @pyir.function
    def add(a: pyir.int32, b: pyir.int32) -> pyir.int32:
        result: int
        pyir.inline("""
            %result = add i32 %a, %b
        """)
        return result
    @pyir.function
    def mul(a: pyir.int32, b: pyir.int32) -> pyir.int32:
        result: int
        pyir.inline("""
            %result = mul i32 %a, %b
        """)
        return result
    fused = pyir.fuse_kernels([add, mul])
    out1, out2 = fused(2, 3)
    assert out1 == 5 and out2 == 6

def test_disk_cache_and_memmap():
    """Test disk IR/binary cache and memmap loading."""
    @pyir.function
    def add(a: pyir.int32, b: pyir.int32) -> pyir.int32:
        result: int
        pyir.inline("""
            %result = add i32 %a, %b
        """)
        return result
    # Use a temp cache dir
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cache = pyir_core.PYIR_CACHE_DIR
        pyir_core.PYIR_CACHE_DIR = tmpdir
        os.makedirs(tmpdir, exist_ok=True)
        # Compile and cache kernel
        key = pyir_core.aot_compile_kernel(add)
        # Check files exist
        ir_path = pyir_core._disk_cache_path(key, 'ir')
        bin_path = pyir_core._disk_cache_path(key, 'bin')
        meta_path = pyir_core._disk_cache_path(key, 'meta')
        assert os.path.exists(ir_path)
        assert os.path.exists(bin_path)
        assert os.path.exists(meta_path)
        # Load from disk (should use memmap for binary)
        disk = pyir_core.load_kernel_from_disk(key)
        assert 'ir' in disk and 'binary' in disk and 'meta' in disk
        assert isinstance(disk['binary'], np.memmap)
        # Explicitly close memmap to avoid PermissionError on Windows
        if hasattr(disk['binary'], '_mmap'):
            disk['binary']._mmap.close()
        del disk['binary']
        gc.collect()
        # Clean up
        pyir_core.PYIR_CACHE_DIR = orig_cache

def test_async_aot_compile():
    """Test async AOT kernel compilation returns a Future and completes."""
    @pyir.function
    def add(a: pyir.int32, b: pyir.int32) -> pyir.int32:
        result: int
        pyir.inline("""
            %result = add i32 %a, %b
        """)
        return result
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cache = pyir_core.PYIR_CACHE_DIR
        pyir_core.PYIR_CACHE_DIR = tmpdir
        os.makedirs(tmpdir, exist_ok=True)
        # Async compile
        fut = pyir_core.aot_compile_kernel(add, async_mode=True)
        assert hasattr(fut, 'result')
        key = fut.result(timeout=5)
        # Check files exist
        ir_path = pyir_core._disk_cache_path(key, 'ir')
        assert os.path.exists(ir_path)
        pyir_core.PYIR_CACHE_DIR = orig_cache

def test_load_aot_kernel():
    """Test load_aot_kernel loads from disk if available, else compiles and caches."""
    @pyir.function
    def add(a: pyir.int32, b: pyir.int32) -> pyir.int32:
        result: int
        pyir.inline("""
            %result = add i32 %a, %b
        """)
        return result
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cache = pyir_core.PYIR_CACHE_DIR
        pyir_core.PYIR_CACHE_DIR = tmpdir
        os.makedirs(tmpdir, exist_ok=True)
        # Should not exist yet
        ir_module, _, _ = get_kernel_ir_objects(add)
        ir = str(ir_module)
        name = add.__name__
        dtype_ = getattr(add, '_arg_types', [None])[0] or 'int32'
        target_ = 'cpu'
        fast_math_ = True
        key = pyir_core._kernel_cache_key(name, ir, dtype_, target_, fast_math_)
        assert not os.path.exists(pyir_core._disk_cache_path(key, 'ir'))
        # First call: compiles and caches
        disk = pyir_core.load_aot_kernel(add)
        assert 'ir' in disk and 'binary' in disk
        # Second call: loads from disk
        disk2 = pyir_core.load_aot_kernel(add)
        assert disk2['ir'] == disk['ir']
        # Explicitly close both memmaps to avoid PermissionError on Windows
        if hasattr(disk['binary'], '_mmap'):
            disk['binary']._mmap.close()
        if hasattr(disk2['binary'], '_mmap'):
            disk2['binary']._mmap.close()
        del disk['binary']
        del disk2['binary']
        gc.collect()
        pyir_core.PYIR_CACHE_DIR = orig_cache 

def test_hybridcache_eviction_and_priority():
    """Test HybridCache weighted eviction and user priority (deterministic)."""
    from pyir.core import HybridCache, cache_priority
    cache = HybridCache(max_size=3, name="test")
    # Set all weights to zero except zeta (user_priority)
    cache.set_weights(alpha=0, beta=0, gamma=0, delta=0, epsilon=0, zeta=1)
    class Dummy:
        def __init__(self, name, user_priority=0):
            self._cache_meta = dict(user_priority=user_priority)
            self.name = name
        def __call__(self):
            return self.name
    # Insert 3 kernels with different priorities
    cache['a'] = Dummy('a', user_priority=1)
    cache['b'] = Dummy('b', user_priority=2)
    cache['c'] = Dummy('c', user_priority=3)
    # Insert 'd' with higher priority to trigger eviction
    cache['d'] = Dummy('d', user_priority=4)
    # The lowest-priority kernel should be evicted
    priorities = {k: cache.user_priority[k] for k in cache.data}
    assert min(priorities.values()) >= 2
    # The highest-priority kernel should remain
    assert 'd' in cache
    # Test user priority decorator
    @cache_priority(10)
    def kernel(): pass
    dummy = Dummy('e', user_priority=kernel._cache_meta['user_priority'])
    cache['e'] = dummy
    cache['f'] = Dummy('f', user_priority=0)  # triggers another eviction
    priorities = {k: cache.user_priority[k] for k in cache.data}
    assert min(priorities.values()) >= 3
    assert 'e' in cache

def test_hybridcache_pattern_prefetch():
    """Test HybridCache pattern-driven prefetching (sequential access, robust)."""
    from pyir.core import HybridCache
    cache = HybridCache(max_size=5, name="test")
    # Insert kernels with tuple keys and numeric suffixes
    for i in range(3):
        cache[(f'kernel_{i}',)] = lambda: i
        _ = cache[(f'kernel_{i}',)]
    # Access sequentially to trigger prefetch
    for i in range(3):
        _ = cache[(f'kernel_{i}',)]
    # Prefetch queue should not be empty after sequential access
    assert len(cache.prefetch_queue) > 0

def test_hybridcache_cpu_gpu_levels():
    """Test that CPU and GPU caches are independent and configurable."""
    cpu_cache = pyir_core._cpu_kernel_cache
    gpu_cache = pyir_core._gpu_kernel_cache
    cpu_cache.clear()
    gpu_cache.clear()
    cpu_cache['x'] = lambda: 'cpu'
    gpu_cache['y'] = lambda: 'gpu'
    assert 'x' in cpu_cache
    assert 'y' in gpu_cache
    assert 'y' not in cpu_cache
    assert 'x' not in gpu_cache
    cpu_cache.clear()
    gpu_cache.clear() 

def test_cache_utilities():
    """Test cache utilities."""
    initial_stats = pyir.get_cache_stats()
    @pyir.function
    def add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    add(1.0, 2.0)
    stats = pyir.get_cache_stats()
    assert stats["cpu_cache_size"] >= initial_stats["cpu_cache_size"]
    assert "gpu_cache_size" in stats 