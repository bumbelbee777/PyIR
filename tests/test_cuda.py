import pytest
import numpy as np
import pyir

pytestmark = pytest.mark.skipif(
    not hasattr(pyir, 'cuda_malloc'), reason="CUDA/Numba not available"
)

def test_cuda_memory():
    x = pyir.cuda_malloc(10, dtype=np.float32)
    y = pyir.cuda_malloc(10, dtype=np.float32)
    pyir.cuda_memcpy_htod(x, np.arange(10, dtype=np.float32))
    pyir.cuda_memcpy_htod(y, np.ones(10, dtype=np.float32))
    assert x.shape == (10,)
    assert y.shape == (10,)

def test_cuda_kernel_decorator():
    @pyir.cuda_kernel(grid=(1,1,1), block=(1,1,1))
    def dummy_kernel(n: pyir.int32, x_ptr):
        pyir.inline(pyir.grid_stride_loop('i', 'n', lambda i: f"""
            %xi = getelementptr float, float* %x_ptr, i32 {i}
            %xval = load float, float* %xi
        """))
    assert hasattr(dummy_kernel, '_is_cuda_kernel')
    assert dummy_kernel._cuda_grid == (1,1,1)
    assert dummy_kernel._cuda_block == (1,1,1)

def test_grid_stride_loop_ir():
    ir = pyir.grid_stride_loop('0', '10', lambda i: f"%tmp = add i32 {i}, 1")
    assert 'call i32 @llvm.nvvm.read.ptx.sreg.tid.x()' in ir
    assert 'br label %gsloop_cond' in ir 