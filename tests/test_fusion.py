"""
Tests for the refactored fusion logic with IR object model
"""
import pytest
import numpy as np
import pyir

@pytest.fixture(autouse=True)
def clear_pyir_state():
    pyir.clear_kernel_cache()
    if hasattr(pyir, '_function_registry'):
        pyir._function_registry.clear()

def test_basic_fusion():
    """Test basic kernel fusion with IR object model."""
    
    @pyir.function
    def add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    
    @pyir.function
    def mul(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fmul float %a, %b
        """)
        return result
    
    # Test individual kernels
    assert add(2.0, 3.0) == 5.0
    assert mul(2.0, 3.0) == 6.0
    
    # Test fusion
    fused = pyir.fuse_kernels([add, mul], name="add_mul")
    result = fused(2.0, 3.0)
    
    # Should return tuple of (add_result, mul_result)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == 5.0  # add result
    assert result[1] == 6.0  # mul result

def test_fusion_with_named_outputs():
    """Test fusion with named outputs."""
    
    @pyir.function
    def add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    
    @pyir.function
    def mul(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fmul float %a, %b
        """)
        return result
    
    fused = pyir.fuse_kernels(
        [add, mul], 
        name="add_mul_named",
        output_names=["sum", "product"],
        return_type="dict"
    )
    
    result = fused(2.0, 3.0)
    assert isinstance(result, dict)
    assert result["sum"] == 5.0
    assert result["product"] == 6.0

def test_fusion_compatibility_analysis():
    """Test fusion compatibility analysis."""
    
    @pyir.function
    def add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    
    @pyir.function
    def mul(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fmul float %a, %b
        """)
        return result
    
    # Test compatibility analysis
    analysis = pyir.analyze_fusion_compatibility([add, mul])
    assert analysis["compatible"] == True
    assert len(analysis["issues"]) == 0
    assert analysis["arg_count"] == 2
    assert "a" in analysis["arg_types"]
    assert "b" in analysis["arg_types"]

def test_ir_object_utilities():
    """Test IR object model utilities."""
    
    @pyir.function
    def add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    
    # Test kernel metadata extraction
    metadata = pyir.get_kernel_metadata(add)
    assert metadata["name"] == "add"
    assert metadata["is_cuda"] == False
    assert metadata["is_simd"] == False
    assert "a" in metadata["parameters"]
    assert "b" in metadata["parameters"]
    
    # Test IR object extraction
    ir_objects = pyir.get_kernel_ir_objects(add)
    assert ir_objects is not None
    ir_module, output_vars, output_names = ir_objects
    assert isinstance(ir_module, pyir.IRModule)
    assert len(ir_module.functions) > 0

def test_simd_integration():
    """Test SIMD integration with fusion."""
    
    @pyir.simd_kernel(width=4, dtype=pyir.float32)
    @pyir.function
    def vadd(a: pyir.vec4f, b: pyir.vec4f) -> pyir.vec4f:
        result: pyir.vec4f
        pyir.inline("""
            %result = fadd <4 x float> %a, %b
        """)
        return result
    
    @pyir.simd_kernel(width=4, dtype=pyir.float32)
    @pyir.function
    def vmul(a: pyir.vec4f, b: pyir.vec4f) -> pyir.vec4f:
        result: pyir.vec4f
        pyir.inline("""
            %result = fmul <4 x float> %a, %b
        """)
        return result
    
    # Test SIMD kernel metadata
    metadata = pyir.get_kernel_metadata(vadd)
    assert metadata["is_simd"] == True
    assert metadata["simd_width"] == 4
    assert metadata["simd_dtype"] == pyir.float32
    
    # Test SIMD fusion compatibility
    analysis = pyir.analyze_fusion_compatibility([vadd, vmul])
    assert analysis["compatible"] == True
    assert analysis["device_types"]["simd"] == True

def test_autovectorization():
    """Test autovectorization utilities."""
    
    @pyir.function
    def add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    
    @pyir.function
    def mul(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fmul float %a, %b
        """)
        return result
    
    # Test autovectorization
    simd_kernels = pyir.autovectorize_kernels([add, mul], width=4, dtype=pyir.float32)
    
    # The result should be a fused SIMD kernel
    metadata = pyir.get_kernel_metadata(simd_kernels)
    assert metadata["is_simd"] == True
    assert metadata["simd_width"] == 4

def test_cache_utilities():
    """Test cache utilities."""
    # Get initial cache stats
    initial_stats = pyir.get_cache_stats()
    @pyir.function
    def add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    # Call function to populate cache
    add(1.0, 2.0)
    # Check cache stats
    stats = pyir.get_cache_stats()
    assert stats["cpu_cache_size"] >= initial_stats["cpu_cache_size"]
    assert "gpu_cache_size" in stats
    # Clear cache
    pyir.clear_kernel_cache()
    stats_after_clear = pyir.get_cache_stats()
    assert stats_after_clear["cpu_cache_size"] == 0

def test_operator_overloading():
    """Test fusion operator overloading."""
    
    @pyir.function
    def add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    
    @pyir.function
    def mul(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fmul float %a, %b
        """)
        return result
    
    # Test operator overloading
    add_fusable = pyir.as_fusable(add)
    mul_fusable = pyir.as_fusable(mul)
    
    fused = add_fusable + mul_fusable
    result = fused(2.0, 3.0)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == 5.0  # add result
    assert result[1] == 6.0  # mul result

if __name__ == "__main__":
    # Run tests
    test_basic_fusion()
    test_fusion_with_named_outputs()
    test_fusion_compatibility_analysis()
    test_ir_object_utilities()
    test_simd_integration()
    test_autovectorization()
    test_cache_utilities()
    test_operator_overloading()
    print("All fusion tests passed!") 