import pytest
import numpy as np
import pyir
from pyir.core.function import set_autofusion_window, _autofusion_queue, _autofusion_history
import asyncio

# Helper to clear autofusion state before each test
def clear_autofusion_state():
    _autofusion_queue.clear()
    _autofusion_history.clear()

@pytest.fixture(autouse=True)
def autofusion_isolation():
    clear_autofusion_state()
    yield
    clear_autofusion_state()

def make_add():
    @pyir.function
    def add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    return add

def make_mul():
    @pyir.function
    def mul(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fmul float %a, %b
        """)
        return result
    return mul

def make_sub():
    @pyir.function
    def sub(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fsub float %a, %b
        """)
        return result
    return sub

def make_triple():
    @pyir.function
    def triple(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
            %result = fmul float %result, 3.0
        """)
        return result
    return triple

def make_async_add():
    @pyir.function
    async def add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    return add

def make_simd_add():
    @pyir.function(simd=True)
    def simd_add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    return simd_add

def make_tuple_kernel():
    @pyir.function
    def tup(a: pyir.float32, b: pyir.float32) -> tuple:
        x: float
        y: float
        pyir.inline("""
            %x = fadd float %a, %b
            %y = fmul float %a, %b
        """)
        return (x, y)
    return tup

def make_simd_mul():
    @pyir.function(simd=True)
    def simd_mul(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fmul float %a, %b
        """)
        return result
    return simd_mul

def make_async_mul():
    @pyir.function
    async def mul(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fmul float %a, %b
        """)
        return result
    return mul

def test_autofusion_basic():
    add = make_add()
    mul = make_mul()
    add(1.0, 2.0)
    mul(3.0, 4.0)
    fused = getattr(mul, '_fused', None)
    assert fused is not None
    out = fused(2.0, 3.0)
    assert np.isclose(out[0], 5.0)
    assert np.isclose(out[1], 6.0)
    assert getattr(fused, '_is_fused', False)
    assert getattr(fused, '_fusion_history', None)

def test_autofusion_window():
    set_autofusion_window(3)
    add = make_add()
    mul = make_mul()
    sub = make_sub()
    add(1.0, 2.0)
    mul(3.0, 4.0)
    sub(5.0, 1.0)
    fused = sub._fused
    assert fused is not None
    out = fused(2.0, 3.0)
    assert np.isclose(out[0], 5.0)
    assert np.isclose(out[1], 6.0)
    assert np.isclose(out[2], -1.0)
    assert len(fused._fusion_history[-1]) == 3
    set_autofusion_window(2)  # Reset

def test_autofusion_optout():
    @pyir.function(autofusion=False)
    def add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    mul = make_mul()
    add(1.0, 2.0)
    mul(3.0, 4.0)
    assert not hasattr(mul, '_fused') or mul._fused is None

def test_autofusion_incompatible():
    add = make_add()
    @pyir.function
    def sub(a: pyir.float32, b: pyir.float32, c: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fsub float %a, %b
        """)
        return result
    add(1.0, 2.0)
    sub(3.0, 4.0, 5.0)
    assert not hasattr(sub, '_fused') or sub._fused is None

def test_autofusion_device_attr():
    @pyir.function(simd=True)
    def simd_add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    mul = make_mul()
    simd_add(1.0, 2.0)
    mul(3.0, 4.0)
    assert not hasattr(mul, '_fused') or mul._fused is None

def test_autofusion_redundant():
    add = make_add()
    mul = make_mul()
    add(1.0, 2.0)
    mul(3.0, 4.0)
    fused1 = mul._fused
    # Call again, should not create a new fusion
    add(5.0, 6.0)
    mul(7.0, 8.0)
    fused2 = mul._fused
    assert fused1 is fused2
    assert len(fused1._fusion_history) == 1

def test_autofusion_metadata():
    add = make_add()
    mul = make_mul()
    add(1.0, 2.0)
    mul(3.0, 4.0)
    fused = mul._fused
    assert hasattr(fused, '_fusion_history')
    assert hasattr(fused, '_is_fused')
    assert fused._is_fused
    assert fused._fusion_history

def test_autofusion_async():
    add = make_async_add()
    mul = make_async_mul()
    async def run():
        await add(1.0, 2.0)
        await mul(3.0, 4.0)
        fused = getattr(mul, '_fused', None)
        assert fused is not None
        assert getattr(fused, '_is_async_kernel', False)
        out = await fused(2.0, 3.0)
        assert np.isclose(out[0], 5.0)  # add(2,3)
        assert np.isclose(out[1], 6.0)  # mul(2,3)
    asyncio.run(run())

def test_autofusion_simd():
    add = make_simd_add()
    mul = make_simd_mul()
    add(1.0, 2.0)
    mul(3.0, 4.0)
    fused = getattr(mul, '_fused', None)
    assert fused is not None
    assert getattr(fused, '_is_simd_kernel', False)
    out = fused(2.0, 3.0)
    assert np.isclose(out[0], 5.0)
    assert np.isclose(out[1], 6.0)

def test_autofusion_tuple_output():
    tup1 = make_tuple_kernel()
    tup2 = make_tuple_kernel()
    tup1(1.0, 2.0)
    tup2(3.0, 4.0)
    fused = getattr(tup2, '_fused', None)
    assert fused is not None
    out = fused(2.0, 3.0)
    assert isinstance(out[0], tuple) and isinstance(out[1], tuple)
    assert np.isclose(out[0][0], 5.0) and np.isclose(out[0][1], 6.0)
    assert np.isclose(out[1][0], 5.0) and np.isclose(out[1][1], 6.0)

def test_autofusion_side_effects():
    @pyir.function
    def pure(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fadd float %a, %b
        """)
        return result
    @pyir.function
    def impure(a: pyir.float32, b: pyir.float32) -> pyir.float32:
        result: float
        pyir.inline("""
            %result = fmul float %a, %b
        """)
        return result
    impure._has_side_effects = True  # Set after definition
    pure(1.0, 2.0)
    impure(3.0, 4.0)
    assert getattr(impure, '_fused', None) is None

def test_autofusion_order_sensitivity():
    add = make_add()
    mul = make_mul()
    add(1.0, 2.0)
    mul(3.0, 4.0)
    fused1 = mul._fused
    # Reverse order
    add2 = make_add()
    mul2 = make_mul()
    mul2(3.0, 4.0)
    add2(1.0, 2.0)
    fused2 = add2._fused
    assert fused1 is not fused2

def test_autofusion_dynamic_optout():
    add = make_add()
    mul = make_mul()
    add(1.0, 2.0)
    mul(3.0, 4.0)
    fused1 = mul._fused
    # Dynamically opt out add
    add._no_autofusion = True
    add(5.0, 6.0)
    mul(7.0, 8.0)
    fused2 = mul._fused
    assert fused1 is fused2  # No new fusion

def test_autofusion_large_window():
    set_autofusion_window(4)
    add = make_add()
    mul = make_mul()
    sub = make_sub()
    triple = make_triple()
    add(1.0, 2.0)
    mul(3.0, 4.0)
    sub(5.0, 1.0)
    triple(2.0, 2.0)
    fused = triple._fused
    assert fused is not None
    out = fused(2.0, 3.0)
    assert len(out) == 4
    set_autofusion_window(2) 