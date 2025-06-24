# PyIR: Inline LLVM IR Directly From Python 🚀

[![PyPI version](https://badge.fury.io/py/pyir.svg)](https://badge.fury.io/py/pyir)

**Production-grade, ergonomic LLVM IR and numerical kernel system for Python**

---

> **Write, compile, and JIT LLVM IR from Python with type-safe decorators, NumPy/CUDA integration, AD, kernel fusion, complex numbers, and more—all in a small, robust, and extensible package.**

---

## ✨ Features
- **Inline LLVM IR** with ergonomic decorators and helpers
- **Type-safe**: Pythonic type annotations for all arguments and returns
- **NumPy integration**: Write elementwise and reduction kernels for arrays
- **Automatic Differentiation (AD)**: Reverse-mode, with reductions and control flow
- **Kernel Fusion**: Compose and fuse multiple kernels
- **Complex number support**: `complex64`, `complex128`, and Python `complex` type
- **User-defined type inference**: Seamless struct, vector, array, and opaque types
- **CUDA (experimental)**: Write GPU kernels in Python, device memory helpers, grid-stride loops, and `@cuda_kernel` decorator
- **Introspection & Visualization**: Inspect, pretty-print, and visualize IR
- **Extensible**: Add custom ops, gradients, and shape inference

---

## 🚀 Why PyIR? Native Power, Pythonic Ease

**PyIR combines the best of both worlds:**
- **Native C++/LLVM performance** with the ergonomics and flexibility of Python.
- **No recompilation needed:** Change your kernel, rerun, and go—no build steps, no waiting for C++/CUDA compiles.
- **No heavy JIT inspection:** Unlike traditional JITs, PyIR exposes and pretty-prints the IR, lets you inline, inspect, and visualize kernels, and makes debugging and extension easy.
- **Type-safe, robust, and extensible:** Write kernels with Pythonic type annotations, robust type inference, and support for user-defined types, complex numbers, and more.
- **Fusion, AD, and NumPy/CUDA integration:** Compose, differentiate, and accelerate kernels with a single decorator—no boilerplate, no C++ templates, no opaque JIT magic.

**Example:**
- In C++/CUDA, you'd write, compile, and debug kernels, manage memory, and recompile for every change.
- In PyIR, you write a Python function, decorate it, and run—see the IR, fuse, differentiate, and launch on CPU or GPU instantly.

---

## 📦 Installation
```bash
pip install pyir  # (or clone this repo)
```

---

## 🧑‍💻 Quick Example
```python
import pyir

@pyir.function
def add(a: pyir.int32, b: pyir.int32) -> pyir.int32:
    result: int
    pyir.inline("""
        %result = add i32 %a, %b
    """)
    return result

print(add(2, 3))  # 5
```

---

## 🧬 NumPy, Complex, & AD Example
```python
import numpy as np
import pyir

@pyir.function
def muladd(a: pyir.float32, b: pyir.float32, c: pyir.float32) -> pyir.float32:
    tmp: float
    pyir.inline("""
        %tmp = fmul float %a, %b
        %result = fadd float %tmp, %c
    """)
    return result

# NumPy kernel (now always native vectorized)
elemwise = pyir.numpy_kernel(muladd)
x = np.arange(5, dtype=np.float32)
y = np.ones(5, dtype=np.float32)
z = np.full(5, 2.0, dtype=np.float32)
print(elemwise(x, y, z))  # [2. 3. 4. 5. 6.]

# Complex numbers
@pyir.function
def addc(a: pyir.complex64, b: pyir.complex64) -> pyir.complex64:
    result: pyir.complex64
    pyir.inline("""
        %result = insertvalue {float, float} %a, %b, 0
    """)
    return result

# Automatic differentiation
grad_fn = pyir.grad(muladd)
dx, dy, dz = grad_fn(2.0, 3.0, 4.0)
print(f"∂f/∂a={dx}, ∂f/∂b={dy}, ∂f/∂c={dz}")
```

---

## ⚡ CUDA Example (Experimental)
```python
import pyir
import numpy as np

@pyir.cuda_kernel(grid=(16,1,1), block=(256,1,1))
def saxpy(n: pyir.int32, a: pyir.float32, x_ptr, y_ptr, out_ptr):
    pyir.inline(pyir.grid_stride_loop('i', 'n', lambda i: f"""
        %xi = getelementptr float, float* %x_ptr, i32 {i}
        %yi = getelementptr float, float* %y_ptr, i32 {i}
        %outi = getelementptr float, float* %out_ptr, i32 {i}
        %xval = load float, float* %xi
        %yval = load float, float* %yi
        %tmp = fmul float %a, %xval
        %res = fadd float %tmp, %yval
        store float %res, float* %outi
    """))

# Device memory helpers (Numba-backed)
x_dev = pyir.cuda_malloc(1024, dtype=np.float32)
y_dev = pyir.cuda_malloc(1024, dtype=np.float32)
out_dev = pyir.cuda_malloc(1024, dtype=np.float32)
pyir.cuda_memcpy_htod(x_dev, np.ones(1024, dtype=np.float32))
pyir.cuda_memcpy_htod(y_dev, np.ones(1024, dtype=np.float32))
# Launch kernel (stub, see docs for updates)
saxpy(1024, 2.0, x_dev, y_dev, out_dev)
```

---

## 🏎️ SIMD Kernels (Vectorization)

**PyIR makes it easy to write portable, high-performance SIMD kernels:**
- Use `@pyir.simd_kernel` to mark a kernel as vectorized (e.g., 4-wide float32).
- Use vector types like `pyir.vec4f` and SIMD IR helpers (`pyir.vadd`, `pyir.vmul`, ...).
- Write vectorized math, loads, and stores directly in Python.

```python
import pyir

@pyir.simd_kernel(width=4, dtype=pyir.float32)
def vadd4(a: pyir.vec4f, b: pyir.vec4f) -> pyir.vec4f:
    result: pyir.vec4f
    pyir.inline("""
        %result = fadd <4 x float> %a, %b
    """)
    return result

# Use SIMD kernel for fast vector math
x = ... # pointer to <4 x float>
y = ... # pointer to <4 x float>
pyir.vadd(x, y, type='<4 x float>')
```

- PyIR generates portable LLVM IR for SIMD, so your code runs fast on any modern CPU (x86, ARM, etc.).
- You can fuse, differentiate, and inspect SIMD kernels just like scalar ones!

---

## 📚 Documentation & Links
- [API Reference](#) (coming soon)
- [Report Issues](https://github.com/bumbelbee777/pyir/issues)
- [Contributing](#) (coming soon)

---

## 🛠️ License
MIT