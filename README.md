# PyIR: Inline LLVM IR Directly From Python 🚀

**Powerful, advanced, ergonomic LLVM IR and numerical kernel system for Python**

---

> **Write, compile, and JIT LLVM IR from Python with type-safe decorators, NumPy integration, automatic differentiation, kernel fusion, complex numbers, and more—all in a small, robust, and extensible package.**

---

## ✨ Features
- **Inline LLVM IR** with ergonomic decorators and helpers
- **Type-safe**: Pythonic type annotations for all arguments and returns
- **NumPy integration**: Write elementwise and reduction kernels for arrays with multiple execution policies (vectorized, parallel, async, sandboxed)
- **Automatic Differentiation (AD)**: Reverse-mode, forward-mode (JVP), higher-order, and fused kernel support
- **Kernel Fusion**: Compose and fuse multiple kernels with IR object model
- **Advanced Caching**: Multi-level, hybrid LFU-LRU, async prefetch, disk/memmap caching
- **Complex number support**: `complex64`, `complex128`, and Python `complex` type
- **User-defined types**: Seamless struct, vector, array, and opaque types
- **SIMD/Vectorization**: Write portable, high-performance SIMD kernels with `@simd_kernel`
- **Async/Parallel Execution**: Run kernels with async/await, thread pools, or parallel policies
- **Introspection & Visualization**: Inspect, pretty-print, and visualize IR
- **Extensible**: Add custom ops, gradients, and shape inference
- **Security**: Sandboxed execution policies for safe kernel evaluation

---

## 🚀 Why PyIR? Native Power, Pythonic Ease

**PyIR combines the best of both worlds:**
- **Native C++/LLVM performance** with the ergonomics and flexibility of Python
- **No recompilation needed:** Change your kernel, rerun, and go—no build steps, no waiting for C++/CUDA compiles
- **No heavy JIT inspection:** Unlike traditional JITs, PyIR exposes and pretty-prints the IR, lets you inline, inspect, and visualize kernels, and makes debugging and extension easy
- **Type-safe, robust, and extensible:** Write kernels with Pythonic type annotations, robust type inference, and support for user-defined types, complex numbers, and more
- **Fusion, AD, async, and NumPy integration:** Compose, differentiate, and accelerate kernels with a single decorator—no boilerplate, no C++ templates, no opaque JIT magic
- **Advanced caching:** Multi-level, async, and disk caching for instant kernel/data loading and blazing performance

**PyIR is the only Python system with ergonomic LLVM IR, robust AD (reverse/forward/higher-order), kernel fusion, async, complex, SIMD, and advanced caching—all in a single, extensible package.**

---

## 🥇 PyIR vs. Numba: Why PyIR is More Practical and Feature-Rich

| Feature                | PyIR                | Numba           |
|------------------------|---------------------|-----------------|
| **LLVM IR access**     | Full, ergonomic     | No              |
| **Kernel fusion**      | Yes (sync/async/complex) | No        |
| **Reverse-mode AD**    | Yes                 | No              |
| **Forward-mode AD**    | Yes                 | No              |
| **Higher-order AD**    | Yes                 | No              |
| **Complex numbers**    | Yes                 | Partial         |
| **SIMD/Vectorization** | Yes (portable)      | Yes (limited)   |
| **Async kernels**      | Yes                 | No              |
| **Multi-level caching**| Yes (LFU/LRU, disk, memmap, async) | No |
| **Custom ops/gradients**| Yes                | No              |
| **Type system**        | Modern, Pythonic    | Python native   |
| **IR introspection**   | Yes                 | No              |
| **Extensibility**      | High                | Medium          |
| **Security**           | Sandboxed execution | No              |

**PyIR is the only Python system with ergonomic LLVM IR, robust AD (reverse/forward/higher-order), kernel fusion, async, complex, SIMD, and advanced caching—all in a single, extensible package.**

---

## 📦 Installation
```bash
pip install -e .
```

---

## 🧑‍💻 Quick Start

### Basic LLVM IR Functions
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

# Forward-mode AD (JVP)
jvp_add = pyir.jvp(add)
primal, jvp_val = jvp_add(2, 3, tangents=[1, 0])
print(f"add(2,3)={primal}, JVP wrt a={jvp_val}")

# Higher-order AD
d2_add = pyir.higher_order_grad(add, order=2)
print(f"Second derivative: {d2_add(2, 3)}")
```

### NumPy Integration with Multiple Execution Policies
```python
import numpy as np
import pyir

@pyir.function
def muladd(a: pyir.float32, b: pyir.float32, c: pyir.float32) -> pyir.float32:
    tmp: float
    result: float
    pyir.inline("""
        %tmp = fmul float %a, %b
        %result = fadd float %tmp, %c
    """)
    return result

# Vectorized kernel (default, fastest)
elemwise = pyir.numpy_kernel(muladd, policy="vectorized")
x = np.arange(5, dtype=np.float32)
y = np.ones(5, dtype=np.float32)
z = np.full(5, 2.0, dtype=np.float32)
print(elemwise(x, y, z))  # [2. 3. 4. 5. 6.]

# Parallel execution
parallel_kernel = pyir.numpy_kernel(muladd, policy="parallel", num_threads=4)
result = parallel_kernel(x, y, z)

# Async execution
async_kernel = pyir.numpy_kernel(muladd, policy="async")
import asyncio
future = async_kernel(x, y, z)
result = await future

# Sandboxed execution (secure)
sandboxed_kernel = pyir.numpy_kernel(muladd, policy="sandboxed")
result = sandboxed_kernel(x, y, z)
```

### Complex Numbers and Advanced Types
```python
import pyir

# Complex number operations
@pyir.function
def add_complex(a: pyir.complex64, b: pyir.complex64) -> pyir.complex64:
    result: pyir.complex64
    pyir.inline("""
        %result = insertvalue {float, float} %a, %b, 0
    """)
    return result

# Vector types for SIMD
@pyir.function
def vec_add(a: pyir.vec4f, b: pyir.vec4f) -> pyir.vec4f:
    result: pyir.vec4f
    pyir.inline("""
        %result = fadd <4 x float> %a, %b
    """)
    return result

# Struct types
@pyir.function
def point_add(a: pyir.struct([pyir.float32, pyir.float32]), 
              b: pyir.struct([pyir.float32, pyir.float32])) -> pyir.struct([pyir.float32, pyir.float32]):
    result: pyir.struct([pyir.float32, pyir.float32])
    pyir.inline("""
        %x1 = extractvalue {float, float} %a, 0
        %y1 = extractvalue {float, float} %a, 1
        %x2 = extractvalue {float, float} %b, 0
        %y2 = extractvalue {float, float} %b, 1
        %sum_x = fadd float %x1, %x2
        %sum_y = fadd float %y1, %y2
        %result = insertvalue {float, float} undef, float %sum_x, 0
        %result = insertvalue {float, float} %result, float %sum_y, 1
    """)
    return result
```

### Automatic Differentiation
```python
import pyir

@pyir.function
def sigmoid(x: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %neg_x = fsub float 0.0, %x
        %exp_neg = call float @llvm.exp.f32(float %neg_x)
        %denom = fadd float 1.0, %exp_neg
        %result = fdiv float 1.0, %denom
    """)
    return result

# Reverse-mode AD
grad_sigmoid = pyir.grad(sigmoid)
dx = grad_sigmoid(0.5)
print(f"∂sigmoid/∂x at x=0.5: {dx}")

# Forward-mode AD (JVP)
jvp_sigmoid = pyir.jvp(sigmoid)
primal, jvp_val = jvp_sigmoid(0.5, tangents=[1.0])
print(f"sigmoid(0.5)={primal}, JVP={jvp_val}")

# Higher-order derivatives
d2_sigmoid = pyir.higher_order_grad(sigmoid, order=2)
d2x = d2_sigmoid(0.5)
print(f"∂²sigmoid/∂x² at x=0.5: {d2x}")
```

### Kernel Fusion
```python
import pyir
import numpy as np

@pyir.function
def square(x: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %result = fmul float %x, %x
    """)
    return result

@pyir.function
def cube(x: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %result = fmul float %x, %x
        %result = fmul float %result, %x
    """)
    return result

# Fuse multiple kernels
fused = pyir.fuse_kernels([square, cube], name="power_fused")
result = fused(2.0)  # Returns (4.0, 8.0)

# Fuse with named outputs
fused_named = pyir.fuse_kernels(
    [square, cube], 
    name="power_named",
    output_names=["squared", "cubed"],
    return_type="dict"
)
result = fused_named(3.0)  # Returns {"squared": 9.0, "cubed": 27.0}

# Fuse NumPy kernels
square_np = pyir.numpy_kernel(square)
cube_np = pyir.numpy_kernel(cube)
fused_np = pyir.fuse_kernels([square_np, cube_np])

x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
result = fused_np(x)  # Returns (squared_array, cubed_array)
```

### SIMD Vectorization
```python
import pyir

# Auto-vectorize scalar kernels
@pyir.function
def scalar_add(a: pyir.float32, b: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %result = fadd float %a, %b
    """)
    return result

# Create SIMD version
simd_add = pyir.autovectorize_kernel(scalar_add, width=4, dtype=pyir.float32)

# Or use the decorator directly
@pyir.simd_kernel(width=4, dtype=pyir.float32)
def vadd4(a: pyir.vec4f, b: pyir.vec4f) -> pyir.vec4f:
    result: pyir.vec4f
    pyir.inline("""
        %result = fadd <4 x float> %a, %b
    """)
    return result

# Auto-vectorize multiple kernels and fuse them
kernels = [scalar_add, square, cube]
fused_simd = pyir.autovectorize_kernels(kernels, width=8, dtype=pyir.float32)
```

### Advanced Caching and AOT Compilation
```python
import pyir

@pyir.function
def expensive_kernel(x: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %result = call float @llvm.sin.f32(float %x)
        %result = fmul float %result, %result
    """)
    return result

# Prefetch kernel for faster first call
pyir.prefetch_kernel(expensive_kernel)

# AOT compile and cache to disk
pyir.aot_compile_kernel(expensive_kernel, target='cpu', fast_math=True)

# Load from disk cache
cached_kernel = pyir.load_aot_kernel(expensive_kernel)
```

### Real-World Example: Neural Network Layer
```python
import pyir
import numpy as np

@pyir.function
def linear_layer(x: pyir.float32, weight: pyir.float32, bias: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %weighted = fmul float %x, %weight
        %result = fadd float %weighted, %bias
    """)
    return result

@pyir.function
def relu(x: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %zero = fcmp ogt float %x, 0.0
        %result = select i1 %zero, float %x, float 0.0
    """)
    return result

# Create fused layer
layer = pyir.fuse_kernels([linear_layer, relu], name="linear_relu_layer")

# Create NumPy kernel for batch processing
batch_layer = pyir.numpy_kernel(layer, policy="vectorized")

# Use in neural network
x = np.random.randn(1000, 64).astype(np.float32)
weights = np.random.randn(64, 32).astype(np.float32)
biases = np.random.randn(32).astype(np.float32)

# Process batch
output = batch_layer(x, weights, biases)

# Get gradients for training
grad_layer = pyir.grad(layer)
dx, dw, db = grad_layer(x[0], weights[0], biases[0])
```

### Performance Benchmarking
```python
import pyir
import numpy as np
import time

@pyir.function
def fma(a: pyir.float32, b: pyir.float32, c: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %result = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
    """)
    return result

# Create vectorized kernel
fma_np = pyir.numpy_kernel(fma, policy="vectorized")

# Benchmark
x = np.random.randn(1000000).astype(np.float32)
y = np.random.randn(1000000).astype(np.float32)
z = np.random.randn(1000000).astype(np.float32)

# Warm up
_ = fma_np(x, y, z)

# Time the kernel
start = time.perf_counter()
result = fma_np(x, y, z)
end = time.perf_counter()

print(f"PyIR FMA: {(end - start) * 1000:.2f} ms")
print(f"Throughput: {len(x) / (end - start) / 1e6:.1f} M elements/sec")
```

---

## 🔧 Advanced Features

### Custom Gradients and Shape Inference
```python
import pyir

# Register custom gradient for a new operation
def custom_grad_fn(lhs, rhs, grads, assigns):
    # Custom gradient computation
    pass

pyir.register_custom_gradient("my_op", custom_grad_fn)

# Register custom shape inference
def custom_shape_fn(lhs, rhs, shapes, arg_shapes):
    # Custom shape inference logic
    return output_shape

pyir.register_custom_shape_inference("my_op", custom_shape_fn)
```

### Kernel Introspection and Debugging
```python
import pyir

@pyir.function
def debug_kernel(x: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %result = fmul float %x, %x
    """)
    return result

# Get kernel metadata
metadata = pyir.get_kernel_metadata(debug_kernel)
print(f"Kernel: {metadata['name']}")
print(f"Has IR: {metadata['has_ir']}")
print(f"Parameters: {metadata['parameters']}")

# Get IR objects for inspection
ir_module, output_vars, output_names = pyir.get_kernel_ir_objects(debug_kernel)
print(f"IR Module:\n{ir_module}")
```

---

## 📚 Documentation & Links
- [API Reference](#) (coming soon)
- [Report Issues](https://github.com/bumbelbee777/pyir/issues)
- [Contributing](#) (coming soon)

---

## 🛠️ License
MIT