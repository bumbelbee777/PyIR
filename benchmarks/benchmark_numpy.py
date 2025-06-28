import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pyir
import numba
from matplotlib.animation import FuncAnimation

# PyIR elementwise add with no optimizations
@pyir.function(no_optims=True)
def add_pyir_no_optims(a: pyir.float32, b: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %result = fadd float %a, %b
    """)
    return result

# Ensure IR is generated and registered
add_pyir_no_optims(1.0, 2.0)
add_pyir_np_no_optims = pyir.numpy_kernel(add_pyir_no_optims, no_optims=True)

# PyIR elementwise add
@pyir.function
def add_pyir(a: pyir.float32, b: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %result = fadd float %a, %b
    """)
    return result

# Ensure IR is generated and registered
add_pyir(1.0, 2.0)
add_pyir_np = pyir.numpy_kernel(add_pyir)

# PyIR elementwise multiply
@pyir.function
def mul_pyir(a: pyir.float32, b: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %result = fmul float %a, %b
    """)
    return result

# Ensure IR is generated and registered
mul_pyir(1.0, 2.0)
mul_pyir_np = pyir.numpy_kernel(mul_pyir)

# PyIR fused multiply-add
@pyir.function
def fma_pyir(a: pyir.float32, b: pyir.float32, c: pyir.float32) -> pyir.float32:
    tmp: float
    result: float
    pyir.inline("""
        %tmp = fmul float %a, %b
        %result = fadd float %tmp, %c
    """)
    return result

# Ensure IR is generated and registered
fma_pyir(1.0, 2.0, 3.0)
fma_pyir_np = pyir.numpy_kernel(fma_pyir)

# PyIR sum reduction
@pyir.function
def sum_pyir(a: pyir.float32, b: pyir.float32) -> pyir.float32:
    result: float
    pyir.inline("""
        %result = fadd float %a, %b
    """)
    return result

# Ensure IR is generated and registered
sum_pyir(1.0, 2.0)
sum_pyir_np = lambda arr: np.sum(arr)  # Use numpy for reduction for now

# PyIR vectorized elementwise add (native loop kernel)
add_pyir_vec = pyir.vectorized_kernel(add_pyir)

# PyIR vectorized elementwise multiply
mul_pyir_vec = pyir.vectorized_kernel(mul_pyir)

# PyIR vectorized FMA
fma_pyir_vec = pyir.vectorized_kernel(fma_pyir)

# PyIR fused vectorized add+mul+fma
fused_vec = pyir.vectorized_fuse_kernels([add_pyir, mul_pyir, fma_pyir], name="fused_add_mul_fma_vec")

# Numba elementwise add
@numba.njit
def add_numba(a, b):
    return a + b

# Numba elementwise multiply
@numba.njit
def mul_numba(a, b):
    return a * b

# Numba fused multiply-add
@numba.njit
def fma_numba(a, b, c):
    return a * b + c

# Numba sum reduction
@numba.njit
def sum_numba(arr):
    s = 0.0
    for i in range(arr.size):
        s += arr[i]
    return s


def bench_kernel(fn, *args, repeat=5, number=3):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(number):
            fn(*args)
        end = time.perf_counter()
        times.append((end - start) / number)
    return min(times)

sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
results = []

for size in sizes:
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    c = np.random.rand(size).astype(np.float32)
    # Warmup
    add_pyir_np(a, b)
    add_pyir_vec(a, b)
    add_numba(a, b)
    mul_pyir_np(a, b)
    mul_pyir_vec(a, b)
    mul_numba(a, b)
    fma_pyir_np(a, b, c)
    fma_pyir_vec(a, b, c)
    fma_numba(a, b, c)
    fused_vec(a, b, c)
    sum_pyir_np(a)
    sum_numba(a)
    # Timings
    t_add_pyir = bench_kernel(add_pyir_np, a, b)
    t_add_pyir_vec = bench_kernel(add_pyir_vec, a, b)
    t_add_numba = bench_kernel(add_numba, a, b)
    t_mul_pyir = bench_kernel(mul_pyir_np, a, b)
    t_mul_pyir_vec = bench_kernel(mul_pyir_vec, a, b)
    t_mul_numba = bench_kernel(mul_numba, a, b)
    t_fma_pyir = bench_kernel(fma_pyir_np, a, b, c)
    t_fma_pyir_vec = bench_kernel(fma_pyir_vec, a, b, c)
    t_fma_numba = bench_kernel(fma_numba, a, b, c)
    t_fused_vec = bench_kernel(fused_vec, a, b, c)
    t_sum_pyir = bench_kernel(sum_pyir_np, a)
    t_sum_numba = bench_kernel(sum_numba, a)
    results.append({
        'size': size,
        'add_pyir': t_add_pyir,
        'add_pyir_vec': t_add_pyir_vec,
        'add_numba': t_add_numba,
        'mul_pyir': t_mul_pyir,
        'mul_pyir_vec': t_mul_pyir_vec,
        'mul_numba': t_mul_numba,
        'fma_pyir': t_fma_pyir,
        'fma_pyir_vec': t_fma_pyir_vec,
        'fma_numba': t_fma_numba,
        'fused_vec': t_fused_vec,
        'sum_pyir': t_sum_pyir,
        'sum_numba': t_sum_numba,
    })
    print(f"[size={size}] add_pyir={t_add_pyir:.6f}s, add_pyir_vec={t_add_pyir_vec:.6f}s, add_numba={t_add_numba:.6f}s, mul_pyir={t_mul_pyir:.6f}s, mul_pyir_vec={t_mul_pyir_vec:.6f}s, mul_numba={t_mul_numba:.6f}s, fma_pyir={t_fma_pyir:.6f}s, fma_pyir_vec={t_fma_pyir_vec:.6f}s, fma_numba={t_fma_numba:.6f}s, fused_vec={t_fused_vec:.6f}s, sum_pyir={t_sum_pyir:.6f}s, sum_numba={t_sum_numba:.6f}s")

df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
plt.title("PyIR vs Numba: Elementwise Add")
plt.plot(df['size'], df['add_pyir'], label='PyIR (Python loop)', marker='o')
plt.plot(df['size'], df['add_pyir_vec'], label='PyIR (native loop)', marker='o')
plt.plot(df['size'], df['add_numba'], label='Numba', marker='o')
plt.xlabel('Array Size')
plt.ylabel('Time (s)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig('benchmarks/pyir_vs_numba_add.png')

plt.figure(figsize=(10, 6))
plt.title("PyIR vs Numba: Elementwise Multiply")
plt.plot(df['size'], df['mul_pyir'], label='PyIR (Python loop)', marker='o')
plt.plot(df['size'], df['mul_pyir_vec'], label='PyIR (native loop)', marker='o')
plt.plot(df['size'], df['mul_numba'], label='Numba', marker='o')
plt.xlabel('Array Size')
plt.ylabel('Time (s)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig('benchmarks/pyir_vs_numba_mul.png')

plt.figure(figsize=(10, 6))
plt.title("PyIR vs Numba: Fused Multiply-Add")
plt.plot(df['size'], df['fma_pyir'], label='PyIR (Python loop)', marker='o')
plt.plot(df['size'], df['fma_pyir_vec'], label='PyIR (native loop)', marker='o')
plt.plot(df['size'], df['fma_numba'], label='Numba', marker='o')
plt.xlabel('Array Size')
plt.ylabel('Time (s)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig('benchmarks/pyir_vs_numba_fma.png')

plt.figure(figsize=(10, 6))
plt.title("PyIR vs Numba: Sum Reduction")
plt.plot(df['size'], df['sum_pyir'], label='PyIR', marker='o')
plt.plot(df['size'], df['sum_numba'], label='Numba', marker='o')
plt.xlabel('Array Size')
plt.ylabel('Time (s)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig('benchmarks/pyir_vs_numba_sum.png')

plt.figure(figsize=(10, 6))
plt.title("PyIR: Fused Vectorized Kernel (add, mul, fma)")
plt.plot(df['size'], df['fused_vec'], label='PyIR (fused native loop)', marker='o')
plt.xlabel('Array Size')
plt.ylabel('Time (s)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig('benchmarks/pyir_fused_vec.png')

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title("PyIR vs Numba: Add Kernel Scaling")
ax.set_xlabel('Array Size')
ax.set_ylabel('Time (s)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', ls='--')
line1, = ax.plot([], [], 'o-', label='PyIR (Python loop)')
line2, = ax.plot([], [], 'o-', label='Numba')
line3, = ax.plot([], [], 'o-', label='PyIR (native loop)')
ax.legend()

def animate(i):
    line1.set_data(df['size'][:i+1], df['add_pyir'][:i+1])
    line2.set_data(df['size'][:i+1], df['add_numba'][:i+1])
    line3.set_data(df['size'][:i+1], df['add_pyir_vec'][:i+1])
    return line1, line2, line3

ani = FuncAnimation(fig, animate, frames=len(df), interval=800, blit=True)
ani.save('benchmarks/pyir_vs_numba_add.gif', writer='pillow')

print("Benchmarking complete. Charts saved in benchmarks/.")
