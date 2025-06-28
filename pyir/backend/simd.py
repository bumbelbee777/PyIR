"""
pyir.simd: SIMD-friendly IR helpers and explicit SIMD kernel support for PyIR
"""
import functools
import re
from ..core import ssa, IRInstr, IRBlock, IRFunction, IRModule
from ..typing import vec, float32, float64, int32, int64


def vadd(a, b, out=None, type='"<4 x float>"'):
    out = out or ssa('vadd')
    return f"{out} = fadd {type} {a}, {b}"

def vsub(a, b, out=None, type='"<4 x float>"'):
    out = out or ssa('vsub')
    return f"{out} = fsub {type} {a}, {b}"

def vmul(a, b, out=None, type='"<4 x float>"'):
    out = out or ssa('vmul')
    return f"{out} = fmul {type} {a}, {b}"

def vdiv(a, b, out=None, type='"<4 x float>"'):
    out = out or ssa('vdiv')
    return f"{out} = fdiv {type} {a}, {b}"

def vload(ptr, out=None, type='"<4 x float>"'):
    out = out or ssa('vload')
    return f"{out} = load {type}, {type}* {ptr}"

def vstore(val, ptr, type='"<4 x float>"'):
    return f"store {type} {val}, {type}* {ptr}"

def vfma(a, b, c, out=None, type='"<4 x float>"'):
    """Fused multiply-add for SIMD vectors."""
    out = out or ssa('vfma')
    return f"{out} = call {type} @llvm.fma.{type[1:-1]}({type} {a}, {type} {b}, {type} {c})"

def vmax(a, b, out=None, type='"<4 x float>"'):
    """Elementwise maximum for SIMD vectors."""
    out = out or ssa('vmax')
    return f"{out} = call {type} @llvm.maxnum.{type[1:-1]}({type} {a}, {type} {b})"

def vmin(a, b, out=None, type='"<4 x float>"'):
    """Elementwise minimum for SIMD vectors."""
    out = out or ssa('vmin')
    return f"{out} = call {type} @llvm.minnum.{type[1:-1]}({type} {a}, {type} {b})"

def vabs(a, out=None, type='"<4 x float>"'):
    """Elementwise absolute value for SIMD vectors."""
    out = out or ssa('vabs')
    return f"{out} = call {type} @llvm.fabs.{type[1:-1]}({type} {a})"

def vsqrt(a, out=None, type='"<4 x float>"'):
    """Elementwise square root for SIMD vectors."""
    out = out or ssa('vsqrt')
    return f"{out} = call {type} @llvm.sqrt.{type[1:-1]}({type} {a})"

# SIMD kernel decorator with enhanced IR object support

def simd_kernel(width=4, dtype=float32):
    """
    Decorator to mark a function as a SIMD kernel (vectorized, e.g., 4-wide float32).
    Usage:
        @pyir.simd_kernel(width=4, dtype=pyir.float32)
        def vadd4(a, b): ...
    """
    def decorator(fn):
        simd_width = width
        simd_dtype = dtype
        # Auto-detect SIMD width and dtype from argument annotations if not specified
        if simd_width is None or simd_dtype is None:
            import inspect
            sig = inspect.signature(fn)
            for param in sig.parameters.values():
                if hasattr(param.annotation, '__name__') and 'vec' in param.annotation.__name__:
                    # Extract width from vec type annotation
                    if simd_width is None:
                        type_name = param.annotation.__name__
                        if type_name.startswith('vec'):
                            try:
                                simd_width = int(type_name[3:-1])
                            except ValueError:
                                pass
                    if simd_dtype is None:
                        type_name = param.annotation.__name__
                        if type_name.endswith('f'):
                            simd_dtype = float32
                        elif type_name.endswith('d'):
                            simd_dtype = float64
                        elif type_name.endswith('i'):
                            simd_dtype = int32
                        elif type_name.endswith('l'):
                            simd_dtype = int64
        if simd_width is None:
            simd_width = 4
        if simd_dtype is None:
            simd_dtype = float32
        fn._is_simd_kernel = True
        fn._simd_width = simd_width
        fn._simd_dtype = simd_dtype
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        # Propagate SIMD attributes to the wrapper
        wrapper._is_simd_kernel = True
        wrapper._simd_width = simd_width
        wrapper._simd_dtype = simd_dtype
        wrapper.is_simd = True
        return wrapper
    return decorator

def autovectorize_kernel(fn, width=4, dtype=float32):
    """
    Given a scalar kernel, generate a SIMD version using @simd_kernel.
    Returns the SIMD kernel.
    """
    return simd_kernel(width=width, dtype=dtype)(fn)

def autovectorize_numpy_kernel(fn, width=4, dtype=float32):
    """
    Given a scalar NumPy kernel, generate a SIMD version using @simd_kernel and wrap as a NumPy kernel.
    Returns the SIMD NumPy kernel.
    """
    from ..interop.numpy import numpy_kernel
    simd_fn = simd_kernel(width=width, dtype=dtype)(fn)
    return numpy_kernel(simd_fn)

def create_simd_ir_function(name, args, ret_type, width=4, dtype=float32):
    """
    Create a SIMD IR function with proper vector types and attributes.
    Returns IRFunction object optimized for SIMD.
    """
    # Convert scalar types to vector types
    simd_args = []
    for arg_name, arg_type in args:
        if arg_type in ['float', 'double']:
            simd_type = f'<{width} x {arg_type}>'
        elif arg_type in ['i32', 'i64']:
            simd_type = f'<{width} x {arg_type}>'
        else:
            simd_type = arg_type  # Keep as-is for pointers, etc.
        simd_args.append((arg_name, simd_type))
    
    # Convert return type to vector if it's a scalar
    if ret_type in ['float', 'double']:
        simd_ret_type = f'<{width} x {ret_type}>'
    elif ret_type in ['i32', 'i64']:
        simd_ret_type = f'<{width} x {ret_type}>'
    else:
        simd_ret_type = ret_type
    
    # Add SIMD-specific attributes
    attrs = "fast"
    
    return IRFunction(name, simd_args, simd_ret_type, attrs)

def vectorize_ir_block(block, width=4, dtype=float32):
    """
    Convert a scalar IR block to SIMD by vectorizing instructions.
    Returns a new IRBlock with vectorized instructions.
    """
    simd_block = IRBlock(block.label)
    
    for instr in block.instrs:
        instr_str = str(instr)
        
        # Vectorize arithmetic operations
        if 'fadd' in instr_str:
            # Convert scalar fadd to vector fadd
            instr_str = re.sub(r'fadd\s+(\w+)\s+', rf'fadd <{width} x \1> ', instr_str)
        elif 'fsub' in instr_str:
            instr_str = re.sub(r'fsub\s+(\w+)\s+', rf'fsub <{width} x \1> ', instr_str)
        elif 'fmul' in instr_str:
            instr_str = re.sub(r'fmul\s+(\w+)\s+', rf'fmul <{width} x \1> ', instr_str)
        elif 'fdiv' in instr_str:
            instr_str = re.sub(r'fdiv\s+(\w+)\s+', rf'fdiv <{width} x \1> ', instr_str)
        elif 'add' in instr_str and 'fadd' not in instr_str:
            instr_str = re.sub(r'add\s+(\w+)\s+', rf'add <{width} x \1> ', instr_str)
        elif 'sub' in instr_str and 'fsub' not in instr_str:
            instr_str = re.sub(r'sub\s+(\w+)\s+', rf'sub <{width} x \1> ', instr_str)
        elif 'mul' in instr_str and 'fmul' not in instr_str:
            instr_str = re.sub(r'mul\s+(\w+)\s+', rf'mul <{width} x \1> ', instr_str)
        
        # Vectorize loads and stores
        elif 'load' in instr_str:
            instr_str = re.sub(r'load\s+(\w+),\s+(\w+)\*', rf'load <{width} x \1>, <{width} x \1>*', instr_str)
        elif 'store' in instr_str:
            instr_str = re.sub(r'store\s+(\w+)\s+', rf'store <{width} x \1> ', instr_str)
        
        simd_block.add(IRInstr(instr_str))
    
    return simd_block

def detect_simd_capabilities():
    """
    Auto-detect SIMD capabilities of the current system.
    Returns dict with available SIMD features and optimal widths.
    """
    import platform
    import subprocess
    
    capabilities = {
        'x86_64': {
            'sse': 4,
            'sse2': 4,
            'sse3': 4,
            'ssse3': 4,
            'sse4_1': 4,
            'sse4_2': 4,
            'avx': 8,
            'avx2': 8,
            'avx512f': 16
        },
        'aarch64': {
            'neon': 4,
            'sve': 8  # Variable length, but 8 is common
        }
    }
    
    arch = platform.machine()
    detected = {}
    
    if arch == 'x86_64':
        try:
            # Try to detect CPU features (simplified)
            import os
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    for feature, width in capabilities[arch].items():
                        if feature.upper() in cpuinfo:
                            detected[feature] = width
        except:
            # Fallback to common features
            detected = {'sse2': 4, 'avx': 8}
    elif arch == 'aarch64':
        detected = {'neon': 4}
    
    return detected

def get_optimal_simd_width(dtype=float32):
    """
    Get the optimal SIMD width for a given data type based on system capabilities.
    """
    capabilities = detect_simd_capabilities()
    
    if not capabilities:
        return 4  # Default fallback
    
    # Find the highest available width
    max_width = max(capabilities.values())
    
    # Adjust based on data type size
    if dtype == float64:
        # Double precision typically uses half the vector width
        return max_width // 2
    else:
        return max_width

# Example usage:
# simd_add = pyir.autovectorize_kernel(add)
# simd_elemwise = pyir.autovectorize_numpy_kernel(muladd)
# optimal_width = pyir.get_optimal_simd_width(pyir.float32)
