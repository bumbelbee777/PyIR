import llvmlite.binding as llvm
from .._engine import _engine
import threading
import hashlib

_module_cache = {}
_module_cache_lock = threading.Lock()

def jit_compile_ir(ir_code: str, fn_name: str = None):
    """
    Compile and add LLVM IR code to the JIT engine. No sandboxing or security logic.
    Automatically adds needed LLVM intrinsic declarations if used but not declared.
    """
    needed_intrinsics = [
        {'pattern': '@llvm.ctpop.i64', 'declare': 'declare i64 @llvm.ctpop.i64(i64)'},
        {'pattern': '@llvm.sqrt.f32', 'declare': 'declare float @llvm.sqrt.f32(float)'},
        {'pattern': '@llvm.sqrt.f64', 'declare': 'declare double @llvm.sqrt.f64(double)'},
        {'pattern': '@llvm.sin.f32', 'declare': 'declare float @llvm.sin.f32(float)'},
        {'pattern': '@llvm.sin.f64', 'declare': 'declare double @llvm.sin.f64(double)'},
        {'pattern': '@llvm.cos.f32', 'declare': 'declare float @llvm.cos.f32(float)'},
        {'pattern': '@llvm.cos.f64', 'declare': 'declare double @llvm.cos.f64(double)'},
        {'pattern': '@llvm.exp.f32', 'declare': 'declare float @llvm.exp.f32(float)'},
        {'pattern': '@llvm.exp.f64', 'declare': 'declare double @llvm.exp.f64(double)'},
        {'pattern': '@llvm.log.f32', 'declare': 'declare float @llvm.log.f32(float)'},
        {'pattern': '@llvm.log.f64', 'declare': 'declare double @llvm.log.f64(double)'},
        {'pattern': '@llvm.pow.f32', 'declare': 'declare float @llvm.pow.f32(float, float)'},
        {'pattern': '@llvm.pow.f64', 'declare': 'declare double @llvm.pow.f64(double, double)'},
        {'pattern': '@llvm.fabs.f32', 'declare': 'declare float @llvm.fabs.f32(float)'},
        {'pattern': '@llvm.fabs.f64', 'declare': 'declare double @llvm.fabs.f64(double)'},
        {'pattern': '@llvm.fma.f32', 'declare': 'declare float @llvm.fma.f32(float, float, float)'},
        {'pattern': '@llvm.fma.f64', 'declare': 'declare double @llvm.fma.f64(double, double, double)'},
        # Add more as needed
    ]
    for info in needed_intrinsics:
        if info['pattern'] in ir_code and info['declare'] not in ir_code:
            ir_code = info['declare'] + '\n' + ir_code
    ir_hash = hashlib.sha256(ir_code.encode()).hexdigest()
    
    with _module_cache_lock:
        if ir_hash in _module_cache:
            cached_module = _module_cache[ir_hash]
            # Add the cached module to the engine
            _engine.add_module(cached_module)
            return cached_module
    
    # Validate IR
    try:
        mod = llvm.parse_assembly(ir_code)
        mod.verify()
    except Exception as e:
        raise ValueError(f"[pyir.jit_compile_ir] Error validating IR:\n{e}\n--- IR ---\n{ir_code}")

    with _module_cache_lock:
        _module_cache[ir_hash] = mod
    
    # Add to engine
    _engine.add_module(mod)
    _engine.finalize_object()
    _engine.run_static_constructors()
    return mod

_compile_ir = jit_compile_ir
