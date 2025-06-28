import llvmlite.binding as llvm
from .._engine import _engine
import threading
import hashlib

_module_cache = {}
_module_cache_lock = threading.Lock()

def jit_compile_ir(ir_code: str, fn_name: str = None):
    """
    Compile and add LLVM IR code to the JIT engine. No sandboxing or security logic.
    """
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
