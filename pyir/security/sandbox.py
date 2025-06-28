"""
pyir.security.sandbox: Sandboxed JIT and related utilities
"""

sandbox_mode: bool = False

def sandboxed_jit(ir_code, fn_name, args=None, ret_ctype=None):
    if not sandbox_mode:
        pass
    from .._engine import _engine
    import llvmlite.binding as llvm
    import multiprocessing, pickle, ctypes
    def _compile_ir(ir_code, fn_name, args=None, ret_ctype=None):
        try:
            from ..core.ir import validate_ir
            validate_ir(ir_code)
        except Exception as e:
            raise ValueError(f"[pyir._compile_ir] Error validating IR:\n{e}\n")
        mod = llvm.parse_assembly(ir_code)
        mod.verify()
        _engine.add_module(mod)
        _engine.finalize_object()
        _engine.run_static_constructors()
        return mod
    return _compile_ir(ir_code, fn_name, args=args, ret_ctype=ret_ctype)
