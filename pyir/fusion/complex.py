def is_complex_kernel(fn):
    """Detect if a kernel is complex-valued."""
    import inspect
    from pyir.typing import python_type_map
    sig = inspect.signature(fn)
    for n, p in sig.parameters.items():
        if hasattr(p.annotation, 'llvm') and ('{' in p.annotation.llvm and 'float' in p.annotation.llvm):
            return True
        elif p.annotation in (complex,):
            return True
    return False
