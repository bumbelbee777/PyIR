from .core import fuse_kernels

class FusableKernel:
    def __init__(self, fn):
        self.fn = fn
    
    def __add__(self, other):
        if isinstance(other, FusableKernel):
            return FusableKernel(fuse_kernels([self.fn, other.fn]))
        return NotImplemented
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

def as_fusable(fn):
    """Wrap a kernel to support + fusion operator."""
    return FusableKernel(fn)