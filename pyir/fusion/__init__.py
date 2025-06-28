from .core import (
    set_function_registry, fuse_kernels, get_kernel_ir_objects, analyze_fusion_compatibility,
    get_kernel_metadata
)
from .vectorized import vectorized_fuse_kernels
from .ergonomic import FusableKernel, as_fusable
from .autovec import autovectorize_kernels
from pyir.core.function import _function_registry

__all__ = [
    "set_function_registry",
    "fuse_kernels",
    "get_kernel_ir_objects",
    "analyze_fusion_compatibility",
    "get_kernel_metadata",
    "vectorized_fuse_kernels",
    "FusableKernel",
    "as_fusable",
    "autovectorize_kernels",
    "_function_registry"
]
