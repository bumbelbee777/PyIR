from enum import Enum
from typing import Callable, Any, Union

class ExecutionPolicy(Enum):
    """
    Built-in execution policies for PyIR kernels.
    """
    SERIAL = "serial"         # Python loop (reference, slow)
    VECTORIZED = "vectorized" # Native LLVM IR loop (default)
    PARALLEL = "parallel"     # Multithreaded vectorized execution
    ASYNC = "async"           # Async execution (coroutine/future)
    CUDA = "cuda"             # CUDA device execution (future)
    SANDBOXED = "sandboxed"   # Sandboxed subprocess execution (can wrap another policy)
    # Add more as needed

class SandboxedPolicy:
    """
    Represents a sandboxed execution policy with a backend policy.
    Example: SandboxedPolicy(ExecutionPolicy.VECTORIZED)
    """
    def __init__(self, backend: Any = ExecutionPolicy.VECTORIZED):
        self.backend = validate_policy(backend)
    def __eq__(self, other):
        return isinstance(other, SandboxedPolicy) and self.backend == other.backend
    def __repr__(self):
        return f"SandboxedPolicy({self.backend!r})"

# Type for a policy: either a built-in enum, a SandboxedPolicy, or a user-supplied callable
PolicyType = Union[ExecutionPolicy, SandboxedPolicy, Callable[..., Any]]

def validate_policy(policy: Any) -> PolicyType:
    """
    Validate and normalize a policy argument.
    Accepts ExecutionPolicy, SandboxedPolicy, string, or callable.
    For sandboxed, accepts e.g. 'sandboxed:vectorized' or SandboxedPolicy(...).
    """
    if isinstance(policy, SandboxedPolicy):
        return SandboxedPolicy(policy.backend)
    if isinstance(policy, ExecutionPolicy):
        return policy
    if isinstance(policy, str):
        if policy.lower().startswith("sandboxed"):
            # Parse backend, e.g. 'sandboxed:vectorized'
            parts = policy.split(":", 1)
            backend = parts[1] if len(parts) > 1 else "vectorized"
            return SandboxedPolicy(backend)
        try:
            return ExecutionPolicy(policy.lower())
        except ValueError:
            raise ValueError(f"Unknown execution policy: {policy}")
    if callable(policy):
        return policy
    raise TypeError(f"Invalid execution policy: {policy}")

"""
pyir.security.policy: Security policies and safe mode for PyIR
"""

# Global safe mode flag for PyIR

__all__ = ["ExecutionPolicy", "SandboxedPolicy"]