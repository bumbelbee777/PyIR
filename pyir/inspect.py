import inspect as _inspect
from .core import named_types

# These should be set by the main module
_function_registry = None
_global_consts = None
_get_module_ir = None

def _setup_inspect(function_registry, global_consts, get_module_ir):
    global _function_registry, _global_consts, _get_module_ir
    _function_registry = function_registry
    _global_consts = global_consts
    _get_module_ir = get_module_ir

def inspect_function(fn):
    """Print the generated IR for a @pyir.function-decorated function."""
    name = fn.__name__
    for k in _function_registry:
        if k.startswith(name + "__"):
            print(_function_registry[k])
            return _function_registry[k]
    print(f"[pyir.inspect_function] No IR found for function '{name}'.")
    return None

def list_functions():
    """List all registered function names in the current module."""
    return list(_function_registry.keys())

def get_function_ir(name):
    """Get the IR for a specific registered function by name (mangled)."""
    return _function_registry.get(name, None)

def inspect_globals():
    """Print all registered global constants and named types."""
    print("Globals:")
    for name, (llvm_type, value) in _global_consts.items():
        print(f"  @{name} = constant {llvm_type} {value}")
    print("Named types:")
    for name, typ in named_types.items():
        print(f"  %{name} = type {typ.llvm}")

def pretty_print_ir(ir):
    """Pretty-print IR with line numbers."""
    for i, line in enumerate(ir.splitlines(), 1):
        print(f"{i:4}: {line}")

def get_function_signature(fn):
    """Return the argument and return types for a @pyir.function-decorated function."""
    sig = _inspect.signature(fn)
    args = [(name, param.annotation) for name, param in sig.parameters.items()]
    ret = sig.return_annotation
    return {'args': args, 'return': ret}

def get_dependencies(fn):
    """List called functions in the IR for a given function (by scanning for 'call' instructions)."""
    name = fn.__name__
    for k in _function_registry:
        if k.startswith(name + "__"):
            ir = _function_registry[k]
            calls = []
            for line in ir.splitlines():
                if 'call ' in line:
                    parts = line.split('call ')[1].split('(')[0].strip().split()
                    if parts:
                        calls.append(parts[-1])
            return calls
    return []

def save_ir(filename, ir=None):
    """Save IR to a file. If ir is None, saves the whole module IR."""
    if ir is None:
        ir = _get_module_ir()
    with open(filename, 'w') as f:
        f.write(ir)
    print(f"[pyir.save_ir] IR saved to {filename}")

def visualize_ir(ir, filename='ir_graph.dot'):
    """
    Output a Graphviz dot file for the IR's call graph (very basic, experimental).
    Usage: pyir.visualize_ir(pyir.get_module_ir(), 'module.dot')
    """
    import re
    nodes = set()
    edges = set()
    for line in ir.splitlines():
        if line.strip().startswith('define'):
            fn = line.split('@')[1].split('(')[0]
            nodes.add(fn)
            current = fn
        if 'call ' in line:
            callee = line.split('call ')[1].split('(')[0].strip().split()[-1]
            edges.add((current, callee))
    with open(filename, 'w') as f:
        f.write('digraph IR {\n')
        for n in nodes:
            f.write(f'  "{n}";\n')
        for a, b in edges:
            f.write(f'  "{a}" -> "{b}";\n')
        f.write('}\n')
    print(f"[pyir.visualize_ir] IR graph saved to {filename}") 