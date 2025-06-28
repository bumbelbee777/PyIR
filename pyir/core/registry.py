_function_registry = {}

def register_function(name: str, ir: str):
    from .._engine import pyir_debug
    _function_registry[name] = ir
    if pyir_debug:
        print(f"[pyir.function] Registered function: {name}")
        print(f"[pyir.function] Registry keys: {list(_function_registry.keys())}") 