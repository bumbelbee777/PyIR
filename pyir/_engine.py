import llvmlite.binding as llvm

# Enable debug prints for unit tests
pyir_debug = True

# Initialize LLVM (once)
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

# Create target machine and execution engine
_target = llvm.Target.from_default_triple()
_target_machine = _target.create_target_machine()
_backing_mod = llvm.parse_assembly("")  # empty module placeholder
_engine = llvm.create_mcjit_compiler(_backing_mod, _target_machine)

def add_module(mod):
    return _engine.add_module(mod)

def finalize_object():
    return _engine.finalize_object()

def run_static_constructors():
    return _engine.run_static_constructors()

def get_function_address(name):
    return _engine.get_function_address(name) 