import llvmlite.binding as llvm

# Initialize LLVM (once)
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

# Create target machine and execution engine
_target = llvm.Target.from_default_triple()
_target_machine = _target.create_target_machine()
_backing_mod = llvm.parse_assembly("")  # empty module placeholder
_engine = llvm.create_mcjit_compiler(_backing_mod, _target_machine) 