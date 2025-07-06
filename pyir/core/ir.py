"""
pyir.core.ir: IR, SSA, macros, and IR object model utilities
"""
import re
import inspect
import warnings
import multiprocessing
import pickle
import itertools
from typing import Dict, Any
import llvmlite.binding as llvm

from ..typing import *
from .._engine import pyir_debug
from ..security.safe_mode import safe_mode
from .jit import jit_compile_ir
from .registry import _function_registry, register_function

_ssa_counters = {}
_macro_registry = {}

_llvm_target_machine = None

# IR Templates for common operations (performance optimization)
_ir_templates = {
    'add': {
        'int32': "define i32 @{name}(i32 %a, i32 %b) {{\n  %result = add i32 %a, %b\n  ret i32 %result\n}}",
        'float': "define float @{name}(float %a, float %b) {{\n  %result = fadd {fast_math} float %a, %b\n  ret float %result\n}}",
        'float_fast': "define float @{name}(float %a, float %b) {{\n  %result = fadd fast float %a, %b\n  ret float %result\n}}"
    },
    'mul': {
        'int32': "define i32 @{name}(i32 %a, i32 %b) {{\n  %result = mul i32 %a, %b\n  ret i32 %result\n}}",
        'float': "define float @{name}(float %a, float %b) {{\n  %result = fmul {fast_math} float %a, %b\n  ret float %result\n}}",
        'float_fast': "define float @{name}(float %a, float %b) {{\n  %result = fmul fast float %a, %b\n  ret float %result\n}}"
    },
    'fma': {
        'float': "define float @{name}(float %a, float %b, float %c) {{\n  %tmp = fmul {fast_math} float %a, %b\n  %result = fadd {fast_math} float %tmp, %c\n  ret float %result\n}}",
        'float_fast': "define float @{name}(float %a, float %b, float %c) {{\n  %tmp = fmul fast float %a, %b\n  %result = fadd fast float %tmp, %c\n  ret float %result\n}}"
    }
}

class IRTemplate:
    """Fast IR template instantiation for common operations."""
    
    @classmethod
    def get_template(cls, operation: str, dtype: str, fast_math: bool = False) -> str:
        """Get a template for a common operation."""
        template_key = f"{dtype}_fast" if fast_math else dtype
        return _ir_templates.get(operation, {}).get(template_key)
    
    @classmethod
    def instantiate(cls, operation: str, name: str, dtype: str, fast_math: bool = False, **kwargs) -> str:
        """Instantiate a template with given parameters."""
        template = cls.get_template(operation, dtype, fast_math)
        if template is None:
            return None
        
        # Prepare template variables
        template_vars = {
            'name': name,
            'fast_math': 'fast' if fast_math else '',
            **kwargs
        }
        
        return template.format(**template_vars)
    
    @classmethod
    def is_supported(cls, operation: str, dtype: str) -> bool:
        """Check if a template exists for the given operation and dtype."""
        return operation in _ir_templates and dtype in _ir_templates[operation]

# Optimized IR generation using templates
def create_optimized_ir_function(name: str, operation: str, dtype: str, fast_math: bool = False, **kwargs) -> str:
    """Create IR using templates when possible, fallback to object model."""
    # Try template first
    ir_str = IRTemplate.instantiate(operation, name, dtype, fast_math, **kwargs)
    if ir_str is not None:
        return ir_str
    
    # Fallback to object model
    return _create_ir_function_object_model(name, operation, dtype, fast_math, **kwargs)

def _create_ir_function_object_model(name: str, operation: str, dtype: str, fast_math: bool = False, **kwargs) -> str:
    """Fallback IR generation using object model."""
    # This is the existing object model approach
    # Implementation would go here
    pass

def register_function(name: str, ir: str):
    """
    Register a function's IR in the global registry.
    """
    _function_registry[name] = ir
    if pyir_debug:
        print(f"[pyir.function] Registered function: {name}")
        print(f"[pyir.function] Registry keys: {list(_function_registry.keys())}")

python_type_map = {
    int: int32,
    float: float32,
    bool: IntType(1),
    complex: complex128,
    'int8': int8,
    'int16': int16,
    'int32': int32,
    'int64': int64,
    'float16': float16,
    'float32': float32,
    'float64': float64,
    'complex64': complex64,
    'complex128': complex128,
}

def infer_type_from_value(val):
    if isinstance(val, bool):
        return IntType(1)
    elif isinstance(val, int):
        if -(2**7) <= val < 2**7:
            return int8
        elif -(2**15) <= val < 2**15:
            return int16
        elif -(2**31) <= val < 2**31:
            return int32
        else:
            return int64
    elif isinstance(val, float):
        return float64 if abs(val) > 1e38 or (val != 0 and abs(val) < 1e-38) else float32
    elif isinstance(val, complex):
        # Use complex128 for Python complex literals by default
        return complex128
    # User-defined types: struct, vec, array, opaque, named_types
    elif hasattr(val, 'llvm') and hasattr(val, 'ctype'):
        return val
    elif isinstance(val, str) and val in named_types:
        return named_types[val]
    return int32

def ssa(base: str) -> str:
    count = _ssa_counters.get(base, 0) + 1
    _ssa_counters[base] = count
    return f"%{base}{count}"

def define_macro(name: str, template: str):
    _macro_registry[name] = template

def validate_ir(ir: str):
    import llvmlite.binding as llvm
    if ir is None or ir == "":
        raise ValueError(f"[pyir.validate_ir] Empty IR passed!")
    try:
        llvm.parse_assembly(ir)
    except Exception as e:
        raise ValueError(f"[pyir.validate_ir] Invalid IR:\n{e}\n--- IR ---\n{ir}")
    return ir

def inline(ir: str, sugar=False):
    """Process inline IR and handle variable scoping."""
    import inspect
    import re
    
    # Get the calling frame to set variables in the caller's scope
    frame = inspect.currentframe().f_back
    if frame:
        # Parse IR to find variable assignments
        lines = ir.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('%') and '=' in line:
                # Extract variable name from IR
                lhs = line.split('=')[0].strip()[1:]  # Remove '%' prefix
                # Set a default value for this variable in the caller's frame
                # This allows the Python function to return the variable
                frame.f_locals[lhs] = 0  # Default value that can be returned
    
    return IRInstr(ir)

def _compile_ir(ir_code, fn_name, args=None, ret_ctype=None):
    import llvmlite.binding as llvm
    from .._engine import _engine
    try:
        validate_ir(ir_code)
    except Exception as e:
        raise ValueError(f"[pyir._compile_ir] Error validating IR:\n{e}\n")
    mod = llvm.parse_assembly(ir_code)
    mod.verify()
    _engine.add_module(mod)
    _engine.finalize_object()
    _engine.run_static_constructors()
    return mod

def sandboxed_jit(ir_code, fn_name, args=None, ret_ctype=None):
    return _compile_ir(ir_code, fn_name, args=args, ret_ctype=ret_ctype)

class IRInstr:
    def __init__(self, text):
        self.text = text
        self._str_cache = None
    
    def __str__(self):
        if self._str_cache is None:
            self._str_cache = self.text
        return self._str_cache

class IRBlock:
    def __init__(self, label):
        self.label = label
        self.instrs = []
        self._str_cache = None
        self._dirty = True
    
    def add(self, instr):
        self.instrs.append(instr if isinstance(instr, IRInstr) else IRInstr(instr))
        self._dirty = True
    
    def __str__(self):
        if not self._dirty and self._str_cache is not None:
            return self._str_cache
        
        lines = [str(i) for i in self.instrs]
        if lines and lines[0].endswith(':'):
            result = '\n'.join(lines)
        else:
            result = f"{self.label}:\n  " + "\n  ".join(lines)
        
        self._str_cache = result
        self._dirty = False
        return result

class IRFunction:
    def __init__(self, name, args, ret_type, attrs="", fast_math=False):
        import re
        # --- PATCH: sanitize function name for LLVM compatibility ---
        self.name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        self.args = args
        self.ret_type = ret_type
        self.attrs = attrs
        self.blocks = []
        self.fast_math = fast_math
        self._str_cache = None
        self._dirty = True
    
    def add_block(self, block):
        self.blocks.append(block)
        self._dirty = True
    
    def __str__(self):
        if not self._dirty and self._str_cache is not None:
            return self._str_cache
        
        args_str = ", ".join(f"{t} %{n}" for n, t in self.args)
        body = []
        
        for b in self.blocks:
            lines = []
            for instr in b.instrs:
                s = str(instr)
                if self.fast_math:
                    if not hasattr(self, '_fast_math_pattern'):
                        import re
                        self._fast_math_pattern = re.compile(r'\b(fadd|fsub|fmul|fdiv|frem)\b')
                    s = self._fast_math_pattern.sub(r'\1 fast', s)
                lines.append(s)
            
            if lines and lines[0].endswith(':'):
                body.append("\n  ".join(lines))
            else:
                body.append(f"{b.label}:\n  " + "\n  ".join(lines))
        
        result = f"define {self.ret_type} @{self.name}({args_str}) {{\n" + "\n".join(body) + "\n}"
        self._str_cache = result
        self._dirty = False
        return result

class IRModule:
    def __init__(self):
        self.functions = []
        self.globals = []
        self._str_cache = None
        self._dirty = True
    
    def add_function(self, fn):
        self.functions.append(fn)
        self._dirty = True
    
    def add_global(self, g):
        self.globals.append(g)
        self._dirty = True
    
    def __str__(self):
        if not self._dirty and self._str_cache is not None:
            return self._str_cache
        
        # Emit struct type declarations at the top if present (deduplicated)
        struct_decls = set()
        for fn in self.functions:
            if hasattr(fn, '_struct_type_decl'):
                struct_decls.add(fn._struct_type_decl)
        
        result = "\n".join(sorted(struct_decls) + self.globals + [str(f) for f in self.functions])
        self._str_cache = result
        self._dirty = False
        return result
    
def get_llvm_target_machine():
    """Get or create an optimized LLVM target machine for JIT compilation."""
    global _llvm_target_machine
    if _llvm_target_machine is None:
        # Initialize LLVM
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        # Create target machine with optimizations
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine(
            opt=3,  # -O3 optimization level
            reloc='pic',
            codemodel='jitdefault'
        )
        _llvm_target_machine = target_machine
    
    return _llvm_target_machine

def create_ir_function_from_string(ir_str, name=None):
    import re
    sig_match = re.search(r'define\s+(?:[^@]*@)?([^(]+)\(([^)]*)\)[^{]*{', ir_str)
    if not sig_match:
        raise ValueError(f"[pyir] Could not parse function signature from IR")
    func_name = name or sig_match.group(1).strip()
    args_str = sig_match.group(2).strip()
    args = []
    if args_str:
        for arg in args_str.split(','):
            arg = arg.strip()
            if arg:
                parts = arg.split('%')
                if len(parts) == 2:
                    arg_type = parts[0].strip()
                    arg_name = parts[1].strip()
                    args.append((arg_name, arg_type))
    ret_type_match = re.search(r'define\s+([^@]+)@', ir_str)
    ret_type = ret_type_match.group(1).strip() if ret_type_match else "void"
    attrs_match = re.search(r'define\s+(?:[^@]*@)?[^(]+\([^)]*\)\s+([^{]*)', ir_str)
    attrs = attrs_match.group(1).strip() if attrs_match else ""
    ir_fn = IRFunction(func_name, args, ret_type, attrs)
    blocks = re.findall(r'([^:]+):\s*\n((?:\s+[^\n]+\n?)*)', ir_str)
    for block_label, block_content in blocks:
        block = IRBlock(block_label.strip())
        for line in block_content.strip().split('\n'):
            line = line.strip()
            if line:
                block.add(IRInstr(line))
        ir_fn.add_block(block)
    return ir_fn

def merge_ir_functions(functions, merged_name="merged"):
    if not functions:
        raise ValueError("[pyir] No functions to merge")
    base_fn = functions[0]
    merged_fn = IRFunction(merged_name, base_fn.args, base_fn.ret_type, base_fn.attrs)
    for i, fn in enumerate(functions):
        for block in fn.blocks:
            if block.label == 'entry' and i > 0:
                block.label = f'entry_{i}'
            merged_fn.add_block(block)
    return merged_fn

def _compile_ir(ir_code: str, fn_name: str = None, args=None, ret_ctype=None):
    try:
        validate_ir(ir_code)  # Always validate IR before JIT
    except Exception as e:
        raise ValueError(f"[pyir._compile_ir] Error validating IR:\n{e}\n")
    from ..security.sandbox import sandbox_mode
    if sandbox_mode:
        # Run JIT and execution in a subprocess for sandboxing
        def worker(ir_code_bytes, fn_name, args_bytes, ret_ctype_bytes, result_queue):
            import llvmlite.binding as llvm
            import ctypes
            import pickle
            try:
                ir_code = pickle.loads(ir_code_bytes)
                args = pickle.loads(args_bytes) if args_bytes else None
                ret_ctype = pickle.loads(ret_ctype_bytes) if ret_ctype_bytes else ctypes.c_double
                mod = llvm.parse_assembly(ir_code)
                mod.verify()
                from pyir import _engine
                _engine.add_module(mod)
                _engine.finalize_object()
                _engine.run_static_constructors()
                if fn_name and args is not None:
                    addr = _engine.get_function_address(fn_name)
                    # Support int, float, bool, tuple/list, struct, array, vector, nested, custom ctypes, opaque types, numpy arrays, device pointers
                    def infer_ctype(val):
                        import numpy as np
                        try:
                            import numba.cuda as nbcuda
                        except ImportError:
                            nbcuda = None
                        if isinstance(val, bool):
                            return ctypes.c_bool
                        elif isinstance(val, int):
                            return ctypes.c_int64
                        elif isinstance(val, float):
                            return ctypes.c_double
                        elif isinstance(val, (tuple, list)):
                            return infer_ctype(val[0]) * len(val)
                        elif hasattr(val, '_fields_') and issubclass(type(val), ctypes.Structure):
                            return type(val)
                        elif isinstance(val, ctypes.Array):
                            return type(val)
                        elif hasattr(val, '_type_') and hasattr(val, '_length_'):
                            return type(val)
                        elif hasattr(val, '_as_parameter_'):
                            return type(val)
                        elif isinstance(val, bytes):
                            return ctypes.c_char * len(val)
                        elif nbcuda and isinstance(val, nbcuda.cudadrv.devicearray.DeviceNDArray):
                            # Device pointer as int64
                            return ctypes.c_uint64
                        elif isinstance(val, np.ndarray):
                            # Numpy array as pointer
                            return ctypes.c_void_p
                        else:
                            raise TypeError(f"[pyir.sandbox] Unsupported arg type: {type(val)}")
                    ctypes_args = [infer_ctype(a) for a in args]
                    cfunc_ty = ctypes.CFUNCTYPE(ret_ctype, *ctypes_args)
                    cfunc = cfunc_ty(addr)
                    def flatten(val):
                        import numpy as np
                        try:
                            import numba.cuda as nbcuda
                        except ImportError:
                            nbcuda = None
                        if isinstance(val, (tuple, list)):
                            return [x for v in val for x in flatten(v)]
                        elif hasattr(val, '_fields_') and issubclass(type(val), ctypes.Structure):
                            return [getattr(val, f[0]) for f in val._fields_]
                        elif isinstance(val, ctypes.Array):
                            return list(val)
                        elif isinstance(val, bytes):
                            return list(val)
                        elif nbcuda and isinstance(val, nbcuda.cudadrv.devicearray.DeviceNDArray):
                            return [val.device_ctypes_pointer.value]
                        elif isinstance(val, np.ndarray):
                            return [val.ctypes.data]
                        else:
                            return [val]
                    flat_args = []
                    for a in args:
                        flat_args.extend(flatten(a))
                    result = cfunc(*flat_args)
                    def unpack(val):
                        if hasattr(val, '_fields_') and issubclass(type(val), ctypes.Structure):
                            return tuple(unpack(getattr(val, f[0])) for f in val._fields_)
                        elif isinstance(val, ctypes.Array):
                            return tuple(unpack(x) for x in val)
                        elif isinstance(val, bytes):
                            return val
                        else:
                            return val
                    result_queue.put(unpack(result))
                else:
                    result_queue.put(True)
            except Exception as e:
                result_queue.put(e)
        ctx = multiprocessing.get_context('spawn')
        result_queue = ctx.Queue()
        p = ctx.Process(target=worker, args=(pickle.dumps(ir_code), fn_name, pickle.dumps(args) if args else None, pickle.dumps(ret_ctype) if ret_ctype else None, result_queue))
        p.start()
        p.join()
        if not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                raise RuntimeError(f"[pyir] Sandbox subprocess error: {result}")
            return result
        else:
            raise RuntimeError("[pyir] Sandbox subprocess failed with no result.")
    try:
        mod = llvm.parse_assembly(ir_code)
        mod.verify()
        from pyir import _engine
        _engine.add_module(mod)
        _engine.finalize_object()
        _engine.run_static_constructors()
        if pyir_debug:
            print(f"[pyir._compile_ir] Successfully compiled and added module for {fn_name}")
        return mod
    except Exception as e:
        msg = f"Failed to compile LLVM IR"
        if fn_name:
            msg += f" for function '{fn_name}'"
        msg += f":\n{e}\n--- IR snippet ---\n{ir_code[:500]}\n--- End IR ---\n"
        msg += "Check your IR syntax, types, and ensure all variables are uniquely named."
        raise ValueError(msg)

def _execute_ir_function(func_name: str, args: tuple, sig: inspect.Signature):
    """
    Execute a compiled IR function with the given arguments.
    """
    import ctypes
    from .._engine import _engine
    
    # Get the IR for this function
    ir_code = _function_registry[func_name]
    if ir_code is None:
        raise ValueError(f"No IR available for function {func_name}")
    
    try:
        # Compile the IR if not already compiled
        _compile_ir(ir_code, func_name)
        
        # Get the function address
        func_addr = _engine.get_function_address(func_name)
        if func_addr == 0:
            raise RuntimeError(f"Could not get address for function {func_name}")
        
        # Determine argument and return types
        arg_types = []
        for param_name, param in sig.parameters.items():
            param_type = param.annotation
            if hasattr(param_type, 'ctype'):
                arg_types.append(param_type.ctype)
            else:
                # Default mappings
                if param_type in (int, 'i32'):
                    arg_types.append(ctypes.c_int32)
                elif param_type in (float, 'float'):
                    arg_types.append(ctypes.c_float)
                else:
                    arg_types.append(ctypes.c_float)  # Default to float
        
        # Determine return type
        if sig.return_annotation != inspect.Signature.empty:
            ret_type = sig.return_annotation
            if hasattr(ret_type, 'ctype'):
                ret_ctype = ret_type.ctype
            else:
                # Default mappings
                if ret_type in (int, 'i32'):
                    ret_ctype = ctypes.c_int32
                elif ret_type in (float, 'float'):
                    ret_ctype = ctypes.c_float
                else:
                    ret_ctype = ctypes.c_float  # Default to float
        else:
            ret_ctype = ctypes.c_float  # Default to float
        
        # Create the function type
        func_type = ctypes.CFUNCTYPE(ret_ctype, *arg_types)
        
        # Create the callable
        compiled_func = func_type(func_addr)
        
        # Convert arguments to appropriate ctypes
        converted_args = []
        for i, arg in enumerate(args):
            arg_type = arg_types[i]
            if arg_type == ctypes.c_int32:
                converted_args.append(int(arg))
            elif arg_type == ctypes.c_float:
                converted_args.append(float(arg))
            else:
                converted_args.append(arg)
        
        # Call the compiled function
        result = compiled_func(*converted_args)
        
        if pyir_debug:
            print(f"[pyir.function] Executed {func_name}({args}) = {result}")
        
        return result
        
    except Exception as e:
        if pyir_debug:
            print(f"[pyir.function] Error executing IR function {func_name}: {e}")
            print(f"[pyir.function] IR code:\n{ir_code}")
        raise

def collect_ir_from_function(fn):
    """
    Execute the function with a mock pyir.inline to collect IR instructions.
    Returns a list of IR instruction strings.
    """
    import types
    import inspect
    sig = inspect.signature(fn)
    ir_instructions = []
    # Create a mock pyir module with inline collector
    class IRCollector:
        def __init__(self):
            self.instructions = []
        def inline(self, ir_str: str, sugar=False):
            self.instructions.append(ir_str.strip())
            return None
    collector = IRCollector()
    # Prepare mock globals
    mock_globals = fn.__globals__.copy()
    mock_pyir = types.ModuleType('pyir')
    mock_pyir.inline = collector.inline
    # Copy type annotations if present
    pyir_mod = mock_globals.get('pyir', None)
    for attr in ['int8','int16','int32','int64','float16','float32','float64','complex64','complex128','ptr']:
        if pyir_mod is not None and hasattr(pyir_mod, attr):
            setattr(mock_pyir, attr, getattr(pyir_mod, attr))
    mock_globals['pyir'] = mock_pyir
    # Prepare dummy arguments
    mock_args = {}
    for param_name, param in sig.parameters.items():
        param_type = param.annotation
        if hasattr(param_type, 'llvm'):
            mock_args[param_name] = 0
        else:
            mock_args[param_name] = 0
    # Execute function to collect IR
    try:
        mock_func = types.FunctionType(
            fn.__code__,
            mock_globals,
            fn.__name__,
            fn.__defaults__,
            fn.__closure__
        )
        mock_func(**mock_args)
    except Exception:
        pass  # Ignore errors during IR collection
    return collector.instructions