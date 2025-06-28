import ctypes

named_types = {}

class IntType:
    """LLVM integer type (e.g., pyir.int32)"""
    def __init__(self, bits: int):
        self.bits = bits
        self.llvm = f"i{bits}"
        if bits == 1:
            self.ctype = ctypes.c_bool
        else:
            self.ctype = getattr(ctypes, f"c_int{bits}")
    def __repr__(self):
        return f"pyir.int{self.bits}"

class int8(IntType):
    bits = 8
    llvm = "i8"
    ctype = ctypes.c_int8
    def __init__(self): super().__init__(8)
    def __repr__(self): return "pyir.int8"
class int16(IntType):
    bits = 16
    llvm = "i16"
    ctype = ctypes.c_int16
    def __init__(self): super().__init__(16)
    def __repr__(self): return "pyir.int16"
class int32(IntType):
    bits = 32
    llvm = "i32"
    ctype = ctypes.c_int32
    def __init__(self): super().__init__(32)
    def __repr__(self): return "pyir.int32"
class int64(IntType):
    bits = 64
    llvm = "i64"
    ctype = ctypes.c_int64
    def __init__(self): super().__init__(64)
    def __repr__(self): return "pyir.int64"

class FloatType:
    """LLVM float type (e.g., pyir.float64)"""
    def __init__(self, bits: int):
        self.bits = bits
        self.llvm = "double" if bits == 64 else "float"
        self.ctype = ctypes.c_double if bits == 64 else ctypes.c_float
    def __repr__(self):
        return f"pyir.float{self.bits}"

class float16(FloatType):
    bits = 16
    llvm = "half"
    ctype = ctypes.c_uint16
    def __init__(self): super().__init__(16)
    def __repr__(self): return "pyir.float16"
class float32(FloatType):
    bits = 32
    llvm = "float"
    ctype = ctypes.c_float
    def __init__(self): super().__init__(32)
    def __repr__(self): return "pyir.float32"
class float64(FloatType):
    bits = 64
    llvm = "double"
    ctype = ctypes.c_double
    def __init__(self): super().__init__(64)
    def __repr__(self): return "pyir.float64"

class ComplexType:
    """LLVM complex type (e.g., pyir.complex64, pyir.complex128)"""
    def __init__(self, bits: int):
        self.bits = bits
        if bits == 64:
            self.llvm = '{float, float}'
            self.ctype = ctypes.c_float * 2
        elif bits == 128:
            self.llvm = '{double, double}'
            self.ctype = ctypes.c_double * 2
        else:
            raise ValueError(f"Unsupported complex type bits: {bits}")
    def __repr__(self):
        return f"pyir.complex{self.bits}"

class complex64(ComplexType):
    bits = 64
    llvm = '{float, float}'
    ctype = ctypes.c_float * 2
    def __init__(self): super().__init__(64)
    def __repr__(self): return "pyir.complex64"
class complex128(ComplexType):
    bits = 128
    llvm = '{double, double}'
    ctype = ctypes.c_double * 2
    def __init__(self): super().__init__(128)
    def __repr__(self): return "pyir.complex128"

class bool(IntType):
    bits = 1
    llvm = "i1"
    ctype = ctypes.c_bool
    def __init__(self): super().__init__(1)
    def __repr__(self): return "pyir.bool"

# Backward-compatible instances (optional, for legacy code)
int8_ = int8()
int16_ = int16()
int32_ = int32()
int64_ = int64()
float16_ = float16()
float32_ = float32()
float64_ = float64()
complex64_ = complex64()
complex128_ = complex128()
bool_ = bool()

class VectorType:
    def __init__(self, elem_type, count):
        self.elem_type = elem_type
        self.count = count
        self.llvm = f"<{count} x {elem_type.llvm}>"
        self.ctype = elem_type.ctype * count
    def __repr__(self):
        return f"pyir.vec({self.elem_type}, {self.count})"
def vec(elem_type, count): return VectorType(elem_type, count)

# Vector type classes
class vec4f(VectorType):
    def __init__(self): super().__init__(float32, 4)
    def __repr__(self): return "pyir.vec4f"
class vec8f(VectorType):
    def __init__(self): super().__init__(float32, 8)
    def __repr__(self): return "pyir.vec8f"
class vec4d(VectorType):
    def __init__(self): super().__init__(float64, 4)
    def __repr__(self): return "pyir.vec4d"
class vec4i(VectorType):
    def __init__(self): super().__init__(int32, 4)
    def __repr__(self): return "pyir.vec4i"
class vec8i(VectorType):
    def __init__(self): super().__init__(int32, 8)
    def __repr__(self): return "pyir.vec8i"

# Backward-compatible instances
vec4f_ = vec4f()
vec8f_ = vec8f()
vec4d_ = vec4d()
vec4i_ = vec4i()
vec8i_ = vec8i()

# Additional common instances
vec4f = vec4f()
vec8f = vec8f()
vec4d = vec4d()
vec4i = vec4i()
vec8i = vec8i()

class StructType:
    def __init__(self, field_types):
        self.field_types = field_types
        self.llvm = f"{{{', '.join(t.llvm for t in field_types)}}}"
        self.ctype = type('Struct', (ctypes.Structure,), {'_fields_': [(f'f{i}', t.ctype) for i, t in enumerate(field_types)]})
    def __repr__(self):
        return f"pyir.struct([{', '.join(map(str, self.field_types))}])"
def struct(field_types): return StructType(field_types)

class ArrayType:
    def __init__(self, elem_type, count):
        self.elem_type = elem_type
        self.count = count
        self.llvm = f"[{count} x {elem_type.llvm}]"
        self.ctype = elem_type.ctype * count
    def __repr__(self):
        return f"pyir.array({self.elem_type}, {self.count})"
def array(elem_type, count): return ArrayType(elem_type, count)

class VoidType:
    llvm = 'void'
    ctype = None
    def __repr__(self): return "pyir.void"
void = VoidType()

class PointerType:
    def __init__(self, base_type):
        self.base_type = base_type
        self.llvm = f"{base_type.llvm}*"
        self.ctype = ctypes.POINTER(base_type.ctype) if hasattr(base_type, 'ctype') and base_type.ctype else None
    def __repr__(self):
        return f"pyir.ptr({self.base_type})"
def ptr(base_type): return PointerType(base_type)

class FunctionPointerType:
    def __init__(self, arg_types, ret_type):
        self.arg_types = arg_types
        self.ret_type = ret_type
        self.llvm = f"{ret_type.llvm} ({', '.join(t.llvm for t in arg_types)})*"
        self.ctype = ctypes.CFUNCTYPE(ret_type.ctype, *(t.ctype for t in arg_types)) if ret_type.ctype else None
    def __repr__(self):
        return f"pyir.fnptr([{', '.join(map(str, self.arg_types))}], {self.ret_type})"
def fnptr(arg_types, ret_type): return FunctionPointerType(arg_types, ret_type)

class OpaqueType:
    def __init__(self, name):
        self.name = name
        self.llvm = f"%{name}"
        self.ctype = None
    def __repr__(self):
        return f"pyir.opaque('{self.name}')"
def opaque(name): return OpaqueType(name)

def define_type(name, llvm_type):
    named_types[name] = llvm_type
    return llvm_type

# Add python_type_map for type mangling and registration
python_type_map = {
    int: int32,
    float: float32,
    bool: bool,
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
