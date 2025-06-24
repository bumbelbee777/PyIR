from .core import ssa
import textwrap

def add(a, b, out=None, type='i32'):
    """Emit IR for integer addition."""
    out = out or ssa('add')
    return f"{out} = add {type} {a}, {b}"

def sub(a, b, out=None, type='i32'):
    out = out or ssa('sub')
    return f"{out} = sub {type} {a}, {b}"

def mul(a, b, out=None, type='i32'):
    out = out or ssa('mul')
    return f"{out} = mul {type} {a}, {b}"

def div(a, b, out=None, type='i32'):
    out = out or ssa('div')
    return f"{out} = sdiv {type} {a}, {b}"

def fadd(a, b, out=None, type='double'):
    out = out or ssa('fadd')
    return f"{out} = fadd {type} {a}, {b}"

def fsub(a, b, out=None, type='double'):
    out = out or ssa('fsub')
    return f"{out} = fsub {type} {a}, {b}"

def fmul(a, b, out=None, type='double'):
    out = out or ssa('fmul')
    return f"{out} = fmul {type} {a}, {b}"

def fdiv(a, b, out=None, type='double'):
    out = out or ssa('fdiv')
    return f"{out} = fdiv {type} {a}, {b}"

def if_then(cond_var, true_block, false_block=None, label_base='if'):
    """
    Emit IR for an if-then(-else) block.
    cond_var: pyir.ssa variable holding i1 (0/1)
    true_block, false_block: callables returning IR strings
    Returns: IR string for the conditional
    """
    then_label = ssa(f'{label_base}_then')[1:]
    end_label = ssa(f'{label_base}_end')[1:]
    else_label = ssa(f'{label_base}_else')[1:] if false_block else end_label
    ir = [
        f"br i1 {cond_var}, label %{then_label}, label %{else_label}",
        f"{then_label}:",
        textwrap.indent(true_block(), '  '),
        f"  br label %{end_label}",
    ]
    if false_block:
        ir += [
            f"{else_label}:",
            textwrap.indent(false_block(), '  '),
            f"  br label %{end_label}",
        ]
    ir.append(f"{end_label}:")
    return '\n'.join(ir)

# --- High-level loop helper ---
def for_loop(var, start, end, body, type='i32', label_base='for'):
    """
    Emit IR for a simple counted for loop.
    var: loop variable name (string)
    start, end: pyir.ssa vars or constants
    body: callable taking loop var pyir.ssa name, returns IR string
    Returns: IR string for the loop
    """
    preheader = ssa(f'{label_base}_preheader')[1:]
    header = ssa(f'{label_base}_header')[1:]
    body_label = ssa(f'{label_base}_body')[1:]
    end_label = ssa(f'{label_base}_end')[1:]
    ivar = ssa(var)
    ir = [
        f"br label %{preheader}",
        f"{preheader}:",
        f"  {ivar} = phi {type} [{start}, %entry], [%next_{var}, %{body_label}]",
        f"  %cond = icmp slt {type} {ivar}, {end}",
        f"  br i1 %cond, label %{body_label}, label %{end_label}",
        f"{body_label}:",
        textwrap.indent(body(ivar), '    '),
        f"  %next_{var} = add {type} {ivar}, 1",
        f"  br label %{preheader}",
        f"{end_label}:"
    ]
    return '\n'.join(ir)

# --- Pointer arithmetic and dereferencing helpers ---
def gep(ptr_var, indices, out=None, type='i32*'):
    """Emit IR for getelementptr (GEP) instruction.
    ptr_var: base pointer pyir.ssa name
    indices: list of indices (pyir.ssa names or constants)
    type: pointer type (e.g., 'i32*')
    Returns: IR string for the GEP result
    """
    out = out or ssa('gep')
    idx_str = ', '.join(f'i32 {i}' for i in indices)
    return f"{out} = getelementptr inbounds {type} {ptr_var}, {idx_str}"

def ptr_add(ptr_var, offset, out=None, type='i8*'):
    """Emit IR for pointer addition (GEP on i8*)."""
    out = out or ssa('ptradd')
    return f"{out} = getelementptr i8, i8* {ptr_var}, i64 {offset}"

def ptr_sub(ptr_var, offset, out=None, type='i8*'):
    """Emit IR for pointer subtraction (GEP on i8* with negative offset)."""
    out = out or ssa('ptrsub')
    out = out or ssa('ptrsub')
    return f"{out} = getelementptr i8, i8* {ptr_var}, i64 -{offset}"

def load(ptr_var, out=None, type='i32'):
    """Emit IR for loading from a pointer."""
    out = out or ssa('load')
    return f"{out} = load {type}, {type}* {ptr_var}"

def store(value, ptr_var, type='i32'):
    """Emit IR for storing a value to a pointer."""
    return f"store {type} {value}, {type}* {ptr_var}"

def call_fnptr(fnptr_var, args, out=None, ret_type='i32', arg_types=None):
    """Emit IR for calling a function pointer.
    fnptr_var: pyir.ssa name of the function pointer
    args: list of pyir.ssa names or constants
    arg_types: list of LLVM types for arguments
    ret_type: LLVM type of return value
    Returns: IR string for the call result
    """
    out = out or ssa('call')
    if arg_types is None:
        raise ValueError("[pyir.call_fnptr] arg_types must be provided.")
    arg_str = ', '.join(f'{t} {a}' for t, a in zip(arg_types, args))
    return f"{out} = call {ret_type} {fnptr_var}({arg_str})"
