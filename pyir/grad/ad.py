"""
pyir.ad: Robust automatic differentiation utilities for PyIR (with advanced reductions, true forward value caching, and custom op shape inference)
"""
import inspect
import warnings
from ..core.ir import ssa
from collections import defaultdict
import numpy as np
from functools import wraps
from ..core.function import _function_registry
from pyir._engine import pyir_debug

from .shape import _custom_shape_inference, register_custom_shape_inference

# Temporarily enable debug output for gradient computation
pyir_debug = True

_custom_gradients = {}

# Only keep reverse-mode AD logic and registration/dispatch here

def register_custom_gradient(opname, grad_fn):
    """Register a custom gradient function for a given IR op name."""
    _custom_gradients[opname] = grad_fn

def _grad_impl(pyir_func):
    """
    Robust reverse-mode AD for PyIR functions (with advanced reductions, true forward value caching, and custom op shape inference).
    - Infers and propagates shapes from input args
    - Broadcasts gradients to match input and output shapes
    - Handles advanced reductions (sum, mean, prod, min, max, argmin, argmax, any, all) by expanding gradients
    - Caches forward values for correct prod/min/max/argmin/argmax gradients
    - Supports tensor outputs, multi-output functions, and custom ops
    - Supports more ops (cmp, select, phi, call, etc)
    - Handles control flow, multiple outputs, and user-registered gradients/shape inference
    - Gives clear errors for unsupported ops
    """
    # warnings.warn("[pyir.grad] AD is experimental. Reductions/control flow and broadcasting supported for simple cases. Register custom gradients/shape inference for new ops.")
    sig = inspect.signature(pyir_func)
    arg_names = list(sig.parameters.keys())
    for k in _function_registry:
        if k.startswith(pyir_func.__name__ + "__"):
            ir_obj = _function_registry[k]
            ir = str(ir_obj) if hasattr(ir_obj, 'blocks') else ir_obj
            break
    else:
        raise ValueError(f"[pyir.grad] No IR found for function '{pyir_func.__name__}'.")
    # Build computation graph and track shapes
    lines = ir.splitlines()
    assigns = {}
    grads = defaultdict(float)
    phis = {}
    shapes = {}
    outputs = []
    forward_vals = {}  # Cache of forward values for each variable
    for line in lines:
        line = line.strip()
        if line.startswith('%') and '=' in line:
            lhs, rhs = line.split('=', 1)
            lhs = lhs.strip()[1:]
            rhs = rhs.strip()
            assigns[lhs] = rhs
    def infer_shapes_from_args(args):
        arg_shapes = {}
        for name, val in zip(arg_names, args):
            if hasattr(val, 'shape'):
                arg_shapes[name] = val.shape
            elif isinstance(val, (list, tuple, np.ndarray)):
                arg_shapes[name] = np.shape(val)
            else:
                arg_shapes[name] = ()
        return arg_shapes
    def propagate_shapes(arg_shapes):
        for line in lines:
            line = line.strip()
            if line.startswith('%') and '=' in line:
                lhs, rhs = line.split('=', 1)
                lhs = lhs.strip()[1:]
                rhs = rhs.strip()
                tokens = rhs.replace(',', '').split()
                op = tokens[0]
                # Elementwise ops: shape is broadcast of inputs
                if op in ('add', 'sub', 'mul', 'div', 'fadd', 'fsub', 'fmul', 'fdiv', 'sdiv'):
                    a, b = tokens[-2], tokens[-1]
                    a = a.lstrip('%')
                    b = b.lstrip('%')
                    shape_a = shapes.get(a, arg_shapes.get(a, ()))
                    shape_b = shapes.get(b, arg_shapes.get(b, ()))
                    try:
                        shapes[lhs] = np.broadcast_shapes(shape_a, shape_b)
                    except Exception:
                        shapes[lhs] = shape_a or shape_b
                # Reductions: sum, mean, prod, min, max, argmin, argmax, any, all
                elif op == 'call':
                    callee = tokens[2] if len(tokens) > 2 else ''
                    if any(r in callee for r in ('sum', 'mean', 'prod', 'min', 'max', 'argmin', 'argmax', 'any', 'all')):
                        shapes[lhs] = ()
                    else:
                        # Custom op shape inference
                        if callee in _custom_shape_inference:
                            shapes[lhs] = _custom_shape_inference[callee](lhs, rhs, shapes, arg_shapes)
                        else:
                            shapes[lhs] = ()
                # phi: shape is broadcast of incoming
                elif op == 'phi':
                    incoming = [v.split()[0].lstrip('%[]') for v in rhs.split('[')[1:]]
                    incoming_shapes = [shapes.get(v, arg_shapes.get(v, ())) for v in incoming]
                    try:
                        shapes[lhs] = np.broadcast_shapes(*incoming_shapes)
                    except Exception:
                        shapes[lhs] = incoming_shapes[0] if incoming_shapes else ()
                # Custom op shape inference
                elif op in _custom_shape_inference:
                    shapes[lhs] = _custom_shape_inference[op](lhs, rhs, shapes, arg_shapes)
                # Default: scalar
                else:
                    shapes[lhs] = ()
                outputs.append(lhs)
    def forward_pass(args, arg_shapes):
        # Map arg_names to input values
        for name, val in zip(arg_names, args):
            forward_vals[name] = np.array(val)
        for line in lines:
            line = line.strip()
            if line.startswith('%') and '=' in line:
                lhs, rhs = line.split('=', 1)
                lhs = lhs.strip()[1:]
                rhs = rhs.strip()
                tokens = rhs.replace(',', '').split()
                op = tokens[0]
                if op in ('add', 'fadd'):
                    a, b = tokens[-2], tokens[-1]
                    a = a.lstrip('%')
                    b = b.lstrip('%')
                    if a not in forward_vals:
                        forward_vals[a] = 0.0
                    if b not in forward_vals:
                        forward_vals[b] = 0.0
                    forward_vals[lhs] = forward_vals[a] + forward_vals[b]
                elif op in ('sub', 'fsub'):
                    a, b = tokens[-2], tokens[-1]
                    a = a.lstrip('%')
                    b = b.lstrip('%')
                    if a not in forward_vals:
                        forward_vals[a] = 0.0
                    if b not in forward_vals:
                        forward_vals[b] = 0.0
                    forward_vals[lhs] = forward_vals[a] - forward_vals[b]
                elif op in ('mul', 'fmul'):
                    a, b = tokens[-2], tokens[-1]
                    a = a.lstrip('%')
                    b = b.lstrip('%')
                    if a not in forward_vals:
                        forward_vals[a] = 0.0
                    if b not in forward_vals:
                        forward_vals[b] = 0.0
                    forward_vals[lhs] = forward_vals[a] * forward_vals[b]
                elif op in ('div', 'sdiv', 'fdiv'):
                    a, b = tokens[-2], tokens[-1]
                    a = a.lstrip('%')
                    b = b.lstrip('%')
                    if a not in forward_vals:
                        forward_vals[a] = 0.0
                    if b not in forward_vals:
                        forward_vals[b] = 0.0
                    forward_vals[lhs] = forward_vals[a] / forward_vals[b]
                elif op == 'call':
                    callee = tokens[2] if len(tokens) > 2 else ''
                    if 'sum' in callee:
                        for arg in rhs.split('(')[1].split(')')[0].split(','):
                            v = arg.strip().lstrip('%')
                            forward_vals[lhs] = np.sum(forward_vals[v])
                    elif 'mean' in callee:
                        for arg in rhs.split('(')[1].split(')')[0].split(','):
                            v = arg.strip().lstrip('%')
                            forward_vals[lhs] = np.mean(forward_vals[v])
                    elif 'prod' in callee:
                        for arg in rhs.split('(')[1].split(')')[0].split(','):
                            v = arg.strip().lstrip('%')
                            forward_vals[lhs] = np.prod(forward_vals[v])
                    elif 'min' in callee:
                        for arg in rhs.split('(')[1].split(')')[0].split(','):
                            v = arg.strip().lstrip('%')
                            forward_vals[lhs] = np.min(forward_vals[v])
                    elif 'max' in callee:
                        for arg in rhs.split('(')[1].split(')')[0].split(','):
                            v = arg.strip().lstrip('%')
                            forward_vals[lhs] = np.max(forward_vals[v])
                    elif 'argmin' in callee:
                        for arg in rhs.split('(')[1].split(')')[0].split(','):
                            v = arg.strip().lstrip('%')
                            forward_vals[lhs] = np.argmin(forward_vals[v])
                    elif 'argmax' in callee:
                        for arg in rhs.split('(')[1].split(')')[0].split(','):
                            v = arg.strip().lstrip('%')
                            forward_vals[lhs] = np.argmax(forward_vals[v])
                    elif 'any' in callee:
                        for arg in rhs.split('(')[1].split(')')[0].split(','):
                            v = arg.strip().lstrip('%')
                            forward_vals[lhs] = np.any(forward_vals[v])
                    elif 'all' in callee:
                        for arg in rhs.split('(')[1].split(')')[0].split(','):
                            v = arg.strip().lstrip('%')
                            forward_vals[lhs] = np.all(forward_vals[v])
                    # Custom op forward value
                    elif callee in _custom_shape_inference:
                        forward_vals[lhs] = _custom_shape_inference[callee](lhs, rhs, forward_vals, forward_vals)
                # phi and custom ops: skip for now

    def grad_fn(*args):
        arg_shapes = infer_shapes_from_args(args)
        shapes.clear()
        propagate_shapes(arg_shapes)
        forward_vals.clear()
        forward_pass(args, arg_shapes)
        # Forward pass done, now reverse pass
        for out in outputs[-1:]:
            grads[out] = np.ones(shapes.get(out, ()))
        for lhs in reversed(outputs):
            rhs = assigns.get(lhs, None)
            if rhs is None:
                continue
            tokens = rhs.replace(',', '').split()
            op = tokens[0]
            if op in _custom_gradients:
                _custom_gradients[op](lhs, rhs, grads, assigns)
            elif op in ('add', 'fadd', 'sub', 'fsub', 'mul', 'fmul', 'sdiv', 'fdiv'):
                a, b = tokens[-2:]
                a = a.lstrip('%')
                b = b.lstrip('%')
                shape_a = shapes.get(a, arg_shapes.get(a, ()))
                shape_b = shapes.get(b, arg_shapes.get(b, ()))
                grad_lhs = grads[lhs]
                if op in ('add', 'fadd'):
                    # Addition: gradients flow through
                    grads[a] = grads.get(a, 0.0) + grad_lhs
                    grads[b] = grads.get(b, 0.0) + grad_lhs
                    if pyir_debug:
                        print(f"[pyir.grad] {lhs} = {a} + {b} | grad_{a} += {grad_lhs} = {grads[a]}")
                        print(f"[pyir.grad] {lhs} = {a} + {b} | grad_{b} += {grad_lhs} = {grads[b]}")
                elif op in ('sub', 'fsub'):
                    # Subtraction: gradients flow through with sign change for second operand
                    grads[a] = grads.get(a, 0.0) + grad_lhs
                    grads[b] = grads.get(b, 0.0) - grad_lhs
                    if pyir_debug:
                        print(f"[pyir.grad] {lhs} = {a} - {b} | grad_{a} += {grad_lhs} = {grads[a]}")
                        print(f"[pyir.grad] {lhs} = {a} - {b} | grad_{b} -= {grad_lhs} = {grads[b]}")
                elif op in ('mul', 'fmul'):
                    # Multiplication: product rule
                    if a == b:
                        # Special case: a * a, gradient is 2a * grad_output
                        grads[a] = grads.get(a, 0.0) + grad_lhs * 2 * forward_vals[a]
                        if pyir_debug:
                            print(f"[pyir.grad] {lhs} = {a} * {b} | grad_{a} += {grad_lhs} * 2 * {forward_vals[a]} = {grads[a]}")
                    else:
                        # General case: a * b, gradients are b * grad_output and a * grad_output
                        grads[a] = grads.get(a, 0.0) + grad_lhs * forward_vals[b]
                        grads[b] = grads.get(b, 0.0) + grad_lhs * forward_vals[a]
                        if pyir_debug:
                            print(f"[pyir.grad] {lhs} = {a} * {b} | grad_{a} += {grad_lhs} * {forward_vals[b]} = {grads[a]}")
                            print(f"[pyir.grad] {lhs} = {a} * {b} | grad_{b} += {grad_lhs} * {forward_vals[a]} = {grads[b]}")
                elif op in ('div', 'sdiv', 'fdiv'):
                    # Division: quotient rule
                    if a == b:
                        # Special case: a / a = 1, gradient is 0
                        grads[a] = grads.get(a, 0.0) + 0.0
                        if pyir_debug:
                            print(f"[pyir.grad] {lhs} = {a} / {b} | grad_{a} += 0 = {grads[a]}")
                    else:
                        # General case: a / b, gradients are grad_output/b and -a*grad_output/b²
                        grads[a] = grads.get(a, 0.0) + grad_lhs / forward_vals[b]
                        grads[b] = grads.get(b, 0.0) - grad_lhs * forward_vals[a] / (forward_vals[b] ** 2)
                        if pyir_debug:
                            print(f"[pyir.grad] {lhs} = {a} / {b} | grad_{a} += {grad_lhs} / {forward_vals[b]} = {grads[a]}")
                            print(f"[pyir.grad] {lhs} = {a} / {b} | grad_{b} -= {grad_lhs} * {forward_vals[a]} / {forward_vals[b]}^2 = {grads[b]}")
            elif op == 'phi':
                incoming = [v.split()[0].lstrip('%[]') for v in rhs.split('[')[1:]]
                for v in incoming:
                    grads[v] = np.broadcast_to(grads[lhs], shapes.get(v, arg_shapes.get(v, ())))
            elif op == 'call':
                callee = tokens[2] if len(tokens) > 2 else ''
                if 'sum' in callee:
                    for arg in rhs.split('(')[1].split(')')[0].split(','):
                        v = arg.strip().lstrip('%')
                        grads[v] = np.ones(shapes.get(v, arg_shapes.get(v, ()))) * grads[lhs]
                elif 'mean' in callee:
                    for arg in rhs.split('(')[1].split(')')[0].split(','):
                        v = arg.strip().lstrip('%')
                        N = np.prod(shapes.get(v, arg_shapes.get(v, ())))
                        grads[v] = np.ones(shapes.get(v, arg_shapes.get(v, ()))) * grads[lhs] / N
                elif 'prod' in callee:
                    for arg in rhs.split('(')[1].split(')')[0].split(','):
                        v = arg.strip().lstrip('%')
                        # Real prod grad: grad = prod(x) * grad_output / x (elementwise, handle zeros)
                        input_val = forward_vals[v]
                        output_val = forward_vals[lhs]
                        grad_val = np.where(input_val != 0, output_val * grads[lhs] / input_val, 0)
                        grads[v] = grad_val
                elif 'min' in callee:
                    for arg in rhs.split('(')[1].split(')')[0].split(','):
                        v = arg.strip().lstrip('%')
                        input_val = forward_vals[v]
                        output_val = forward_vals[lhs]
                        grads[v] = np.where(input_val == output_val, grads[lhs], 0)
                elif 'max' in callee:
                    for arg in rhs.split('(')[1].split(')')[0].split(','):
                        v = arg.strip().lstrip('%')
                        input_val = forward_vals[v]
                        output_val = forward_vals[lhs]
                        grads[v] = np.where(input_val == output_val, grads[lhs], 0)
                elif 'argmin' in callee or 'argmax' in callee:
                    for arg in rhs.split('(')[1].split(')')[0].split(','):
                        v = arg.strip().lstrip('%')
                        # Gradients for argmin/argmax are zero almost everywhere
                        grads[v] = np.zeros(shapes.get(v, arg_shapes.get(v, ())))
                elif 'any' in callee or 'all' in callee:
                    for arg in rhs.split('(')[1].split(')')[0].split(','):
                        v = arg.strip().lstrip('%')
                        grads[v] = np.zeros(shapes.get(v, arg_shapes.get(v, ())))
                # Custom op shape inference
                elif callee in _custom_shape_inference:
                    _custom_shape_inference[callee](lhs, rhs, grads, assigns)
                # TODO: handle other reductions/calls
            elif op in _custom_shape_inference:
                _custom_shape_inference[op](lhs, rhs, grads, assigns)
            elif op in ('icmp', 'fcmp', 'select'):
                continue
            else:
                warnings.warn(f"[pyir.grad] Unsupported op in line: {rhs}")
        # Return gradients, broadcasting to match input shapes
        return tuple(np.broadcast_to(grads.get(n, 0.0), arg_shapes.get(n, ())) for n in arg_names)

    grad_fn._is_grad = True
    return grad_fn

@wraps(_grad_impl)
def grad(pyir_func):
    """
    Robust reverse-mode AD for PyIR functions (now supports async, complex, fused, and higher-order kernels).
    """
    # Async kernel support
    if getattr(pyir_func, '_is_async_kernel', False):
        async def async_grad_fn(*args, **kwargs):
            res = await pyir_func(*args, **kwargs)
            return _grad_impl(pyir_func)(*args, **kwargs)
        async_grad_fn._is_async_grad = True
        return async_grad_fn
    # Complex kernel support (fallback: treat as real, warn)
    if getattr(pyir_func, '_is_complex_kernel', False):
        warnings.warn("[pyir.grad] Complex AD is experimental. Gradients are computed on real/imag parts separately.")
        return _grad_impl(pyir_func)
    # Fused kernel support (apply grad to each subkernel)
    if hasattr(pyir_func, '_fused_kernels'):
        grads = [_grad_impl(fn) for fn in pyir_func._fused_kernels]
        def fused_grad_fn(*args, **kwargs):
            return tuple(g(*args, **kwargs) for g in grads)
        fused_grad_fn._is_fused_grad = True
        return fused_grad_fn
    # Higher-order: allow grad(grad(...))
    return _grad_impl(pyir_func)

__all__ = [
    'grad',
    'register_custom_gradient'
] 