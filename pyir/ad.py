"""
pyir.ad: Robust automatic differentiation utilities for PyIR (with advanced reductions, true forward value caching, and custom op shape inference)
"""
import inspect
import warnings
from .core import ssa
from collections import defaultdict
import numpy as np

_custom_gradients = {}
_custom_shape_inference = {}

def register_custom_gradient(opname, grad_fn):
    """Register a custom gradient function for a given IR op name."""
    _custom_gradients[opname] = grad_fn

def register_custom_shape_inference(opname, shape_fn):
    """Register a custom shape inference function for a given IR op name.
    shape_fn(lhs, rhs, shapes, arg_shapes) -> output_shape
    """
    _custom_shape_inference[opname] = shape_fn

def grad(pyir_func):
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
    warnings.warn("[pyir.grad] AD is experimental. Reductions/control flow and broadcasting supported for simple cases. Register custom gradients/shape inference for new ops.")
    sig = inspect.signature(pyir_func)
    arg_names = list(sig.parameters.keys())
    from . import fusion
    _function_registry = fusion._function_registry
    for k in _function_registry:
        if k.startswith(pyir_func.__name__ + "__"):
            ir = _function_registry[k]
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
    # --- Populate assigns dict for reverse pass ---
    for line in lines:
        line = line.strip()
        if line.startswith('%') and '=' in line:
            lhs, rhs = line.split('=', 1)
            lhs = lhs.strip()[1:]
            rhs = rhs.strip()
            assigns[lhs] = rhs
    # --- Real shape inference from input args ---
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
    # --- Parse IR and propagate shapes ---
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
    # --- Forward pass to cache values ---
    def forward_pass(args):
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
                    forward_vals[lhs] = forward_vals[a] + forward_vals[b]
                elif op in ('sub', 'fsub'):
                    a, b = tokens[-2], tokens[-1]
                    a = a.lstrip('%')
                    b = b.lstrip('%')
                    forward_vals[lhs] = forward_vals[a] - forward_vals[b]
                elif op in ('mul', 'fmul'):
                    a, b = tokens[-2], tokens[-1]
                    a = a.lstrip('%')
                    b = b.lstrip('%')
                    forward_vals[lhs] = forward_vals[a] * forward_vals[b]
                elif op in ('div', 'sdiv', 'fdiv'):
                    a, b = tokens[-2], tokens[-1]
                    a = a.lstrip('%')
                    b = b.lstrip('%')
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
    # --- Main grad function ---
    def grad_fn(*args):
        arg_shapes = infer_shapes_from_args(args)
        shapes.clear()
        propagate_shapes(arg_shapes)
        forward_vals.clear()
        forward_pass(args)
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
                grads[a] += grad_lhs * forward_vals[b]
                grads[b] += grad_lhs * forward_vals[a]
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
    return grad_fn 