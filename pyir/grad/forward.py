import numpy as np
import inspect
from pyir.core.function import _function_registry
from pyir._engine import pyir_debug

from .ad import grad

def jvp(pyir_func):
    """
    Forward-mode AD (Jacobian-vector product) for PyIR functions.
    Returns a function that computes (output, JVP) given primal and tangent inputs.
    """
    sig = inspect.signature(pyir_func)
    arg_names = list(sig.parameters.keys())
    for k in _function_registry:
        if k.startswith(pyir_func.__name__ + "__"):
            ir_obj = _function_registry[k]
            ir = str(ir_obj) if hasattr(ir_obj, 'blocks') else ir_obj
            break
    else:
        ir = None
    def jvp_fn(*args, tangents=None):
        if tangents is None:
            tangents = [np.ones_like(a) for a in args]
        if ir is not None:
            lines = ir.splitlines()
            values = {n: np.array(v) for n, v in zip(arg_names, args)}
            tangs = {n: np.array(t) for n, t in zip(arg_names, tangents)}
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
                        if a not in values:
                            values[a] = 0.0
                        if b not in values:
                            values[b] = 0.0
                        if a not in tangs:
                            tangs[a] = 0.0
                        if b not in tangs:
                            tangs[b] = 0.0
                        values[lhs] = values[a] + values[b]
                        tangs[lhs] = tangs[a] + tangs[b]
                        if pyir_debug:
                            print(f"[pyir.jvp] {lhs} = {a} + {b} | primal={values[lhs]}, tangent={tangs[lhs]}")
                    elif op in ('sub', 'fsub'):
                        a, b = tokens[-2], tokens[-1]
                        a = a.lstrip('%')
                        b = b.lstrip('%')
                        if a not in values:
                            values[a] = 0.0
                        if b not in values:
                            values[b] = 0.0
                        if a not in tangs:
                            tangs[a] = 0.0
                        if b not in tangs:
                            tangs[b] = 0.0
                        values[lhs] = values[a] - values[b]
                        tangs[lhs] = tangs[a] - tangs[b]
                        if pyir_debug:
                            print(f"[pyir.jvp] {lhs} = {a} - {b} | primal={values[lhs]}, tangent={tangs[lhs]}")
                    elif op in ('mul', 'fmul'):
                        a, b = tokens[-2], tokens[-1]
                        a = a.lstrip('%')
                        b = b.lstrip('%')
                        if a not in values:
                            values[a] = 0.0
                        if b not in values:
                            values[b] = 0.0
                        if a not in tangs:
                            tangs[a] = 0.0
                        if b not in tangs:
                            tangs[b] = 0.0
                        values[lhs] = values[a] * values[b]
                        tangs[lhs] = tangs[a] * values[b] + values[a] * tangs[b]
                        if pyir_debug:
                            print(f"[pyir.jvp] {lhs} = {a} * {b} | primal={values[lhs]}, tangent={tangs[lhs]}")
                    elif op in ('div', 'sdiv', 'fdiv'):
                        a, b = tokens[-2], tokens[-1]
                        a = a.lstrip('%')
                        b = b.lstrip('%')
                        if a not in values:
                            values[a] = 0.0
                        if b not in values:
                            values[b] = 0.0
                        if a not in tangs:
                            tangs[a] = 0.0
                        if b not in tangs:
                            tangs[b] = 0.0
                        values[lhs] = values[a] / values[b]
                        tangs[lhs] = (tangs[a] * values[b] - values[a] * tangs[b]) / (values[b] ** 2)
                        if pyir_debug:
                            print(f"[pyir.jvp] {lhs} = {a} / {b} | primal={values[lhs]}, tangent={tangs[lhs]}")
                    else:
                        values[lhs] = 0.0
                        tangs[lhs] = 0.0
                        if pyir_debug:
                            print(f"[pyir.jvp] {lhs} = {rhs} | (unsupported op, set to 0)")
            out_var = lhs
            return values[out_var], tangs[out_var]
        else:
            primal = pyir_func(*args)
            eps = 1e-8
            args_eps = [a + eps * t for a, t in zip(args, tangents)]
            primal_eps = pyir_func(*args_eps)
            jvp_out = (np.array(primal_eps) - np.array(primal)) / eps
            return primal, jvp_out
    jvp_fn._is_jvp = True
    return jvp_fn

def vjp(pyir_func):
    if not hasattr(pyir_func, 'grad'):
        pyir_func.grad = grad(pyir_func)  # Import grad from .ad
    grad_fn = pyir_func.grad
    def vjp_fn(*args, cotangents=None):
        primal = pyir_func(*args)
        grads = grad_fn(*args)
        if not isinstance(grads, (tuple, list)):
            grads = (grads,)
        if cotangents is None:
            cotangents = [np.ones_like(g) for g in grads]
        if not isinstance(cotangents, (tuple, list)):
            cotangents = (cotangents,)
        vjp_out = tuple(np.sum(g * c) for g, c in zip(grads, cotangents))
        return primal, vjp_out
    vjp_fn._is_vjp = True
    return vjp_fn

def jacobian(pyir_func):
    def jac_fn(*args):
        primal = pyir_func(*args)
        jac = []
        for i, a in enumerate(args):
            basis = [np.zeros_like(x) for x in args]
            basis[i] = np.ones_like(a)
            _, jvp_out = jvp(pyir_func)(*args, tangents=basis)
            jac.append(jvp_out)
        return np.stack(jac, axis=0)
    jac_fn._is_jacobian = True
    return jac_fn
