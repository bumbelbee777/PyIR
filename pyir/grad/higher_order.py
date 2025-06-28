def higher_order_grad(pyir_func, order=2):
    """
    Compute higher-order derivatives (grad of grad ... of grad).
    """
    fn = pyir_func
    for _ in range(order):
        fn = fn.grad if hasattr(fn, 'grad') else fn
    return fn
