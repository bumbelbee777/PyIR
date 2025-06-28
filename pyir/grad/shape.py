_custom_shape_inference = {}

def register_custom_shape_inference(opname, shape_fn):
    """Register a custom shape inference function for a given IR op name.
    shape_fn(lhs, rhs, shapes, arg_shapes) -> output_shape
    """
    _custom_shape_inference[opname] = shape_fn 