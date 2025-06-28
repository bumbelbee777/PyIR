import re

from ..core.ir import IRFunction, IRBlock, IRInstr

def _parse_ir_to_objects(ir_str, ssa_uid, func_name=None):
    """
    Parse IR string into IR objects with SSA renaming.
    Returns (IRFunction, output_vars, output_names).
    If func_name is given, extract that function; otherwise, use the first.
    Uses a brace-counting parser for robustness.
    """
    import re
    # Find all function definitions using brace counting
    lines = ir_str.splitlines()
    functions = []
    in_func = False
    func_lines = []
    brace_count = 0
    func_header = None
    for line in lines:
        if not in_func and line.strip().startswith('define'):
            in_func = True
            func_lines = [line]
            brace_count = line.count('{') - line.count('}')
            func_header = line
        elif in_func:
            func_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                # End of function
                func_block = '\n'.join(func_lines)
                # Extract function name
                m = re.match(r'define\s+[^@]+@([^\(]+)\(', func_header.strip())
                name = m.group(1).strip() if m else None
                functions.append((name, func_block))
                in_func = False
                func_lines = []
                func_header = None
    if not functions:
        raise ValueError(f"[pyir.fusion] Could not parse function definition from IR")
    # Pick the function by name if possible
    if func_name:
        for n, block in functions:
            if n == func_name:
                func_ir = block
                break
        else:
            func_ir = functions[0][1]
    else:
        func_ir = functions[0][1]
    # Now parse the function signature and body as before
    sig_match = re.search(r'define\s+([^{@]+)@([^\(]+)\(([^\)]*)\)[^{]*\{', func_ir)
    if not sig_match:
        print(f"[pyir.fusion] Could not parse function signature from IR block:\n{func_ir}")
        raise ValueError(f"[pyir.fusion] Could not parse function signature from IR")
    ret_type = sig_match.group(1).strip()
    name = sig_match.group(2).strip()
    args_str = sig_match.group(3).strip()
    # Parse arguments
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
    # Parse body
    body_match = re.search(r'\{([\s\S]*?)\}$', func_ir, re.MULTILINE)
    body = body_match.group(1) if body_match else ''
    # Build IRFunction and return
    ir_fn = IRFunction(name, args, ret_type)
    block = IRBlock('entry')
    for line in body.splitlines():
        line = line.strip()
        if line:
            block.add(IRInstr(line))
    ir_fn.add_block(block)
    return ir_fn, [], []

def _merge_ir_functions(functions, ssa_uid):
    """
    Merge multiple IRFunction objects into a single fused function.
    Returns (IRFunction, output_vars, output_names).
    """
    if not functions:
        raise ValueError("[pyir.fusion] No functions to merge")
    
    # Use the first function as the base for signature
    base_fn = functions[0]
    merged_fn = IRFunction(
        f"fused_{ssa_uid}",
        base_fn.args,
        base_fn.ret_type,
        base_fn.attrs
    )
    
    output_vars = []
    output_names = []
    
    # Merge all blocks from all functions
    for i, fn in enumerate(functions):
        for block in fn.blocks:
            # Rename block labels to avoid conflicts
            if block.label == 'entry' and i > 0:
                block.label = f'entry_{i}'
            merged_fn.add_block(block)
            
            # Collect output variables from return instructions
            for instr in block.instrs:
                ret_match = re.search(r'ret\s+[^%]*%([a-zA-Z_][a-zA-Z0-9_]+)', str(instr))
                if ret_match:
                    outvar = f"%{ret_match.group(1)}"
                    output_vars.append(outvar)
                    output_names.append(f"out_{i}")
    
    return merged_fn, output_vars, output_names