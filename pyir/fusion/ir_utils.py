import re

from ..core.ir import IRFunction, IRBlock, IRInstr

def _parse_ir_to_objects(ir_str, ssa_uid):
    """Parse IR string to IR objects, handling both strings and IRFunction objects."""
    if hasattr(ir_str, 'blocks'):  # It's already an IRFunction object
        ir_fn = ir_str
        # Extract output variables from the return instruction
        output_vars = []
        output_names = []
        for block in ir_fn.blocks:
            for instr in block.instrs:
                if str(instr).startswith('ret '):
                    ret_instr = str(instr)
                    # Extract the return variable
                    if 'ret void' not in ret_instr:
                        # Find the variable being returned
                        import re
                        match = re.search(r'ret [^{}]+ %([a-zA-Z_][a-zA-Z0-9_]*)', ret_instr)
                        if match:
                            output_vars.append(match.group(1))
                            output_names.append('result')
        return ir_fn, output_vars, output_names
    else:
        # It's a string, parse it
        from ..core.ir import create_ir_function_from_string
        ir_fn = create_ir_function_from_string(ir_str)
        # Extract output variables from the return instruction
        output_vars = []
        output_names = []
        for block in ir_fn.blocks:
            for instr in block.instrs:
                if str(instr).startswith('ret '):
                    ret_instr = str(instr)
                    # Extract the return variable
                    if 'ret void' not in ret_instr:
                        # Find the variable being returned
                        import re
                        match = re.search(r'ret [^{}]+ %([a-zA-Z_][a-zA-Z0-9_]*)', ret_instr)
                        if match:
                            output_vars.append(match.group(1))
                            output_names.append('result')
        return ir_fn, output_vars, output_names

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