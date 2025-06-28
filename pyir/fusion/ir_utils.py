import re

from ..core.ir import IRFunction, IRBlock, IRInstr

def _parse_ir_to_objects(ir_str, ssa_uid):
    """Parse IR string to IR objects, handling both strings and IRFunction objects. Applies SSA variable renaming with ssa_uid to avoid conflicts when fusing."""
    if hasattr(ir_str, 'blocks'):  # It's already an IRFunction object
        ir_fn = ir_str
    else:
        # It's a string, parse it
        from ..core.ir import create_ir_function_from_string
        ir_fn = create_ir_function_from_string(ir_str)

    # Collect all argument names (do not rename these)
    arg_names = set(n for n, _ in ir_fn.args)
    # Build a mapping from old SSA names to new ones
    ssa_map = {}
    for block in ir_fn.blocks:
        for instr in block.instrs:
            s = str(instr)
            # Match assignments: %var = ...
            m = re.match(r'(%[a-zA-Z_][a-zA-Z0-9_]*)\s*=.*', s)
            if m:
                var = m.group(1)
                if var[1:] not in arg_names:
                    ssa_map[var] = f"%{var[1:]}_{ssa_uid}"
    # Now, rewrite all instructions in all blocks
    for block in ir_fn.blocks:
        new_instrs = []
        for instr in block.instrs:
            s = str(instr)
            # Replace all SSA variable names (LHS and RHS) except arguments
            for old, new in ssa_map.items():
                s = re.sub(rf'(?<![a-zA-Z0-9_]){re.escape(old)}(?![a-zA-Z0-9_])', new, s)
            new_instrs.append(IRInstr(s))
        block.instrs = new_instrs
    # Extract output variables from the return instruction
    output_vars = []
    output_names = []
    for block in ir_fn.blocks:
        for instr in block.instrs:
            if str(instr).startswith('ret '):
                ret_instr = str(instr)
                if 'ret void' not in ret_instr:
                    match = re.search(r'ret [^{}]+ (%[a-zA-Z_][a-zA-Z0-9_]*(?:_[a-f0-9]+)?)', ret_instr)
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