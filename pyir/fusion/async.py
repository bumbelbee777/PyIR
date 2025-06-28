import asyncio

def make_async_fused_fn(fns, name, outputs=None, output_names=None, return_type='tuple'):
    async def fused_fn(*args, out=None):
        # Await all sub-kernels if needed
        results = []
        for fn in fns:
            res = fn(*args)
            if asyncio.iscoroutine(res) or hasattr(res, '__await__'):
                res = await res
            results.append(res)
        if outputs is not None:
            results = [results[i] for i in outputs]
            names = [output_names[i] for i in outputs] if output_names else [f"out{i}" for i in outputs]
        else:
            names = output_names if output_names else [f"out{i}" for i in range(len(results))]
        if return_type == 'namedtuple':
            import collections
            NT = collections.namedtuple(f"FusedOutputs_{name}", names)
            return NT(*results)
        elif return_type == 'dict':
            return {n: v for n, v in zip(names, results)}
        if len(results) == 1:
            return results[0]
        return tuple(results)
    fused_fn._is_async_kernel = True
    return fused_fn
