import torch


__all__ = ["FastGlobalAvgPool2d", "fast_global_avg_pool_2d", "fast_global_max_pool_2d", "fast_concat_pool_2d"]


@torch.jit.script
class FastGlobalAvgPool2d(object):
    """
    JIT-ed global average pooling.
    """
    def __init__(self, flatten=True):
        self.flatten = flatten

    def __call__(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h * w).mean(dim=2)
        if self.flatten:
            return x
        else:
            return x.view(n, c, 1, 1)


@torch.jit.script
def fast_global_avg_pool_2d(
    x,
    flatten :bool = True):
    """
    JIT-ed global average pooling function.
    """
    n, c, h, w = x.size()
    x = x.view(n, c, h * w).mean(dim=2)
    if flatten:
        return x
    return x.view(n, c, 1, 1)


@torch.jit.script
def fast_global_max_pool_2d(
    x,
    flatten :bool = True):
    """
    JIT-ed global max pooling function.
    """
    n, c, h, w = x.size()
    x = x.view(n, c, h * w).max(dim=2)[0]
    if flatten:
        return x
    return x.view(n, c, 1, 1)


@torch.jit.script
def fast_concat_pool_2d(
    x,
    flatten :bool = True):
    """
    Concatenate global average + max pooling.
    """
    avg_pool = fast_global_avg_pool_2d(x, flatten)
    max_pool = fast_global_max_pool_2d(x, flatten)
    return torch.cat((avg_pool, max_pool), 1)