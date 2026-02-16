import itertools

import numpy as np


def itertools_batched(iterable, n, *, strict=False):
    """
    itertools.batched was added in version 3.12.
    Use the "roughly equivalent" from itertools documentation for now.
    """

    # batched('ABCDEFG', 2) → AB CD EF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

def asarray(a, dtype):
    if hasattr(a, "__len__"):
        return np.asarray(a, dtype = dtype)
    else:
        return np.fromiter(a, dtype = dtype)