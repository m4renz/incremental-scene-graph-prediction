from typing import Sequence, Any
from numbers import Number
import numpy as np

def random_drop(x: Sequence[Any], p: Number | tuple[Number, Number], replace: bool = False):
    '''
    random drop p percentage of the input list.
    if p is int and >0, select random p elements.
    if p is float/double, >0 and <=1.0, random select p percentage #in [1-p, 1.0]
    '''
    l = len(x)
    if isinstance(p, Sequence):
        low, high = p
        assert low < high

        if isinstance(low, int):
            if not replace:
                if low >= l: return x
                if high > l: high = l
            select_p = np.random.randint(low, high, 1)[0]
            choices  = np.random.choice(l, select_p, replace=replace).tolist()    
            x = [x[y] for y in choices]
        else:
            assert low >= 0
            assert high <= 1
            percentage = np.random.uniform(low=low, high=high, size=1)[0]
            num_edge = int(len(x) * percentage / 1) or 1
            assert len(x) > 0
            assert num_edge > 0
            choices = np.random.choice(len(x), num_edge, replace=replace).tolist()

        x = [x[y] for y in choices]

    elif p > 0:
        if isinstance(p, int):
            if p < l or replace: # if p >= n_elms, just return the same input
                choices = np.random.choice(l, p, replace=replace).tolist()
                x = [x[y] for y in choices]
        else:
            percentage = p
            num_edge = int(len(x) * percentage / 1) or 1
            assert len(x) > 0
            assert num_edge > 0
            choices = np.random.choice(len(x), num_edge, replace=replace).tolist()
            x = [x[y] for y in choices]
    return x