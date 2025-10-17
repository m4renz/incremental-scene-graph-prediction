from __future__ import annotations
import numpy as np
from typing import Sequence

def is_structured_dtype(dtype: np.dtype) -> bool:
    """
    True, if the given np.dtype object represents a structured dtype.
    """
    return getattr(dtype, 'fields', None) is not None


def concat_dtypes(*dtypes: np.dtype) -> np.dtype:
    """
    Concatenate the given structured dtypes into a combined representation.
    
    Args:
        dtypes: The structured dtypes to concatenate
    Returns:
        the concatenated dtypes
    """

    import numpy.lib.recfunctions as recf
    dt = []
    for dtype in dtypes:
        if not is_structured_dtype(dtype):
            raise ValueError("Structured dtypes required")
        dt.extend(recf._get_fieldspec(dtype))
    return np.dtype(dt)


def _join_record_arrays(*arrays: np.ndarray, fused_dtype: np.dtype = None) -> np.ndarray:
    """
    Join the structured arrays by stacking the fields in it's dtypes. All arrays to be joined 
    must have the same shape.
                                
    Args:
        arrays: The structured arrays to join. 
    Returns:
        The joined array
    """

    if fused_dtype is None:
        raise ValueError("Fused dtype must be given.")
    
    sizes = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum(axis=0)]

    shape = arrays[0].shape + (offsets[-1],)

    joint = np.empty(shape, dtype=np.uint8)

    for a, size, offset in zip(arrays, sizes, offsets):
         joint[..., offset:offset+size] = a[..., np.newaxis].view(np.uint8)
    joint = joint.view(fused_dtype).reshape(arrays[0].shape)
    return joint


def join_record_arrays(arrays: Sequence[np.ndarray]) -> np.ndarray:
    dt = concat_dtypes(*(a.dtype for a in arrays))
    return _join_record_arrays(*arrays, fused_dtype=dt)