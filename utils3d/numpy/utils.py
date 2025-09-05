import numpy as np
from typing import *
from numbers import Number
import warnings
import functools


__all__ = [
    'sliding_window',
    'max_pool_1d',
    'max_pool_2d',
    'max_pool_nd',
    'lookup',
]


def sliding_window(x: np.ndarray, window_size: Union[int, Tuple[int,...]], stride: Optional[Tuple[int,...]] = None, axis: Optional[Tuple[int,...]] = None) -> np.ndarray:
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    if isinstance(window_size, int):
        window_size = (window_size,) * len(axis)
    x = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=axis)
    if stride is not None:
        if isinstance(stride, int):
            stride = (stride,) * len(axis)
        stride_slice = tuple(slice(None) if i not in axis else slice(None, None, stride[axis.index(i)]) for i in range(x.ndim))
        x = x[stride_slice]
    return x


def max_pool_1d(x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == 'f' else np.iinfo(x.dtype).min
        padding_arr = np.full((*x.shape[:axis], padding, *x.shape[axis + 1:]), fill_value=fill_value, dtype=x.dtype)
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = sliding_window(x, kernel_size, stride, axis)
    max_pool = np.nanmax(a_sliding, axis=-1)
    return max_pool


def max_pool_nd(x: np.ndarray, kernel_size: Tuple[int,...], stride: Tuple[int,...], padding: Tuple[int,...], axis: Tuple[int,...]) -> np.ndarray:
    for i in range(len(axis)):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], axis[i])
    return x


def max_pool_2d(x: np.ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return max_pool_nd(x, kernel_size, stride, padding, axis)


def lookup(key: np.ndarray, query: np.ndarray, value: Optional[np.ndarray] = None, default_value: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Look up `query` in `key` like a dictionary.

    ### Parameters
        `key` (np.ndarray): shape (num_keys, *query_key_shape), the array to search in
        `query` (np.ndarray): shape (num_queries, *query_key_shape), the array to search for
        `value` (Optional[np.ndarray]): shape (K, *value_shape), the array to get values from
        `default_value` (Optional[np.ndarray]): shape (*value_shape), default values to return if query is not found

    ### Returns
        If `value` is None, return the indices (num_queries,) of `query` in `key`, or -1. If a query is not found in key, the corresponding index will be -1.
        If `value` is provided, return the corresponding values (num_queries, *value_shape), or default_value if not found.
    """
    _, index, inverse = np.unique(
        np.concatenate([key, query], axis=0),
        axis=0,
        return_index=True,
        return_inverse=True
    )
    result = index[inverse[key.shape[0]:]]
    if value is None:
        return np.where(result < key.shape[0], result, -1)
    return np.where(result < key.shape[0], value[result.clip(0, key.shape[0] - 1)], default_value if default_value is not None else 0)

