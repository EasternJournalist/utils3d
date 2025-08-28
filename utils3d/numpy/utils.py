import numpy as np
from typing import *
from numbers import Number
import warnings
import functools


__all__ = [
    'sliding_window_1d',
    'sliding_window_nd',
    'sliding_window_2d',
    'max_pool_1d',
    'max_pool_2d',
    'max_pool_nd',
    'interpolate',
    'lookup',
]


def sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    """
    Return x view of the input array with x sliding window of the given kernel size and stride.
    The sliding window is performed over the given axis, and the window dimension is append to the end of the output array's shape.

    Args:
        x (np.ndarray): input array with shape (..., axis_size, ...)
        kernel_size (int): size of the sliding window
        stride (int): stride of the sliding window
        axis (int): axis to perform sliding window over
    
    Returns:
        a_sliding (np.ndarray): view of the input array with shape (..., n_windows, ..., kernel_size), where n_windows = (axis_size - kernel_size + 1) // stride
    """
    assert x.shape[axis] >= window_size, f"kernel_size ({window_size}) is larger than axis_size ({x.shape[axis]})"
    axis = axis % x.ndim
    shape = (*x.shape[:axis], (x.shape[axis] - window_size + 1) // stride, *x.shape[axis + 1:], window_size)
    strides = (*x.strides[:axis], stride * x.strides[axis], *x.strides[axis + 1:], x.strides[axis])
    x_sliding = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return x_sliding


def sliding_window_nd(x: np.ndarray, window_size: Tuple[int,...], stride: Tuple[int,...], axis: Tuple[int,...]) -> np.ndarray:
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x


def sliding_window_2d(x: np.ndarray, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, axis)


def max_pool_1d(x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == 'f' else np.iinfo(x.dtype).min
        padding_arr = np.full((*x.shape[:axis], padding, *x.shape[axis + 1:]), fill_value=fill_value, dtype=x.dtype)
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = sliding_window_1d(x, kernel_size, stride, axis)
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



def interpolate(bary: np.ndarray, tri_id: np.ndarray, attr: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Interpolate with given barycentric coordinates and triangle indices

    Args:
        bary (np.ndarray): shape (..., 3), barycentric coordinates
        tri_id (np.ndarray): int array of shape (...), triangle indices
        attr (np.ndarray): shape (N, M), vertices attributes
        faces (np.ndarray): int array of shape (T, 3), face vertex indices

    Returns:
        np.ndarray: shape (..., M) interpolated result
    """
    faces_ = np.concatenate([np.zeros((1, 3), dtype=faces.dtype), faces + 1], axis=0)
    attr_ = np.concatenate([np.zeros((1, attr.shape[1]), dtype=attr.dtype), attr], axis=0)
    return np.sum(bary[..., None] * attr_[faces_[tri_id + 1]], axis=-2)


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

