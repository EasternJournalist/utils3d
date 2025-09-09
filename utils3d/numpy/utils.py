import numpy as np
from numpy import ndarray
import scipy.sparse as sp
from scipy.sparse import csr_array
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
    'segment_roll',
    'csr_matrix_from_indices'
]


def sliding_window(
    x: ndarray, 
    window_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    pad_size: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    pad_mode: str = 'constant',
    pad_value: Number = 0,
    axis: Optional[Tuple[int,...]] = None
) -> ndarray:
    """
    Get a sliding window of the input array.
    This function is a wrapper of `numpy.lib.stride_tricks.sliding_window_view` with additional support for padding and stride.

    ## Parameters
    - `x` (ndarray): Input array.
    - `window_size` (int or Tuple[int,...]): Size of the sliding window. If int
        is provided, the same size is used for all specified axes.
    - `stride` (Optional[Tuple[int,...]]): Stride of the sliding window. If None,
        no stride is applied. If int is provided, the same stride is used for all specified axes.
    - `pad_size` (Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]]): Size of padding to apply before sliding window.
        Corresponding to `axis`.
        - General format is `((before_1, after_1), (before_2, after_2), ...)`.
        - Shortcut formats: 
            - `int` -> same padding before and after for all axes;
            - `(int, int)` -> same padding before and after for each axis;
            - `((int,), (int,) ...)` -> specify padding for each axis, same before and after.
    - `pad_mode` (str): Padding mode to use. Refer to `numpy.pad` for more details.
    - `pad_value` (Union[int, float]): Value to use for constant padding. Only used
        when `pad_mode` is 'constant'.
    - `axis` (Optional[Tuple[int,...]]): Axes to apply the sliding window. If None, all axes are used.

    ## Returns
    - (ndarray): Sliding window of the input array. 
        - If no padding, the output is a view of the input array with zero copy.
        - Otherwise, the output is no longer a view but a copy of the padded array.
    """
    # Process axis
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    if isinstance(window_size, int):
        window_size = (window_size,) * len(axis)
    
    # Pad the input array if needed
    if pad_size is not None:
        if isinstance(pad_size, int):
            pad_size = ((pad_size, pad_size),) * len(axis)
        elif isinstance(pad_size, tuple) and len(pad_size) == 2 and all(isinstance(p, int) for p in pad_size):
            pad_size = (pad_size,) * len(axis)
        elif isinstance(pad_size, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size):
            if len(pad_size) == 1:
                pad_size = pad_size * len(axis)
            else:
                assert len(pad_size) == len(axis), f"pad_size {pad_size} must match the number of axes {len(axis)}"
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in axis else pad_size[axis.index(i)] for i in range(x.ndim)]
        if pad_mode == 'constant':
            x = np.pad(x, full_pad, mode=pad_mode, constant_values=pad_value)
        else:
            x = np.pad(x, full_pad, mode=pad_mode)
    
    # Apply sliding window
    x = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=axis)

    # Apply stride if needed
    if stride is not None:
        if isinstance(stride, int):
            stride = (stride,) * len(axis)
        stride_slice = tuple(slice(None) if i not in axis else slice(None, None, stride[axis.index(i)]) for i in range(x.ndim))
        x = x[stride_slice]

    return x


def max_pool_1d(x: ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == 'f' else np.iinfo(x.dtype).min
        padding_arr = np.full((*x.shape[:axis], padding, *x.shape[axis + 1:]), fill_value=fill_value, dtype=x.dtype)
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = sliding_window(x, kernel_size, stride, axis)
    max_pool = np.nanmax(a_sliding, axis=-1)
    return max_pool


def max_pool_nd(x: ndarray, kernel_size: Tuple[int,...], stride: Tuple[int,...], padding: Tuple[int,...], axis: Tuple[int,...]) -> ndarray:
    for i in range(len(axis)):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], axis[i])
    return x


def max_pool_2d(x: ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return max_pool_nd(x, kernel_size, stride, padding, axis)


def lookup(key: ndarray, query: ndarray, value: Optional[ndarray] = None, default_value: Union[Number, ndarray] = 1) -> ndarray:
    """
    Look up `query` in `key` like a dictionary.

    ## Parameters
        `key` (ndarray): shape `(K, *qk_shape)`, the array to search in
        `query` (ndarray): shape `(Q, *qk_shape)`, the array to search for
        `value` (Optional[ndarray]): shape `(K, *v_shape)`, the array to get values from
        `default_value` (Optional[ndarray]): shape `(*v_shape)`, default values to return if query is not found

    ## Returns
        If `value` is None, return the indices `(Q,)` of `query` in `key`, or -1. If a query is not found in key, the corresponding index will be -1.
        If `value` is provided, return the corresponding values `(Q, *v_shape)`, or default_value if not found.

    ## NOTE
    `O((Q + K) * log(Q + K))` complexity.
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
    return np.where(result < key.shape[0], value[result.clip(0, key.shape[0] - 1)], default_value)


def segment_roll(data: ndarray, offsets: ndarray, shift: int) -> ndarray:
    """Roll the data tensor within each segment defined by offsets.
    """
    lengths = offsets[1:] - offsets[:-1]
    start = np.repeat(offsets[:-1], lengths)
    elem_indices = start + (np.arange(data.shape[0], dtype=offsets.dtype) - start - shift) % np.repeat(lengths, lengths)
    data = data[elem_indices]
    return data


def csr_matrix_from_indices(indices: ndarray, n_cols: int) -> csr_array:
    """Convert a regular indices array to a sparse CSR adjacency matrix format

    ## Parameters
        - `indices` (ndarray): shape (N, M) dense tensor. Each one in `N` has `M` connections.
        - `n_cols` (int): total number of columns in the adjacency matrix

    ## Returns
        Tensor: shape `(N, n_cols)` sparse CSR adjacency matrix
    """
    return csr_array((
        np.ones_like(indices, dtype=bool).ravel(), 
        indices.ravel(),
        np.arange(0, indices.size + 1, indices.shape[1])
    ), shape=(indices.shape[0], n_cols))