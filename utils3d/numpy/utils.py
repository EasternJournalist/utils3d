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
    'pooling',
    'max_pool_2d',
    'lookup',
    'segment_roll',
    'csr_matrix_from_dense_indices',
    'split_groups_by_labels'
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
    Get a sliding window of the input array. Window axis(axes) will be appended as the last dimension(s).
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


def pooling(
    x: ndarray, 
    kernel_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    padding: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    mode: Literal['min', 'max', 'sum', 'mean'] = 'max'
) -> ndarray:
    """Compute the pooling of the input array. 
    NOTE: NaNs will be ignored.

    ## Parameters
        - `x` (ndarray): Input array.
        - `kernel_size` (int or Tuple[int,...]): Size of the pooling window.
        - `stride` (Optional[Tuple[int,...]]): Stride of the pooling window. If None,
            no stride is applied. If int is provided, the same stride is used for all specified axes.
        - `padding` (Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]]): Size of padding to apply before pooling.
            Corresponding to `axis`.
            - General format is `((before_1, after_1), (before_2, after_2), ...)`.
            - Shortcut formats: 
                - `int` -> same padding before and after for all axes;
                - `(int, int)` -> same padding before and after for each axis;
                - `((int,), (int,) ...)` -> specify padding for each axis, same before and after.
        - `axis` (Optional[Tuple[int,...]]): Axes to apply the pooling. If None, all axes are used.
        - `mode` (str): Pooling mode. One of 'min', 'max', 'sum', 'mean'.

    ## Returns
        - (ndarray): Pooled array with the same number of dimensions as input array.
    """
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * len(axis)
    if not isinstance(stride, tuple):
        stride = (stride,) * len(axis)
    if padding is not None:
        if isinstance(padding, int):
            padding = ((padding, padding),) * len(axis)
        elif isinstance(padding, tuple) and len(padding) == 2 and all(isinstance(p, int) for p in padding):
            padding = (padding,) * len(axis)
        elif isinstance(padding, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in padding):
            if len(padding) == 1:
                padding = padding * len(axis)
            else:
                assert len(padding) == len(axis), f"padding {padding} must match the number of axes {len(axis)}"
        else:
            raise ValueError(f"Invalid padding {padding}")
    else:
        padding = ((0, 0),) * len(axis)

    if mode == 'max':
        pad_mode = 'constant'
        pad_value = -np.inf if x.dtype.kind == 'f' else np.iinfo(x.dtype).min
        pool_fn = np.nanmax
    elif mode == 'min':
        pad_mode = 'constant'
        pad_value = np.inf if x.dtype.kind == 'f' else np.iinfo(x.dtype).max
        pool_fn = np.nanmin
    elif mode == 'sum':
        pad_mode = 'constant'
        pad_value = 0
        pool_fn = np.sum
        x = np.where(np.isnan(x), 0, x)
    elif mode == 'mean':
        mask = ~np.isnan(x)
        full_pad = [(0, 0) if i not in axis else padding[axis.index(i)] for i in range(x.ndim)]
        x = pooling(np.pad(x, full_pad, mode='edge'), kernel_size, stride, axis=axis, mode='sum')
        x /= pooling(np.pad(mask, full_pad, mode='edge'), kernel_size, stride, axis=axis, mode='sum')
        return x
    else:
        raise ValueError(f"Invalid pooling mode {mode}. Supported modes are 'min', 'max', 'sum', 'mean'.")

    for i in range(len(axis)):
        x = pool_fn(
            sliding_window(x, kernel_size[i], stride[i], 
                           pad_size=padding[i], pad_mode=pad_mode, pad_value=pad_value, 
                           axis=axis[i]), 
            axis=-1
        )
    return x


def max_pool_2d(x: ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return pooling(x, kernel_size, stride, padding, axis, 'max')


def lookup(key: ndarray, query: ndarray, value: Optional[ndarray] = None, default_value: Union[Number, ndarray] = 0) -> ndarray:
    """
    Look up `query` in `key` like a dictionary.  Useful for COO indexing.

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
    unsqueeze_slicing = tuple(slice(None), *((None,) * (value.ndim - 1)))
    return np.where((result < key.shape[0])[unsqueeze_slicing], value[result.clip(0, key.shape[0] - 1)], default_value)


def segment_roll(data: ndarray, offsets: ndarray, shift: int) -> ndarray:
    """Roll the data tensor within each segment defined by offsets.
    """
    lengths = offsets[1:] - offsets[:-1]
    start = np.repeat(offsets[:-1], lengths)
    elem_indices = start + (np.arange(data.shape[0], dtype=offsets.dtype) - start - shift) % np.repeat(lengths, lengths)
    data = data[elem_indices]
    return data


def csr_matrix_from_dense_indices(indices: ndarray, n_cols: int) -> csr_array:
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


def split_groups_by_labels(labels: ndarray, data: Optional[np.ndarray] = None) -> List[Tuple[ndarray, ndarray]]:
    """
    Split the data into groups based on the provided labels.

    ## Parameters
        - `labels` (ndarray): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
        - `data` (ndarray, optional): shape `(N, *data_dims)` dense tensor. Each one in `N` has `D` features.
            If None, return the indices in each group instead.

    ## Returns
        - groups (List[Tuple[ndarray, ndarray]]): List of length G. Each element is a tuple of (label, data_in_group).
            - `label` (ndarray): shape (*label_dims,) the label of the group.
            - `data_in_group` (ndarray): shape (M, *data_dims) the data points in the group.
            If `data` is None, `data_in_group` will be the indices of the data points in the original array.
    """
    group_labels, inv, counts = np.unique(labels, return_inverse=True, return_counts=True, axis=0)
    if data is None:
        data = np.arange(labels.shape[0])
    sections = np.cumsum(counts, axis=0)[:-1]
    data_groups = np.split(data[np.argsort(inv)], sections)
    return list(zip(group_labels, data_groups))