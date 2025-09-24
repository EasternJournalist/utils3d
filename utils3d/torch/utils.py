from typing import *
from numbers import Number
from itertools import chain

import torch
from torch import Tensor
import torch.nn.functional as F

from .helpers import batched
from ..helpers import no_warnings


__all__ = [
    'sliding_window',
    'masked_min',
    'masked_max',
    'lookup',
    'segment_roll',
    'csr_matrix_from_dense_indices',
    'csr_eliminate_zeros',
    'split_groups_by_labels'
]


def sliding_window(
    x: Tensor, 
    window_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    pad_size: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    pad_mode: str = 'constant',
    pad_value: Number = 0,
    dim: Tuple[int, ...] = None
) -> Tensor:
    """
    Get a sliding window of the input array.
    This function is a wrapper of `torch.nn.functional.unfold` with additional support for padding and stride.

    ## Parameters
    - `x` (Tensor): Input tensor.
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
    - (Tensor): Sliding window of the input array. 
        - If no padding, the output is a view of the input array with zero copy.
        - Otherwise, the output is no longer a view but a copy of the padded array.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if isinstance(dim, int):
        dim = (dim,)
    dim = [dim[i] % x.ndim for i in range(len(dim))]
    if isinstance(window_size, int):
        window_size = (window_size,) * len(dim)
    if stride is None:
        stride = (1,) * len(dim)
    if isinstance(stride, int):
        stride = (stride,) * len(dim)
    assert len(window_size) == len(stride) == len(dim)

    # Pad the input array if needed
    if pad_size is not None:
        if isinstance(pad_size, int):
            pad_size = ((pad_size, pad_size),) * len(dim)
        elif isinstance(pad_size, tuple) and len(pad_size) == 2 and all(isinstance(p, int) for p in pad_size):
            pad_size = (pad_size,) * len(dim)
        elif isinstance(pad_size, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size):
            if len(pad_size) == 1:
                pad_size = pad_size * len(dim)
            else:
                assert len(pad_size) == len(dim), f"pad_size {pad_size} must match the number of axes {len(dim)}"
            pad_size = tuple(p * 2 if len(p) == 1 else p for p in pad_size)
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in dim else pad_size[dim.index(i)] for i in range(x.ndim)]
        full_pad = chain(*reversed(full_pad))
        x = F.pad(x, full_pad, mode=pad_mode, value=pad_value)
    
    for i in range(len(window_size)):
        x = x.unfold(dim[i], window_size[i], stride[i])
    return x


def masked_min(input: Tensor, mask: torch.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Similar to torch.min, but with mask
    """
    if dim is None:
        return torch.where(mask, input, torch.tensor(torch.inf, dtype=input.dtype, device=input.device)).min()
    else:
        return torch.where(mask, input, torch.tensor(torch.inf, dtype=input.dtype, device=input.device)).min(dim=dim, keepdim=keepdim)


def masked_max(input: Tensor, mask: torch.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Similar to torch.max, but with mask
    """
    if dim is None:
        return torch.where(mask, input, torch.tensor(-torch.inf, dtype=input.dtype, device=input.device)).max()
    else:
        return torch.where(mask, input, torch.tensor(-torch.inf, dtype=input.dtype, device=input.device)).max(dim=dim, keepdim=keepdim)
    

def lookup(key: Tensor, query: Tensor, value: Optional[Tensor] = None, default_value: Union[Number, Tensor] = 0) -> torch.LongTensor:
    """
    Look up `query` in `key` like a dictionary. Useful for COO indexing.

    ## Parameters
    - `key` (Tensor): shape `(K, *qk_shape)`, the array to search in
    - `query` (Tensor): shape `(Q, *qk_shape)`, the array to search for
    - `value` (Optional[Tensor]): shape `(K, *v_shape)`, the array to get values from
    - `default_value` (Optional[Tensor]): shape `(*v_shape)`, default values to return if query is not found

    ## Returns
        If `value` is None, return the indices `(Q,)` of `query` in `key`, or -1. If a query is not found in key, the corresponding index will be -1.
        If `value` is provided, return the corresponding values `(Q, *v_shape)`, or default_value if not found.

    ## NOTE
    `O((Q + K) * log(Q + K))` complexity.
    """
    unique, inverse = torch.unique(
        torch.cat([key, query], dim=0),
        dim=0,
        return_inverse=True
    )
    index = torch.full((unique.shape[0],), -1, dtype=torch.long, device=key.device)
    index.scatter_(0, inverse[:key.shape[0]], torch.arange(key.shape[0], device=key.device))
    result = index.index_select(0, inverse[key.shape[0]:])
    if value is None:
        return torch.where(result < key.shape[0], result, -1)
    unsqueeze_slicing = tuple(slice(None), *((None,) * (value.ndim - 1)))
    return torch.where((result < key.shape[0])[unsqueeze_slicing], value[result.clamp(0, key.shape[0] - 1)], default_value)


def csr_matrix_from_dense_indices(indices: Tensor, n_cols: int) -> Tensor:
    """Convert a regular indices array to a sparse CSR adjacency matrix format

    ## Parameters
        - `indices` (Tensor): shape (N, M) dense tensor. Each one in `N` has `M` connections.
        - `values` (Tensor): shape (N, M) values of the connections
        - `n_cols` (int): total number of columns in the adjacency matrix

    ## Returns
        Tensor: shape `(N, n_cols)` sparse CSR adjacency matrix
    """
    return torch.sparse_csr_tensor(
        crow_indices=torch.arange(0, indices.numel() + 1, indices.shape[1], device=indices.device),
        col_indices=indices.view(-1),
        values=torch.ones_like(indices, dtype=torch.bool).view(-1),
        size=(indices.shape[0], n_cols)
    )


def csr_eliminate_zeros(input: Tensor):
    """Remove zero elements from a sparse CSR tensor.
    """
    nonzero = input.values() != 0
    nonzero_element_indices = nonzero.nonzero(as_tuple=False).flatten()
    row_nonzero_count = torch.sparse_csr_tensor(
        input.crow_indices(), 
        input.col_indices(), 
        nonzero, 
        input.size()
    ).long().sum(dim=-1, keepdim=True).to_dense().flatten()
    crow_indices = torch.cat([torch.tensor([0], device=input.device), torch.cumsum(row_nonzero_count, dim=0)])
    col_indices = input.col_indices()[nonzero_element_indices]
    values = input.values()[nonzero_element_indices]
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, input.size())


def segment_roll(data: torch.Tensor, offsets: torch.Tensor, shift: int) -> Tensor:
    """Roll the data tensor within each segment defined by offsets.
    """
    lengths = offsets[1:] - offsets[:-1]
    start = offsets[:-1].repeat_interleave(lengths)
    elem_indices = start + (torch.arange(data.shape[0], dtype=offsets.dtype) - start - shift) % lengths.repeat_interleave(lengths)
    data = data.gather(0, elem_indices)
    return data


def csr_roll_col_indices(input: Tensor, shift: int):
    """Roll the order of column indices of a sparse CSR tensor.
    The result is mathematically equivalent to the original matrix, but with a different column order.

    ## Parameters
        - `input` (Tensor): shape `(N, M)`, sparse CSR tensor
        - `shift` (int): number of positions to shift the column indices

    ## Returns
        Tensor: shape `(N, M)` sparse CSR tensor with rolled column indices
    """
    lengths = input.crow_indices()[1:] - input.crow_indices()[:-1]
    start = input.crow_indices()[:-1].repeat_interleave(lengths)
    elem_indices = start + (torch.arange(input.col_indices().shape[0], dtype=input.col_indices().dtype) - start - shift) % lengths.repeat_interleave(lengths)
    col_indices = input.col_indices().gather(0, elem_indices)
    values = input.values().gather(0, elem_indices)
    return torch.sparse_csr_tensor(input.crow_indices(), col_indices, values, input.size())


def split_groups_by_labels(labels: Tensor, data: Optional[Tensor] = None) -> List[Tuple[Tensor, Tensor]]:
    """
    Split the data into groups based on the provided labels.

    ## Parameters
        - `labels` (Tensor): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
        - `data` (Tensor, optional): shape `(N, *data_dims)` dense tensor. Each one in `N` has `D` features.
            If None, return the indices in each group instead.

    ## Returns
        - groups (List[Tuple[Tensor, Tensor]]): List of length G. Each element is a tuple of (label, data_in_group).
            - `label` (Tensor): shape (*label_dims,) the label of the group.
            - `data_in_group` (Tensor): shape (M, *data_dims) the data points in the group.
            If `data` is None, `data_in_group` will be the indices of the data points in the original array.
    """
    group_labels, inv, counts = torch.unique(labels, return_inverse=True, return_counts=True, dim=0)
    if data is None:
        data = torch.arange(labels.shape[0], device=labels.device)
    data_groups = torch.split(data[torch.argsort(inv)], counts.tolist())
    return list(zip(group_labels, data_groups))