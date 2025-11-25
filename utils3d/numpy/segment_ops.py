import numpy as np
from numpy import ndarray
from typing import *
from numbers import Number, Integral
import warnings
import functools


__all__ = [
    'segment_roll',
    'segment_take',
    'segment_argmax',
    'segment_argmin',
    'segment_concatenate',
    'group_as_segments'
]



def segment_roll(data: ndarray, offsets: ndarray, shift: int) -> ndarray:
    """Roll the data within each segment.
    """
    lengths = np.diff(offsets)
    start = np.repeat(offsets[:-1], lengths)
    elem_indices = start + (np.arange(data.shape[0], dtype=offsets.dtype) - start - shift) % np.repeat(lengths, lengths)
    data = data[elem_indices]
    return data


def segment_take(data: ndarray, offsets: ndarray, taking: ndarray) -> Tuple[ndarray, ndarray]:
    """Take some segments from a segmented array

    ## Parameters
    - `data`: (ndarray) shape `(N, *data_dims)` the data to take segments from
    - `offsets`: (ndarray) shape `(M + 1,)` the offsets of the segmented data
    - `taking`: (ndarray) the indices of segments to take of shape `(K,)`, or boolean mask of shape `(M,)`

    ## Returns
    - `new_data`: (ndarray) shape `(N_new, *data_dims)`
    - `new_offsets`: (ndarray) shape `(K + 1,)` the offsets of the new segmented data
    """
    lengths = np.diff(offsets)
    new_lengths = lengths[taking]
    new_offsets = np.concatenate([[0], np.cumsum(new_lengths)])
    indices = np.arange(new_offsets[-1]) + np.repeat(offsets[taking] - new_offsets[:-1], new_lengths)
    new_data = data[indices]
    return new_data, new_offsets


def segment_concatenate(segments: List[Tuple[ndarray, ndarray]]) -> Tuple[ndarray, ndarray]:
    """Concatenate a list of segmented arrays into a single segmented array

    ## Parameters
    - `segments`: (List[Tuple[ndarray, ndarray]]) list of segmented arrays to concatenate.
        Each element is a tuple of `(data, offsets)`:
        - `data`: (ndarray) shape `(N_i, *data_dims)` the
        - `offsets`: (ndarray) shape `(M_i + 1,)` the offsets of the segmented data

    ## Returns
    - `data`: (ndarray) shape `(N, *data_dims)` the concatenated data
    - `offsets`: (ndarray) shape `(M + 1,)` the offsets of the concatenated segmented data
    """
    data_list = []
    offsets_list = [np.array([0])]
    for data, offsets in segments:
        if len(offsets) > 1:
            data_list.append(data)
            offsets_list.append(offsets[1:] + offsets_list[-1][-1])
    data = np.concatenate(data_list, axis=0)
    offsets = np.concatenate(offsets_list, axis=0)
    return data, offsets


def group_as_segments(labels: ndarray, data: Optional[np.ndarray] = None) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Group as segments by labels

    ## Parameters
    - `labels` (ndarray): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
    - `data` (ndarray, optional): shape `(N, *data_dims)` array.
        If None, return the indices in each group instead.

    ## Returns
    Assuming there are `M` difference labels:

    - `segment_labels`: `(ndarray)` shape `(M, *label_dims)` labels of of each segment
    - `rearranged_data`: `(ndarray)` shape `(N,)` or `(N, *data_dims)` the rearranged data (or indices) where the same labels are grouped as a continous segment.
    - `offsets`: `(ndarray)` shape `(M + 1,)`
    
    `rearranged_data[offsets[i]:offsets[i + 1]]` corresponding to the i-th segment whose label is `segment_labels[i]`
    """
    group_labels, inv, counts = np.unique(labels, return_inverse=True, return_counts=True, axis=0)
    offsets = np.concatenate([[0], np.cumsum(counts, axis=0)])
    if data is None:
        data = np.argsort(inv)
    else:
        data = data[np.argsort(inv)]
    return group_labels, data, offsets


def segment_argmax(data: ndarray, offsets: ndarray) -> ndarray:
    """Compute the argmax of each segment in the segmented data.

    ## Parameters
    - `data`: (ndarray) shape `(N, ...)` the data to compute argmax from. If `data` may have multiple dimensionsm, extra dimensions are treated as batch dimensions.
    - `offsets`: (ndarray) shape `(M + 1,)` the offsets of the segmented data

    ## Returns
    - `argmax_indices`: (ndarray) shape `(M, ...)` the argmax indices of each segment along the first dimension.
    NOTE: If there are multiple maximum values in a segment, the index of the first one is returned.
    """
    seg_maxs = np.maximum.reduceat(data, offsets[:-1], axis=0)
    lengths = np.diff(offsets)
    is_max_mask = data == np.repeat(seg_maxs, lengths, axis=0)
    candidate_indices = np.where(is_max_mask, np.arange(data.shape[0])[(..., *((None,) * (data.ndim - 1)))], data.shape[0])
    argmax_indices = np.minimum.reduceat(candidate_indices, offsets[:-1], axis=0)
    return argmax_indices


def segment_argmin(data: ndarray, offsets: ndarray) -> ndarray:
    """Compute the argmin of each segment in the segmented data.

    ## Parameters
    - `data`: (ndarray) shape `(N, ...)` the data to compute argmin from. If `data` may have multiple dimensionsm, extra dimensions are treated as batch dimensions.
    - `offsets`: (ndarray) shape `(M + 1,)` the offsets of the segmented data

    ## Returns
    - `argmin_indices`: (ndarray) shape `(M, ...)` the argmin indices of each segment along the first dimension.
    NOTE: If there are multiple minimum values in a segment, the index of the first one is returned.
    """
    seg_mins = np.minimum.reduceat(data, offsets[:-1], axis=0)
    lengths = np.diff(offsets)
    is_min_mask = data == np.repeat(seg_mins, lengths, axis=0)
    candidate_indices = np.where(is_min_mask, np.arange(data.shape[0])[(..., *((None,) * (data.ndim - 1)))], data.shape[0])
    argmin_indices = np.minimum.reduceat(candidate_indices, offsets[:-1], axis=0)
    return argmin_indices
