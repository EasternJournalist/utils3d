from typing import *
from numbers import Number
from itertools import chain
from numbers import Integral

import torch
from torch import Tensor
import torch.nn.functional as F

from .helpers import batched
from ..helpers import no_warnings


__all__ = [
    'segment_roll',
    'segment_take',
    'segment_argmax',
    'segment_argmin',
    'group_as_segments'
]



def segment_roll(data: torch.Tensor, offsets: torch.Tensor, shift: int) -> Tensor:
    """Roll the data within each segment.
    """
    lengths = torch.diff(offsets)
    start = offsets[:-1].repeat_interleave(lengths)
    elem_indices = start + (torch.arange(data.shape[0], dtype=offsets.dtype) - start - shift) % lengths.repeat_interleave(lengths)
    data = data.gather(0, elem_indices)
    return data


def segment_take(data: Tensor, offsets: Tensor, taking: Tensor) -> Tuple[Tensor, Tensor]:
    """Take some segments from a segmented array
    """
    lengths = torch.diff(offsets)
    new_lengths = lengths[taking]
    new_offsets = torch.cat([torch.tensor([0], dtype=lengths.dtype, device=lengths.device), torch.cumsum(new_lengths, dim=0)])
    indices = torch.arange(new_offsets[-1]) + torch.repeat_interleave(offsets[taking] - new_offsets[:-1], new_lengths)
    new_data = data.index_select(0, indices)
    return new_data, new_offsets


def group_as_segments(labels: Tensor, data: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Group as segments by labels

    ## Parameters
    
    - `labels` (Tensor): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
    - `data` (Tensor, optional): shape `(N, *data_dims)` array.
        If None, return the indices in each group instead.

    ## Returns

    Assuming there are `M` difference labels:

    - `segment_labels`: `(Tensor)` shape `(M, *label_dims)` labels of of each segment
    - `data`: `(Tensor)` shape `(N,)` or `(N, *data_dims)` the rearranged data (or indices) where the same labels are grouped as a continous segment.
    - `offsets`: `(Tensor)` shape `(M + 1,)`
    
    `data[offsets[i]:offsets[i + 1]]` corresponding to the i-th segment whose label is `segment_labels[i]`
    """
    group_labels, inv, counts = torch.unique(labels, return_inverse=True, return_counts=True, dim=0)
    if data is None:
        data = torch.arange(labels.shape[0], device=labels.device)
    offsets = torch.cat([torch.tensor([0], dtype=counts.dtype, device=counts.device), torch.cumsum(counts, dim=0)])
    data = data[torch.argsort(inv)]
    return group_labels, data, offsets


def segment_argmax(data: Tensor, offsets: Tensor) -> Tensor:
    """Compute the argmax of each segment in the segmented data.

    ## Parameters
    - `data`: (Tensor) shape `(N, ...)` the data to compute argmax from. If `data` may have multiple dimensionsm, extra dimensions are treated as batch dimensions.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of the segmented data

    ## Returns
    - `argmax_indices`: (Tensor) shape `(M, ...)` the argmax indices of each segment along the first dimension.
    NOTE: If there are multiple maximum values in a segment, the index of the first one is returned.
    """
    seg_maxs = torch.segment_reduce(data, 'max', offsets=offsets, axis=0)
    lengths = torch.diff(offsets)
    is_max_mask = data == torch.repeat_interleave(seg_maxs, lengths, dim=0)
    candidate_indices = torch.where(is_max_mask, torch.arange(data.shape[0], device=data.device)[(..., *((None,) * (data.ndim - 1)))], data.shape[0])
    argmax_indices = torch.segment_reduce(candidate_indices, 'min', offsets=offsets, axis=0)
    return argmax_indices


def segment_argmin(data: Tensor, offsets: Tensor) -> Tensor:
    """Compute the argmin of each segment in the segmented data.

    ## Parameters
    - `data`: (Tensor) shape `(N, ...)` the data to compute argmin from. If `data` may have multiple dimensionsm, extra dimensions are treated as batch dimensions.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of the segmented data

    ## Returns
    - `argmin_indices`: (Tensor) shape `(M, ...)` the argmin indices of each segment along the first dimension.
    NOTE: If there are multiple minimum values in a segment, the index of the first one is returned.
    """
    seg_mins = torch.segment_reduce(data, 'min', offsets=offsets, axis=0)
    lengths = torch.diff(offsets)
    is_min_mask = data == torch.repeat_interleave(seg_mins, lengths, dim=0)
    candidate_indices = torch.where(is_min_mask, torch.arange(data.shape[0], device=data.device)[(..., *((None,) * (data.ndim - 1)))], data.shape[0])
    argmin_indices = torch.segment_reduce(candidate_indices, 'min', offsets=offsets, axis=0)
    return argmin_indices
